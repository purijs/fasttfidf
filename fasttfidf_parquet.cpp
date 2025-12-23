#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "common.hpp"

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>

#include <iostream>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <random>
#include <sys/wait.h>
#include <unistd.h>
#include <sys/stat.h>
#include <dirent.h>

namespace py = pybind11;

constexpr size_t MAX_VOCABULARY_SIZE = 10000000;
constexpr int MAX_WORKERS = 128;

// Structure to describe work for a worker process
struct WorkUnit {
    std::string filename;
    int start_row_group;
    int end_row_group;
};

struct SharedTermResult {
    uint64_t hash;
    char word[128];
    size_t doc_count;
};

uint64_t hash_bytes(const char* data, size_t len) {
    uint64_t hash = 14695981039346656037ULL;
    for (size_t i = 0; i < len; i++) {
        hash ^= static_cast<uint64_t>(data[i]);
        hash *= 1099511628211ULL;
    }
    return hash;
}

// Check if path is a directory
bool is_directory(const std::string& path) {
    struct stat sb;
    if (stat(path.c_str(), &sb) == 0) {
        return S_ISDIR(sb.st_mode);
    }
    return false;
}

// Scan directory for .parquet files
std::vector<std::string> scan_parquet_files(const std::string& dir_path) {
    std::vector<std::string> parquet_files;
    
    DIR* dir = opendir(dir_path.c_str());
    if (!dir) {
        throw std::runtime_error("Cannot open directory: " + dir_path);
    }
    
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string filename = entry->d_name;
        
        // Check if ends with .parquet
        if (filename.length() > 8 && 
            filename.substr(filename.length() - 8) == ".parquet") {
            
            std::string full_path = dir_path;
            if (full_path.back() != '/') full_path += '/';
            full_path += filename;
            
            parquet_files.push_back(full_path);
        }
    }
    
    closedir(dir);
    
    // Sort for consistent ordering
    std::sort(parquet_files.begin(), parquet_files.end());
    
    return parquet_files;
}

// Get row group count for a file
int get_row_group_count(const std::string& filename) {
    auto mmap_result = arrow::io::MemoryMappedFile::Open(
        filename, arrow::io::FileMode::READ
    );
    
    if (!mmap_result.ok()) {
        return 0;
    }
    
    auto mmap_file = mmap_result.ValueOrDie();
    
    std::unique_ptr<parquet::arrow::FileReader> reader;
    auto reader_status = parquet::arrow::OpenFile(
        mmap_file,
        arrow::default_memory_pool(),
        &reader
    );
    
    if (!reader_status.ok()) {
        return 0;
    }
    
    return reader->num_row_groups();
}

// Create work units for all files
std::vector<WorkUnit> create_work_units(const std::vector<std::string>& files) {
    std::vector<WorkUnit> work_units;
    
    for (const auto& file : files) {
        int num_row_groups = get_row_group_count(file);
        
        for (int rg = 0; rg < num_row_groups; rg++) {
            WorkUnit unit;
            unit.filename = file;
            unit.start_row_group = rg;
            unit.end_row_group = rg + 1;
            work_units.push_back(unit);
        }
    }
    
    return work_units;
}

class ParquetTfidfVectorizer {
public:
    std::unordered_map<std::string, size_t> vocabulary_;
    std::vector<double> idf_;
    size_t total_docs_ = 0;
    size_t min_df_ = 1;
    size_t max_df_ = 0;
    size_t max_features_ = 0;

    std::shared_ptr<arrow::io::MemoryMappedFile> mmap_file_;
    std::unique_ptr<parquet::arrow::FileReader> reader_;
    int current_row_group_ = 0;
    int64_t current_row_in_group_ = 0;
    std::shared_ptr<arrow::Table> current_table_;
    bool stream_open_ = false;

    void fit(const std::string& path, int num_processes = 0, size_t min_df = 1,
             size_t max_df = 0, size_t max_features = 0, bool verbose = true) {
        
        min_df_ = min_df;
        max_df_ = max_df;
        max_features_ = max_features;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Determine if path is file or directory
        std::vector<std::string> files;
        
        if (is_directory(path)) {
            files = scan_parquet_files(path);
            
            if (files.empty()) {
                throw std::runtime_error("No .parquet files found in directory: " + path);
            }
            
            if (verbose) {
                std::cout << "Found " << files.size() << " Parquet files in directory" << std::endl;
            }
        } else {
            // Single file
            files.push_back(path);
        }
        
        // Create work units from all files
        std::vector<WorkUnit> all_work_units = create_work_units(files);
        
        if (all_work_units.empty()) {
            throw std::runtime_error("No row groups found in Parquet file(s)");
        }
        
        // Count total row groups
        int total_row_groups = all_work_units.size();
        
        if (num_processes < 0) num_processes = 0;
        if (num_processes == 0) num_processes = sysconf(_SC_NPROCESSORS_ONLN);
        if (num_processes > MAX_WORKERS) num_processes = MAX_WORKERS;
        if (num_processes < 1) num_processes = 1;
        
        if (verbose) {
            std::cout << "Fitting on " << files.size() << " file(s) with " 
                     << total_row_groups << " total row groups ("
                     << num_processes << " workers)" << std::endl;
        }
        
        // Distribute work units among workers
        int work_per_worker = (total_row_groups + num_processes - 1) / num_processes;
        std::vector<int> pipes(num_processes);
        std::vector<pid_t> child_pids;
        
        for (int i = 0; i < num_processes; i++) {
            int pipefd[2];
            if (pipe(pipefd) == -1) {
                throw std::runtime_error("pipe() failed");
            }
            
            pid_t pid = fork();
            if (pid == -1) {
                throw std::runtime_error("fork() failed");
            }
            
            if (pid == 0) {
                // Child process
                ::close(pipefd[0]);
                
                int start_idx = i * work_per_worker;
                int end_idx = std::min(start_idx + work_per_worker, total_row_groups);
                
                std::vector<WorkUnit> my_work_units;
                if (start_idx < total_row_groups) {
                     my_work_units.insert(my_work_units.end(), 
                                        all_work_units.begin() + start_idx,
                                        all_work_units.begin() + end_idx);
                }
                
                worker_process(my_work_units, pipefd[1]);
                
                // --- FIX: Use _exit(0) instead of exit(0) ---
                // This prevents the child from running C++ destructors/atexit handlers
                // which corrupts the Python interpreter state in the child.
                _exit(0); 
            } else {
                // Parent process
                ::close(pipefd[1]);
                pipes[i] = pipefd[0];
                child_pids.push_back(pid);
            }
        }
        
        // Collect results from workers
        std::unordered_map<std::string, size_t> global_vocab;
        int completed = 0;
        
        for (int i = 0; i < num_processes; i++) {
            SharedTermResult result;
            while (read(pipes[i], &result, sizeof(result)) == sizeof(result)) {
                std::string word(result.word);
                global_vocab[word] += result.doc_count;
            }
            ::close(pipes[i]);
            completed++;
            
            if (verbose && completed % std::max(1, num_processes / 10) == 0) {
                std::cout << "  " << (completed * 100) / num_processes << "% complete" << std::endl;
            }
        }
        
        // Wait for all children
        for (pid_t pid : child_pids) {
            int status;
            waitpid(pid, &status, 0);
        }
        
        // Count total documents across all files
        total_docs_ = 0;
        for (const auto& file : files) {
            auto mmap_result = arrow::io::MemoryMappedFile::Open(
                file, arrow::io::FileMode::READ
            );
            
            if (!mmap_result.ok()) continue;
            
            auto mmap_file = mmap_result.ValueOrDie();
            
            std::unique_ptr<parquet::arrow::FileReader> reader;
            auto reader_status = parquet::arrow::OpenFile(
                mmap_file,
                arrow::default_memory_pool(),
                &reader
            );
            
            if (!reader_status.ok()) continue;
            
            auto metadata = reader->parquet_reader()->metadata();
            total_docs_ += metadata->num_rows();
        }
        
        if (global_vocab.empty() && total_docs_ > 0) {
            std::cerr << "Warning: No vocabulary extracted. Check 'text' column exists." << std::endl;
        }
        
        // Filter vocabulary by min_df and max_df
        std::unordered_map<std::string, size_t> filtered_vocab;
        for (const auto& [word, doc_freq] : global_vocab) {
            if (doc_freq < min_df_) continue;
            if (max_df_ > 0 && doc_freq > max_df_) continue;
            filtered_vocab[word] = doc_freq;
        }
        
        // Limit by max_features
        if (max_features_ > 0 && filtered_vocab.size() > max_features_) {
            std::vector<std::pair<std::string, size_t>> sorted_vocab(
                filtered_vocab.begin(), filtered_vocab.end()
            );
            std::sort(sorted_vocab.begin(), sorted_vocab.end(),
                     [](const auto& a, const auto& b) { return a.second > b.second; });
            
            filtered_vocab.clear();
            for (size_t i = 0; i < max_features_; i++) {
                filtered_vocab[sorted_vocab[i].first] = sorted_vocab[i].second;
            }
        }
        
        // Build final vocabulary and IDF
        vocabulary_.clear();
        idf_.clear();
        size_t idx = 0;
        
        for (const auto& [word, doc_freq] : filtered_vocab) {
            vocabulary_[word] = idx;
            double idf = std::log((double)total_docs_ / (double)doc_freq) + 1.0;
            idf_.push_back(idf);
            idx++;
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration<double>(end_time - start_time).count();
        
        if (verbose) {
            std::cout << "Fit complete: " << vocabulary_.size() << " features ("
                     << (int)total_time << "s)" << std::endl;
        }
    }

private:
    static void worker_process(const std::vector<WorkUnit>& work_units, int write_fd) {
        try {
            std::unordered_map<std::string, size_t> local_vocab;
            
            // Process each work unit
            for (const auto& unit : work_units) {
                // Open file
                auto mmap_result = arrow::io::MemoryMappedFile::Open(
                    unit.filename, arrow::io::FileMode::READ
                );
                
                if (!mmap_result.ok()) {
                    continue;
                }
                
                auto mmap_file = mmap_result.ValueOrDie();
                
                std::unique_ptr<parquet::arrow::FileReader> reader;
                auto reader_status = parquet::arrow::OpenFile(
                    mmap_file,
                    arrow::default_memory_pool(),
                    &reader
                );
                
                if (!reader_status.ok()) {
                    continue;
                }
                
                // Process assigned row groups for this file
                for (int rg = unit.start_row_group; rg < unit.end_row_group; rg++) {
                    std::shared_ptr<arrow::Table> table;
                    auto status = reader->RowGroup(rg)->ReadTable(&table);
                    
                    if (!status.ok()) {
                        continue;
                    }
                    
                    // Find text column
                    auto text_column = table->GetColumnByName("text");
                    if (!text_column) {
                        continue;
                    }
                    
                    // Process each chunk in the column
                    for (int chunk_idx = 0; chunk_idx < text_column->num_chunks(); chunk_idx++) {
                        auto array = text_column->chunk(chunk_idx);
                        
                        if (array->type()->id() != arrow::Type::STRING &&
                            array->type()->id() != arrow::Type::LARGE_STRING) {
                            continue;
                        }
                        
                        auto string_array = std::static_pointer_cast<arrow::StringArray>(array);
                        
                        // Process each string
                        for (int64_t i = 0; i < string_array->length(); i++) {
                            if (string_array->IsNull(i)) continue;
                            
                            auto text_view = string_array->GetView(i);
                            
                            // Use shared SIMD tokenizer (returns unique terms)
                            auto unique_terms = process_document_get_terms(
                                text_view.data(),
                                text_view.data() + text_view.length()
                            );
                            
                            // Count document frequency for each unique word
                            for (const auto& word : unique_terms) {
                                // Always count if word exists, only check limit for NEW words
                                auto it = local_vocab.find(word);
                                if (it != local_vocab.end()) {
                                    // Word exists, always increment
                                    it->second++;
                                } else if (local_vocab.size() < MAX_VOCABULARY_SIZE) {
                                    // New word, add only if under limit
                                    local_vocab[word] = 1;
                                }
                            }
                        }
                    }
                }
            }
            
            // Send results to parent
            for (const auto& [word, doc_freq] : local_vocab) {
                SharedTermResult result;
                result.hash = hash_bytes(word.c_str(), word.length());
                strncpy(result.word, word.c_str(), 127);
                result.word[127] = '\0';
                result.doc_count = doc_freq;
                
                ssize_t written = write(write_fd, &result, sizeof(result));
                if (written != sizeof(result)) {
                    break;
                }
            }
            
        } catch (...) {
        }
        
        ::close(write_fd);
    }

public:
    void save(const std::string& filename) const {
        std::ofstream out(filename);
        if (!out) {
            throw std::runtime_error("Cannot open file for writing: " + filename);
        }
        
        out << "TFIDF_MODEL_V1\n"
            << "total_docs=" << total_docs_ << "\n"
            << "vocab_size=" << vocabulary_.size() << "\n"
            << "min_df=" << min_df_ << "\n"
            << "max_df=" << max_df_ << "\n"
            << "max_features=" << max_features_ << "\n"
            << "VOCABULARY\n";
        
        std::vector<std::pair<std::string, size_t>> sorted_vocab(
            vocabulary_.begin(), vocabulary_.end()
        );
        std::sort(sorted_vocab.begin(), sorted_vocab.end(),
                 [](const auto& a, const auto& b) { return a.second < b.second; });
        
        for (const auto& [word, idx] : sorted_vocab) {
            double idf = idf_[idx];
            size_t doc_freq = (size_t)std::round(total_docs_ / std::exp(idf - 1.0));
            out << word << "," << doc_freq << "," << idf << "\n";
        }
    }

    void load(const std::string& filename) {
        std::ifstream in(filename);
        if (!in) {
            throw std::runtime_error("Cannot open model file: " + filename);
        }
        
        std::string line;
        if (!std::getline(in, line) || line != "TFIDF_MODEL_V1") {
            throw std::runtime_error("Invalid model format");
        }
        
        while (std::getline(in, line)) {
            if (line == "VOCABULARY") break;
            
            if (line.substr(0, 11) == "total_docs=") {
                total_docs_ = std::stoull(line.substr(11));
            }
        }
        
        vocabulary_.clear();
        idf_.clear();
        size_t idx = 0;
        
        while (std::getline(in, line)) {
            size_t first_comma = line.find(',');
            size_t second_comma = line.find(',', first_comma + 1);
            
            if (first_comma == std::string::npos || second_comma == std::string::npos) {
                continue;
            }
            
            std::string word = line.substr(0, first_comma);
            double idf = std::stod(line.substr(second_comma + 1));
            
            vocabulary_[word] = idx;
            idf_.push_back(idf);
            idx++;
        }
        
        if (vocabulary_.empty()) {
            throw std::runtime_error("No vocabulary loaded");
        }
    }

    void open_stream(const std::string& filename) {
        auto mmap_result = arrow::io::MemoryMappedFile::Open(
            filename, arrow::io::FileMode::READ
        );
        
        if (!mmap_result.ok()) {
            throw std::runtime_error("Cannot open Parquet file: " + 
                                   mmap_result.status().ToString());
        }
        
        mmap_file_ = mmap_result.ValueOrDie();
        
        auto reader_status = parquet::arrow::OpenFile(
            mmap_file_,
            arrow::default_memory_pool(),
            &reader_
        );
        
        if (!reader_status.ok()) {
            throw std::runtime_error("Cannot create reader: " + 
                                   reader_status.ToString());
        }
        
        current_row_group_ = 0;
        current_row_in_group_ = 0;
        current_table_ = nullptr;
        stream_open_ = true;
    }

    py::object get_batch(size_t batch_size_rows) {
        if (!stream_open_ || !reader_) {
            return py::none();
        }
        
        if (current_row_group_ >= reader_->num_row_groups()) {
            return py::none();
        }
        
        std::vector<int32_t> indices;
        std::vector<int32_t> indptr;
        std::vector<uint16_t> data;
        
        indices.reserve(100000);
        data.reserve(100000);
        indptr.push_back(0);
        
        size_t rows_collected = 0;
        
        while (rows_collected < batch_size_rows && 
               current_row_group_ < reader_->num_row_groups()) {
            
            // Load row group if needed
            if (!current_table_ || current_row_in_group_ >= current_table_->num_rows()) {
                auto status = reader_->RowGroup(current_row_group_)->ReadTable(&current_table_);
                
                if (!status.ok()) {
                    current_row_group_++;
                    current_row_in_group_ = 0;
                    current_table_ = nullptr;
                    continue;
                }
                
                current_row_in_group_ = 0;
            }
            
            auto text_column = current_table_->GetColumnByName("text");
            if (!text_column) {
                current_row_group_++;
                current_row_in_group_ = 0;
                current_table_ = nullptr;
                continue;
            }
            
            // Process rows in this row group
            while (current_row_in_group_ < current_table_->num_rows() && 
                   rows_collected < batch_size_rows) {
                
                // Find the chunk containing this row
                int64_t cumulative_rows = 0;
                for (int chunk_idx = 0; chunk_idx < text_column->num_chunks(); chunk_idx++) {
                    auto array = text_column->chunk(chunk_idx);
                    
                    if (current_row_in_group_ < cumulative_rows + array->length()) {
                        int64_t row_in_chunk = current_row_in_group_ - cumulative_rows;
                        
                        auto string_array = std::static_pointer_cast<arrow::StringArray>(array);
                        
                        if (!string_array->IsNull(row_in_chunk)) {
                            auto text_view = string_array->GetView(row_in_chunk);
                            
                            auto terms = process_document_get_terms(
                                text_view.data(),
                                text_view.data() + text_view.length()
                            );
                            
                            std::vector<size_t> term_indices;
                            for (const auto& word : terms) {
                                auto it = vocabulary_.find(word);
                                if (it != vocabulary_.end()) {
                                    term_indices.push_back(it->second);
                                }
                            }
                            
                            std::sort(term_indices.begin(), term_indices.end());
                            
                            if (!term_indices.empty()) {
                                size_t current_idx = term_indices[0];
                                int count = 1;
                                
                                for (size_t i = 1; i <= term_indices.size(); ++i) {
                                    if (i < term_indices.size() && term_indices[i] == current_idx) {
                                        count++;
                                    } else {
                                        indices.push_back((int32_t)current_idx);
                                        data.push_back(count > 65535 ? 65535 : (uint16_t)count);
                                        
                                        if (i < term_indices.size()) {
                                            current_idx = term_indices[i];
                                            count = 1;
                                        }
                                    }
                                }
                            }
                        }
                        
                        indptr.push_back(static_cast<int32_t>(indices.size()));
                        break;
                    }
                    
                    cumulative_rows += array->length();
                }
                
                current_row_in_group_++;
                rows_collected++;
            }
            
            if (current_row_in_group_ >= current_table_->num_rows()) {
                current_row_group_++;
                current_row_in_group_ = 0;
                current_table_ = nullptr;
            }
        }
        
        if (rows_collected == 0) {
            return py::none();
        }
        
        return py::make_tuple(
            py::array_t<uint16_t>(data.size(), data.data()),
            py::array_t<int32_t>(indices.size(), indices.data()),
            py::array_t<int32_t>(indptr.size(), indptr.data())
        );
    }

    // All utility methods (same as CSV version)
    std::unordered_map<std::string, size_t> get_vocabulary() const { return vocabulary_; }
    py::array_t<double> get_idf_array() { return py::array_t<double>(idf_.size(), idf_.data()); }
    std::vector<double> get_idf() const { return idf_; }
    std::vector<std::string> get_feature_names() const {
        std::vector<std::string> names(vocabulary_.size());
        for (const auto& [word, idx] : vocabulary_) names[idx] = word;
        return names;
    }
    
    py::list get_top_idf_words(size_t n = 10) {
        std::vector<std::pair<std::string, double>> word_idf;
        for (const auto& [word, idx] : vocabulary_) {
            word_idf.emplace_back(word, idf_[idx]);
        }
        std::partial_sort(word_idf.begin(),
                         word_idf.begin() + std::min(n, word_idf.size()),
                         word_idf.end(),
                         [](const auto& a, const auto& b) { return a.second > b.second; });
        py::list result;
        for (size_t i = 0; i < std::min(n, word_idf.size()); i++) {
            result.append(py::make_tuple(word_idf[i].first, word_idf[i].second));
        }
        return result;
    }
    
    py::list get_bottom_idf_words(size_t n = 10) {
        std::vector<std::pair<std::string, double>> word_idf;
        for (const auto& [word, idx] : vocabulary_) {
            word_idf.emplace_back(word, idf_[idx]);
        }
        std::partial_sort(word_idf.begin(),
                         word_idf.begin() + std::min(n, word_idf.size()),
                         word_idf.end(),
                         [](const auto& a, const auto& b) { return a.second < b.second; });
        py::list result;
        for (size_t i = 0; i < std::min(n, word_idf.size()); i++) {
            result.append(py::make_tuple(word_idf[i].first, word_idf[i].second));
        }
        return result;
    }
    
    py::object get_word_idf(const std::string& word) {
        auto it = vocabulary_.find(word);
        if (it == vocabulary_.end()) return py::none();
        return py::float_(idf_[it->second]);
    }
    
    py::object get_word_df(const std::string& word) {
        auto it = vocabulary_.find(word);
        if (it == vocabulary_.end()) return py::none();
        size_t df = (size_t)std::round(total_docs_ / std::exp(idf_[it->second] - 1.0));
        return py::int_(df);
    }
    
    py::list search_words(const std::string& pattern, size_t max_results = 100) {
        py::list result;
        size_t count = 0;
        for (const auto& [word, idx] : vocabulary_) {
            if (word.find(pattern) != std::string::npos) {
                result.append(py::make_tuple(word, idf_[idx]));
                if (++count >= max_results) break;
            }
        }
        return result;
    }
    
    py::dict get_vocab_stats() {
        double min_idf = *std::min_element(idf_.begin(), idf_.end());
        double max_idf = *std::max_element(idf_.begin(), idf_.end());
        double sum_idf = std::accumulate(idf_.begin(), idf_.end(), 0.0);
        double mean_idf = sum_idf / idf_.size();
        
        std::vector<double> sorted_idf = idf_;
        std::sort(sorted_idf.begin(), sorted_idf.end());
        double median_idf = sorted_idf[sorted_idf.size() / 2];
        
        py::dict stats;
        stats["vocab_size"] = vocabulary_.size();
        stats["total_docs"] = total_docs_;
        stats["min_idf"] = min_idf;
        stats["max_idf"] = max_idf;
        stats["mean_idf"] = mean_idf;
        stats["median_idf"] = median_idf;
        return stats;
    }
    
    bool has_word(const std::string& word) { return vocabulary_.find(word) != vocabulary_.end(); }
    size_t get_vocab_size() const { return vocabulary_.size(); }
    size_t get_total_docs() const { return total_docs_; }
    
    py::dict export_vocabulary_with_idf() {
        py::dict result;
        for (const auto& [word, idx] : vocabulary_) {
            result[py::str(word)] = idf_[idx];
        }
        return result;
    }
};

PYBIND11_MODULE(fasttfidf_parquet, m) {
    m.doc() = "High-performance TF-IDF for Parquet files with SIMD";
    
    py::class_<ParquetTfidfVectorizer>(m, "TfidfVectorizer")
        .def(py::init<>())
        .def("fit", &ParquetTfidfVectorizer::fit,
             py::arg("path"), py::arg("num_processes") = 0, py::arg("min_df") = 1,
             py::arg("max_df") = 0, py::arg("max_features") = 0, py::arg("verbose") = true,
             "Fit vectorizer on Parquet file or directory of Parquet files")
        .def("save", &ParquetTfidfVectorizer::save)
        .def("load", &ParquetTfidfVectorizer::load)
        .def("open_stream", &ParquetTfidfVectorizer::open_stream)
        .def("get_batch", &ParquetTfidfVectorizer::get_batch)
        .def("get_vocabulary", &ParquetTfidfVectorizer::get_vocabulary)
        .def("get_idf", &ParquetTfidfVectorizer::get_idf)
        .def("get_idf_array", &ParquetTfidfVectorizer::get_idf_array)
        .def("get_feature_names", &ParquetTfidfVectorizer::get_feature_names)
        .def("get_top_idf_words", &ParquetTfidfVectorizer::get_top_idf_words, py::arg("n") = 10)
        .def("get_bottom_idf_words", &ParquetTfidfVectorizer::get_bottom_idf_words, py::arg("n") = 10)
        .def("get_word_idf", &ParquetTfidfVectorizer::get_word_idf)
        .def("get_word_df", &ParquetTfidfVectorizer::get_word_df)
        .def("search_words", &ParquetTfidfVectorizer::search_words,
             py::arg("pattern"), py::arg("max_results") = 100)
        .def("get_vocab_stats", &ParquetTfidfVectorizer::get_vocab_stats)
        .def("has_word", &ParquetTfidfVectorizer::has_word)
        .def("get_vocab_size", &ParquetTfidfVectorizer::get_vocab_size)
        .def("get_total_docs", &ParquetTfidfVectorizer::get_total_docs)
        .def("export_vocabulary_with_idf", &ParquetTfidfVectorizer::export_vocabulary_with_idf);

#ifdef USE_SIMD
    m.attr("simd_enabled") = true;
    m.attr("simd_type") = SIMD_TYPE;
#else
    m.attr("simd_enabled") = false;
#endif
}