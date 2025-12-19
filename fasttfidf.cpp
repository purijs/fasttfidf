#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <cctype>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <random>
#include <limits>

#ifdef __APPLE__
#include <sys/sysctl.h>
#include <mach/mach.h>
#include <mach/mach_host.h>
#endif

namespace py = pybind11;

/// Configuration constants for safety and resource limits
constexpr size_t MAX_LINE_LENGTH = 100 * 1024 * 1024;
constexpr size_t MIN_FILE_SIZE = 5;
constexpr size_t MAX_WORD_LENGTH = 127;
constexpr size_t MAX_VOCABULARY_SIZE = 10000000;
constexpr int MAX_WORKERS = 128;

#ifdef __AVX2__
#include <immintrin.h>
#define USE_SIMD 1
#define SIMD_WIDTH 32
#define SIMD_TYPE "AVX2"
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#define USE_SIMD 1
#define SIMD_WIDTH 16
#define SIMD_TYPE "NEON"
#endif

/**
 * Estimates available system memory in MB.
 * Supports macOS (via mach_host) and Linux (via /proc/meminfo).
 */
size_t get_available_memory_mb() {
    size_t available_mb = 0;
#ifdef __APPLE__
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    vm_statistics64_data_t vmstat;
    kern_return_t ret = host_statistics64(mach_host_self(), HOST_VM_INFO64,
                                          (host_info64_t)&vmstat, &count);
    if (ret == KERN_SUCCESS) {
        uint64_t free_pages = vmstat.free_count + vmstat.inactive_count;
        uint64_t page_size = 0;
        size_t len = sizeof(page_size);
        sysctlbyname("hw.pagesize", &page_size, &len, NULL, 0);
        available_mb = (free_pages * page_size) / (1024 * 1024);
    } else {
        int mib[2] = {CTL_HW, HW_MEMSIZE};
        size_t length = sizeof(size_t);
        size_t total_bytes;
        if (sysctl(mib, 2, &total_bytes, &length, nullptr, 0) == 0) {
            available_mb = total_bytes / (4 * 1024 * 1024);
        }
    }
#else
    std::ifstream meminfo("/proc/meminfo");
    std::string line;
    while (std::getline(meminfo, line)) {
        if (line.substr(0, 13) == "MemAvailable:") {
            try {
                size_t available_kb = std::stoull(line.substr(13));
                available_mb = available_kb / 1024;
            } catch (...) {
            }
            break;
        }
    }
#endif
    if (available_mb == 0 || available_mb > 1000000) {
        available_mb = 2048;
    }
    return available_mb;
}

/**
 * Calculates optimal batch size for workers based on available RAM.
 * reserves 10% safety margin and splits remainder among workers.
 */
size_t calculate_optimal_sub_batch_size(size_t available_mb, int num_workers) {
    if (num_workers <= 0) num_workers = 1;
    if (num_workers > MAX_WORKERS) num_workers = MAX_WORKERS;
    if (available_mb == 0 || available_mb > 1000000) {
        available_mb = 2048;
    }
    
    size_t safe_mb = (size_t)(available_mb * 0.90);
    size_t per_worker_mb = safe_mb / num_workers;
    size_t sub_batch_mb = (size_t)(per_worker_mb / 0.80);
    size_t sub_batch_bytes = sub_batch_mb * 1024 * 1024;
    
    const size_t MIN_SUB_BATCH = 100ULL * 1024 * 1024;
    const size_t MAX_SUB_BATCH = 2048ULL * 1024 * 1024;
    
    if (sub_batch_bytes < MIN_SUB_BATCH) sub_batch_bytes = MIN_SUB_BATCH;
    if (sub_batch_bytes > MAX_SUB_BATCH) sub_batch_bytes = MAX_SUB_BATCH;
    
    return sub_batch_bytes;
}

/// FNV-1a hash implementation
uint64_t hash_bytes(const char* data, size_t len) {
    uint64_t hash = 14695981039346656037ULL;
    for (size_t i = 0; i < len; i++) {
        hash ^= static_cast<uint64_t>(data[i]);
        hash *= 1099511628211ULL;
    }
    return hash;
}

/// IPC structure for passing term counts from child processes
struct SharedTermResult {
    uint64_t hash;
    char word[128];
    size_t doc_count;
};

#ifdef __AVX2__
inline __m256i is_alnum_avx2(const __m256i chars) {
    __m256i zero = _mm256_set1_epi8('0'), nine = _mm256_set1_epi8('9');
    __m256i ge_zero = _mm256_cmpgt_epi8(_mm256_add_epi8(chars, _mm256_set1_epi8(1)), zero);
    __m256i le_nine = _mm256_cmpgt_epi8(_mm256_add_epi8(nine, _mm256_set1_epi8(1)), chars);
    __m256i is_digit = _mm256_and_si256(ge_zero, le_nine);
    __m256i a_upper = _mm256_set1_epi8('A'), z_upper = _mm256_set1_epi8('Z');
    __m256i ge_a_upper = _mm256_cmpgt_epi8(_mm256_add_epi8(chars, _mm256_set1_epi8(1)), a_upper);
    __m256i le_z_upper = _mm256_cmpgt_epi8(_mm256_add_epi8(z_upper, _mm256_set1_epi8(1)), chars);
    __m256i is_upper = _mm256_and_si256(ge_a_upper, le_z_upper);
    __m256i a_lower = _mm256_set1_epi8('a'), z_lower = _mm256_set1_epi8('z');
    __m256i ge_a_lower = _mm256_cmpgt_epi8(_mm256_add_epi8(chars, _mm256_set1_epi8(1)), a_lower);
    __m256i le_z_lower = _mm256_cmpgt_epi8(_mm256_add_epi8(z_lower, _mm256_set1_epi8(1)), chars);
    __m256i is_lower = _mm256_and_si256(ge_a_lower, le_z_lower);
    return _mm256_or_si256(_mm256_or_si256(is_digit, is_upper), is_lower);
}

inline __m256i to_lower_avx2(const __m256i chars) {
    __m256i a_upper = _mm256_set1_epi8('A'), z_upper = _mm256_set1_epi8('Z');
    __m256i ge_a = _mm256_cmpgt_epi8(_mm256_add_epi8(chars, _mm256_set1_epi8(1)), a_upper);
    __m256i le_z = _mm256_cmpgt_epi8(_mm256_add_epi8(z_upper, _mm256_set1_epi8(1)), chars);
    __m256i is_upper = _mm256_and_si256(ge_a, le_z);
    __m256i offset = _mm256_and_si256(is_upper, _mm256_set1_epi8(32));
    return _mm256_add_epi8(chars, offset);
}

/**
 * AVX2 optimized tokenizer.
 * Processes 32 bytes at a time for alphanumeric checks and lowercasing.
 */
std::unordered_set<std::string> process_document_get_terms_avx2(const char* row_start, const char* row_end) {
    std::unordered_set<std::string> unique_terms;
    if (row_end <= row_start) return unique_terms;
    
    const char* B = row_start;
    std::string word_buffer;
    word_buffer.reserve(50);
    
    const size_t len = row_end - row_start;
    const size_t avx_chunks = len / 32;
    const char* avx_end = row_start + (avx_chunks * 32);
    
    while (B < avx_end) {
        __m256i chunk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(B));
        __m256i alnum_mask = is_alnum_avx2(chunk);
        __m256i lower_chunk = to_lower_avx2(chunk);
        
        alignas(32) uint8_t buffer[32];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(buffer), lower_chunk);
        alignas(32) uint8_t mask_buffer[32];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(mask_buffer), alnum_mask);
        
        for (int i = 0; i < 32; i++) {
            if (mask_buffer[i]) {
                if (word_buffer.length() < MAX_WORD_LENGTH) {
                    word_buffer += static_cast<char>(buffer[i]);
                }
            } else {
                if (word_buffer.length() >= 2 && word_buffer.length() <= MAX_WORD_LENGTH) {
                    unique_terms.insert(word_buffer);
                }
                word_buffer.clear();
            }
        }
        B += 32;
    }
    
    // Scalar fallback for remaining bytes
    while (B < row_end) {
        if (std::isalnum(static_cast<unsigned char>(*B))) {
            if (word_buffer.length() < MAX_WORD_LENGTH) {
                word_buffer += std::tolower(static_cast<unsigned char>(*B));
            }
        } else {
            if (word_buffer.length() >= 2 && word_buffer.length() <= MAX_WORD_LENGTH) {
                unique_terms.insert(word_buffer);
            }
            word_buffer.clear();
        }
        B++;
    }
    
    if (word_buffer.length() >= 2 && word_buffer.length() <= MAX_WORD_LENGTH) {
        unique_terms.insert(word_buffer);
    }
    
    return unique_terms;
}
#endif

#ifdef __ARM_NEON
inline uint8x16_t is_alnum_neon(const uint8x16_t chars) {
    uint8x16_t zero = vdupq_n_u8('0'), nine = vdupq_n_u8('9');
    uint8x16_t is_digit = vandq_u8(vcgeq_u8(chars, zero), vcleq_u8(chars, nine));
    uint8x16_t a_upper = vdupq_n_u8('A'), z_upper = vdupq_n_u8('Z');
    uint8x16_t is_upper = vandq_u8(vcgeq_u8(chars, a_upper), vcleq_u8(chars, z_upper));
    uint8x16_t a_lower = vdupq_n_u8('a'), z_lower = vdupq_n_u8('z');
    uint8x16_t is_lower = vandq_u8(vcgeq_u8(chars, a_lower), vcleq_u8(chars, z_lower));
    return vorrq_u8(vorrq_u8(is_digit, is_upper), is_lower);
}

inline uint8x16_t to_lower_neon(const uint8x16_t chars) {
    uint8x16_t a_upper = vdupq_n_u8('A'), z_upper = vdupq_n_u8('Z');
    uint8x16_t is_upper = vandq_u8(vcgeq_u8(chars, a_upper), vcleq_u8(chars, z_upper));
    uint8x16_t offset = vandq_u8(is_upper, vdupq_n_u8(32));
    return vaddq_u8(chars, offset);
}

/**
 * ARM NEON optimized tokenizer.
 * Processes 16 bytes at a time.
 */
std::unordered_set<std::string> process_document_get_terms_neon(const char* row_start, const char* row_end) {
    std::unordered_set<std::string> unique_terms;
    if (row_end <= row_start) return unique_terms;
    
    const char* B = row_start;
    std::string word_buffer;
    word_buffer.reserve(50);
    
    const size_t len = row_end - row_start;
    const size_t neon_chunks = len / 16;
    const char* neon_end = row_start + (neon_chunks * 16);
    
    while (B < neon_end) {
        uint8x16_t chunk = vld1q_u8(reinterpret_cast<const uint8_t*>(B));
        uint8x16_t alnum_mask = is_alnum_neon(chunk);
        uint8x16_t lower_chunk = to_lower_neon(chunk);
        
        alignas(16) uint8_t buffer[16];
        vst1q_u8(buffer, lower_chunk);
        alignas(16) uint8_t mask_buffer[16];
        vst1q_u8(mask_buffer, alnum_mask);
        
        for (int i = 0; i < 16; i++) {
            if (mask_buffer[i]) {
                if (word_buffer.length() < MAX_WORD_LENGTH) {
                    word_buffer += static_cast<char>(buffer[i]);
                }
            } else {
                if (word_buffer.length() >= 2 && word_buffer.length() <= MAX_WORD_LENGTH) {
                    unique_terms.insert(word_buffer);
                }
                word_buffer.clear();
            }
        }
        B += 16;
    }
    
    while (B < row_end) {
        if (std::isalnum(static_cast<unsigned char>(*B))) {
            if (word_buffer.length() < MAX_WORD_LENGTH) {
                word_buffer += std::tolower(static_cast<unsigned char>(*B));
            }
        } else {
            if (word_buffer.length() >= 2 && word_buffer.length() <= MAX_WORD_LENGTH) {
                unique_terms.insert(word_buffer);
            }
            word_buffer.clear();
        }
        B++;
    }
    
    if (word_buffer.length() >= 2 && word_buffer.length() <= MAX_WORD_LENGTH) {
        unique_terms.insert(word_buffer);
    }
    
    return unique_terms;
}
#endif

std::unordered_set<std::string> process_document_get_terms_scalar(const char* row_start, const char* row_end) {
    std::unordered_set<std::string> unique_terms;
    if (row_end <= row_start) return unique_terms;
    
    const char* B = row_start;
    std::string word_buffer;
    word_buffer.reserve(50);
    
    while (B < row_end) {
        word_buffer.clear();
        
        while (B < row_end && !std::isalnum(static_cast<unsigned char>(*B))) B++;
        if (B >= row_end) break;
        
        while (B < row_end && std::isalnum(static_cast<unsigned char>(*B)) &&
               word_buffer.length() < MAX_WORD_LENGTH) {
            word_buffer += std::tolower(static_cast<unsigned char>(*B));
            B++;
        }
        
        while (B < row_end && std::isalnum(static_cast<unsigned char>(*B))) B++;
        
        if (word_buffer.length() >= 2 && word_buffer.length() <= MAX_WORD_LENGTH) {
            unique_terms.insert(word_buffer);
        }
    }
    
    return unique_terms;
}

inline std::unordered_set<std::string> process_document_get_terms(const char* row_start, const char* row_end) {
#ifdef __AVX2__
    return process_document_get_terms_avx2(row_start, row_end);
#elif defined(__ARM_NEON)
    return process_document_get_terms_neon(row_start, row_end);
#else
    return process_document_get_terms_scalar(row_start, row_end);
#endif
}

/**
 * Wrapper for memory-mapped file operations.
 * Handles open, mmap, and sequential read advice.
 */
struct MappedFile {
    const char* data;
    size_t size;
    int fd;
    
    MappedFile() : data(nullptr), size(0), fd(-1) {}
    
    bool open(const char* filename) {
        if (!filename || filename[0] == '\0') {
            return false;
        }
        
        fd = ::open(filename, O_RDONLY);
        if (fd == -1) return false;
        
        struct stat sb;
        if (fstat(fd, &sb) == -1) {
            ::close(fd);
            fd = -1;
            return false;
        }
        
        if (!S_ISREG(sb.st_mode)) {
            ::close(fd);
            fd = -1;
            return false;
        }
        
        size = sb.st_size;
        
        if (size < MIN_FILE_SIZE) {
            ::close(fd);
            fd = -1;
            return false;
        }
        
        data = static_cast<const char*>(mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0));
        if (data == MAP_FAILED) {
            ::close(fd);
            fd = -1;
            data = nullptr;
            return false;
        }
        
        madvise((void*)data, size, MADV_SEQUENTIAL);
        return true;
    }
    
    void close() {
        if (data && data != MAP_FAILED) {
            munmap((void*)data, size);
        }
        if (fd != -1) {
            ::close(fd);
        }
        data = nullptr;
        fd = -1;
        size = 0;
    }
    
    ~MappedFile() {
        close();
    }
    
    size_t count_documents(bool skip_header = true) const {
        if (!data || size == 0) return 0;
        
        size_t count = 0;
        for (size_t i = 0; i < size && count < SIZE_MAX - 1; i++) {
            if (data[i] == '\n') count++;
        }
        
        if (size > 0 && data[size - 1] != '\n') {
            count++;
        }
        
        return skip_header ? (count > 0 ? count - 1 : 0) : count;
    }
};

/**
 * Worker function for child processes.
 * Reads a chunk of the mapped file, counts terms, and writes results to pipe.
 */
void worker_process(const MappedFile& mfile, size_t start_byte, size_t end_byte, 
                    int write_fd, bool skip_header, size_t sub_batch_size) {
    
    if (write_fd < 0 || !mfile.data || 
        end_byte > mfile.size || start_byte >= end_byte) {
        if (write_fd >= 0) ::close(write_fd);
        return;
    }
    
    size_t current_pos = start_byte;
    
    // Adjust start position to beginning of next line if starting mid-file
    if (start_byte > 0) {
        while (current_pos > 0 && *(mfile.data + current_pos - 1) != '\n')
            current_pos--;
    }
    
    if (skip_header && start_byte == 0) {
        while (current_pos < end_byte && *(mfile.data + current_pos) != '\n') 
            current_pos++;
        if (current_pos < end_byte) current_pos++;
    }
    
    while (current_pos < end_byte) {
        std::unordered_map<std::string, size_t> local_vocab;
        
        size_t sub_batch_end = std::min(current_pos + sub_batch_size, end_byte);
        
        if (sub_batch_end < end_byte) {
            while (sub_batch_end < end_byte && *(mfile.data + sub_batch_end) != '\n') 
                sub_batch_end++;
        }
        
        const char* line_start = mfile.data + current_pos;
        const char* chunk_end = mfile.data + sub_batch_end;
        
        while (line_start < chunk_end) {
            const char* line_end = line_start;
            size_t line_length = 0;
            
            while (line_end < chunk_end && *line_end != '\n' && 
                   line_length < MAX_LINE_LENGTH) {
                line_end++;
                line_length++;
            }
            
            if (line_length >= MAX_LINE_LENGTH) {
                while (line_end < chunk_end && *line_end != '\n') {
                    line_end++;
                }
            }
            
            if (line_end > line_start && line_length < MAX_LINE_LENGTH) {
                try {
                    auto terms = process_document_get_terms(line_start, line_end);
                    for (const auto& word : terms) {
                        if (local_vocab.size() < MAX_VOCABULARY_SIZE) {
                            local_vocab[word]++;
                        }
                    }
                } catch (...) {
                }
            }
            
            line_start = line_end + 1;
        }
        
        for (const auto& [word, doc_freq] : local_vocab) {
            SharedTermResult result;
            result.hash = hash_bytes(word.c_str(), word.length());
            strncpy(result.word, word.c_str(), MAX_WORD_LENGTH);
            result.word[MAX_WORD_LENGTH] = '\0';
            result.doc_count = doc_freq;
            
            ssize_t written = write(write_fd, &result, sizeof(result));
            if (written != sizeof(result)) {
                break;
            }
        }
        
        local_vocab.clear();
        current_pos = sub_batch_end;
    }
    
    ::close(write_fd);
}

/**
 * Main TF-IDF Vectorizer class exposed to Python.
 * Handles fitting via multiprocessing and transforming via streaming.
 */
class TfidfVectorizer {
public:
    std::unordered_map<std::string, size_t> vocabulary_;
    std::vector<double> idf_;
    size_t total_docs_ = 0;
    size_t min_df_ = 1;
    size_t max_df_ = 0;
    size_t max_features_ = 0;

    MappedFile stream_file_;
    size_t stream_offset_ = 0;
    bool stream_open_ = false;

    /**
     * fits the model using the provided CSV file.
     * Uses fork/exec for multiprocessing to build vocabulary.
     */
    void fit(const std::string& filename, int num_processes = 0, size_t min_df = 1,
         size_t max_df = 0, size_t max_features = 0, bool verbose = true) {
        min_df_ = min_df;
        max_df_ = max_df;
        max_features_ = max_features;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        MappedFile mfile;
        if (!mfile.open(filename.c_str())) {
            throw std::runtime_error(
                "Cannot open file: " + filename + 
                " (check file exists, is readable, and is a regular file)"
            );
        }
        
        if (num_processes < 0) num_processes = 0;
        if (num_processes == 0) num_processes = sysconf(_SC_NPROCESSORS_ONLN);
        if (num_processes > MAX_WORKERS) num_processes = MAX_WORKERS;
        if (num_processes < 1) num_processes = 1;
        
        size_t available_mb = get_available_memory_mb();
        
        if (available_mb < 100) {
            throw std::runtime_error(
                "Insufficient memory: only " + std::to_string(available_mb) + 
                " MB available"
            );
        }
        
        size_t sub_batch_size = calculate_optimal_sub_batch_size(available_mb, num_processes);
        
        if (verbose) {
            std::cout << "Fitting on " << filename << " ("
                    << (mfile.size / (1024.0 * 1024.0 * 1024.0)) << " GB, "
                    << num_processes << " workers)" << std::endl;
            std::cout << "Memory: " << available_mb << " MB available, "
                    << (sub_batch_size / (1024 * 1024)) << " MB sub-batches" << std::endl;
        }
        
        size_t bytes_per_process = (mfile.size + num_processes - 1) / num_processes;
        std::vector<int> pipes(num_processes);
        std::vector<pid_t> child_pids;
        
        for (int i = 0; i < num_processes; i++) {
            int pipefd[2];
            if (pipe(pipefd) == -1) throw std::runtime_error("pipe() failed");
            
            pid_t pid = fork();
            if (pid == -1) throw std::runtime_error("fork() failed");
            
            if (pid == 0) {
                ::close(pipefd[0]);
                size_t start_byte = i * bytes_per_process;
                size_t end_byte = std::min(start_byte + bytes_per_process, mfile.size);
                worker_process(mfile, start_byte, end_byte, pipefd[1], i == 0, sub_batch_size);
                exit(0);
            } else {
                ::close(pipefd[1]);
                pipes[i] = pipefd[0];
                child_pids.push_back(pid);
            }
        }
        
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
        
        for (pid_t pid : child_pids) { 
            int status; 
            waitpid(pid, &status, 0); 
        }
        
        total_docs_ = mfile.count_documents(true);
        
        if (global_vocab.empty() && total_docs_ > 0) {
            std::cerr << "Warning: No vocabulary extracted. Check CSV format." << std::endl;
        }
        
        std::unordered_map<std::string, size_t> filtered_vocab;
        for (const auto& [word, doc_freq] : global_vocab) {
            if (doc_freq < min_df_) continue;
            if (max_df_ > 0 && doc_freq > max_df_) continue;
            filtered_vocab[word] = doc_freq;
        }
        
        if (max_features_ > 0 && filtered_vocab.size() > max_features_) {
            std::vector<std::pair<std::string, size_t>> sorted_vocab(filtered_vocab.begin(), filtered_vocab.end());
            std::sort(sorted_vocab.begin(), sorted_vocab.end(), 
                    [](const auto& a, const auto& b) { return a.second > b.second; });
            filtered_vocab.clear();
            for (size_t i = 0; i < max_features_; i++) { 
                filtered_vocab[sorted_vocab[i].first] = sorted_vocab[i].second; 
            }
        }
        
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
    
    std::unordered_map<std::string, size_t> get_vocabulary() const { return vocabulary_; }
    
    py::array_t<double> get_idf_array() {
        return py::array_t<double>(idf_.size(), idf_.data());
    }
    
    std::vector<double> get_idf() const { return idf_; }
    
    std::vector<std::string> get_feature_names() const {
        std::vector<std::string> names(vocabulary_.size());
        for (const auto& [word, idx] : vocabulary_) names[idx] = word;
        return names;
    }

    void save(const std::string& filename) const {
        std::ofstream out(filename);
        if (!out) throw std::runtime_error("Cannot open file for writing: " + filename);
        out << "TFIDF_MODEL_V1\ntotal_docs=" << total_docs_ << "\nvocab_size=" << vocabulary_.size() << "\nmin_df=" << min_df_ << "\nmax_df=" << max_df_ << "\nmax_features=" << max_features_ << "\nVOCABULARY\n";
        std::vector<std::pair<std::string, size_t>> sorted_vocab(vocabulary_.begin(), vocabulary_.end());
        std::sort(sorted_vocab.begin(), sorted_vocab.end(), [](const auto& a, const auto& b) { return a.second < b.second; });
        for (const auto& [word, idx] : sorted_vocab) {
            double idf = idf_[idx];
            size_t doc_freq = (size_t)std::round(total_docs_ / std::exp(idf - 1.0));
            out << word << "," << doc_freq << "," << idf << "\n";
        }
        out.close();
    }

    void load(const std::string& filename) {
        std::ifstream in(filename);
        if (!in) {
            throw std::runtime_error("Cannot open model file: " + filename);
        }
        
        std::string line;
        if (!std::getline(in, line)) {
            throw std::runtime_error("Empty model file");
        }
        
        if (line != "TFIDF_MODEL_V1") {
            throw std::runtime_error(
                "Invalid model format (expected TFIDF_MODEL_V1, got: " + line + ")"
            );
        }
        
        while (std::getline(in, line)) {
            if (line == "VOCABULARY") break;
            
            if (line.substr(0, 11) == "total_docs=") {
                try {
                    total_docs_ = std::stoull(line.substr(11));
                    if (total_docs_ > 10000000000ULL) {
                        throw std::runtime_error("Invalid total_docs value");
                    }
                } catch (...) {
                    throw std::runtime_error("Invalid total_docs format");
                }
            }
        }
        
        vocabulary_.clear();
        idf_.clear();
        size_t idx = 0;
        
        while (std::getline(in, line)) {
            if (vocabulary_.size() >= MAX_VOCABULARY_SIZE) {
                std::cerr << "Warning: Vocabulary truncated at " 
                          << MAX_VOCABULARY_SIZE << " words" << std::endl;
                break;
            }
            
            size_t first_comma = line.find(',');
            size_t second_comma = line.find(',', first_comma + 1);
            
            if (first_comma == std::string::npos || second_comma == std::string::npos) {
                continue;
            }
            
            std::string word = line.substr(0, first_comma);
            
            if (word.empty() || word.length() > MAX_WORD_LENGTH) {
                continue;
            }
            
            try {
                double idf = std::stod(line.substr(second_comma + 1));
                
                if (std::isnan(idf) || std::isinf(idf) || idf < 0) {
                    continue;
                }
                
                vocabulary_[word] = idx;
                idf_.push_back(idf);
                idx++;
            } catch (...) {
                continue;
            }
        }
        
        if (vocabulary_.empty()) {
            throw std::runtime_error("No vocabulary loaded from model file");
        }
    }

    void open_stream(const std::string& filename) {
        if (stream_open_) stream_file_.close();
        if (!stream_file_.open(filename.c_str())) {
            throw std::runtime_error("Cannot open CSV for streaming");
        }
        stream_offset_ = 0;
        stream_open_ = true;
    }

    /**
     * Retrieves a batch of documents from the stream.
     * Returns a tuple of (data, indices, indptr) for CSR matrix construction.
     */
    py::object get_batch(size_t batch_size_bytes) {
        if (!stream_open_ || !stream_file_.data) {
            return py::none();
        }
        
        if (stream_offset_ >= stream_file_.size) {
            return py::none();
        }
        
        if (batch_size_bytes > 4ULL * 1024 * 1024 * 1024) {
            batch_size_bytes = 4ULL * 1024 * 1024 * 1024;
        }

        std::vector<int32_t> indices;
        std::vector<int32_t> indptr;
        std::vector<uint16_t> data;

        try {
            indices.reserve(100000);
            data.reserve(100000);
        } catch (const std::bad_alloc&) {
            return py::none();
        }
        
        indptr.push_back(0);

        const char* ptr = stream_file_.data + stream_offset_;
        const char* end_file = stream_file_.data + stream_file_.size;
        const char* end_batch = std::min(ptr + batch_size_bytes, end_file);

        if (end_batch < end_file) {
            while (end_batch > ptr && *end_batch != '\n') end_batch--;
        }

        if (stream_offset_ == 0) {
            while (ptr < end_batch && *ptr != '\n') ptr++;
            if (ptr < end_batch) ptr++;
        }

        std::vector<size_t> term_indices;
        
        while (ptr < end_batch) {
            const char* line_start = ptr;
            while (ptr < end_batch && *ptr != '\n') ptr++;
            const char* line_end = ptr;
            if (ptr < end_batch) ptr++; 

            if (line_end > line_start) {
                auto terms = process_document_get_terms(line_start, line_end);
                
                term_indices.clear();
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
                indptr.push_back(static_cast<int32_t>(indices.size()));
            }
        }

        stream_offset_ = ptr - stream_file_.data;

        return py::make_tuple(
            py::array_t<uint16_t>(data.size(), data.data()),
            py::array_t<int32_t>(indices.size(), indices.data()),
            py::array_t<int32_t>(indptr.size(), indptr.data())
        );
    }

    py::list get_top_idf_words(size_t n = 10) {
        if (vocabulary_.empty()) throw std::runtime_error("Model not fitted");
        
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
        if (vocabulary_.empty()) throw std::runtime_error("Model not fitted");
        
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
        if (vocabulary_.empty()) throw std::runtime_error("Model not fitted");
        
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
        stats["min_df"] = min_df_;
        stats["max_df"] = max_df_;
        stats["max_features"] = max_features_;
        
        return stats;
    }
    
    py::object get_word_at_index(size_t idx) {
        if (idx >= vocabulary_.size()) return py::none();
        for (const auto& [word, word_idx] : vocabulary_) {
            if (word_idx == idx) return py::str(word);
        }
        return py::none();
    }
    
    bool has_word(const std::string& word) {
        return vocabulary_.find(word) != vocabulary_.end();
    }
    
    py::list get_words_in_idf_range(double min_idf, double max_idf, size_t max_results = 1000) {
        py::list result;
        size_t count = 0;
        for (const auto& [word, idx] : vocabulary_) {
            double word_idf = idf_[idx];
            if (word_idf >= min_idf && word_idf <= max_idf) {
                result.append(py::make_tuple(word, word_idf));
                if (++count >= max_results) break;
            }
        }
        return result;
    }
    
    py::list get_words_in_df_range(size_t min_df, size_t max_df, size_t max_results = 1000) {
        py::list result;
        size_t count = 0;
        for (const auto& [word, idx] : vocabulary_) {
            size_t df = (size_t)std::round(total_docs_ / std::exp(idf_[idx] - 1.0));
            if (df >= min_df && df <= max_df) {
                result.append(py::make_tuple(word, df, idf_[idx]));
                if (++count >= max_results) break;
            }
        }
        return result;
    }
    
    size_t get_vocab_size() const {
        return vocabulary_.size();
    }
    
    size_t get_total_docs() const {
        return total_docs_;
    }
    
    py::dict export_vocabulary_with_idf() {
        py::dict result;
        for (const auto& [word, idx] : vocabulary_) {
            result[py::str(word)] = idf_[idx];
        }
        return result;
    }
    
    py::list get_random_words(size_t n = 10, unsigned int seed = 42) {
        if (vocabulary_.empty()) throw std::runtime_error("Model not fitted");
        
        std::vector<std::pair<std::string, double>> all_words;
        for (const auto& [word, idx] : vocabulary_) {
            all_words.emplace_back(word, idf_[idx]);
        }
        
        std::mt19937 gen(seed);
        std::shuffle(all_words.begin(), all_words.end(), gen);
        
        py::list result;
        for (size_t i = 0; i < std::min(n, all_words.size()); i++) {
            result.append(py::make_tuple(all_words[i].first, all_words[i].second));
        }
        return result;
    }
};

PYBIND11_MODULE(fasttfidf, m) {
    m.doc() = "High-performance TF-IDF with SIMD (AVX2/NEON) and Zero-Copy Streaming";

    py::class_<TfidfVectorizer>(m, "TfidfVectorizer")
        .def(py::init<>())
        .def("fit", &TfidfVectorizer::fit,
             py::arg("filename"), py::arg("num_processes") = 0, py::arg("min_df") = 1,
             py::arg("max_df") = 0, py::arg("max_features") = 0, py::arg("verbose") = true)
        .def("get_vocabulary", &TfidfVectorizer::get_vocabulary)
        .def("get_idf", &TfidfVectorizer::get_idf)
        .def("get_feature_names", &TfidfVectorizer::get_feature_names)
        .def("save", &TfidfVectorizer::save, py::arg("filename"))
        .def("load", &TfidfVectorizer::load, py::arg("filename"))
        .def("get_idf_array", &TfidfVectorizer::get_idf_array)
        .def("open_stream", &TfidfVectorizer::open_stream)
        .def("get_batch", &TfidfVectorizer::get_batch)
        .def("get_top_idf_words", &TfidfVectorizer::get_top_idf_words, 
             py::arg("n") = 10)
        .def("get_bottom_idf_words", &TfidfVectorizer::get_bottom_idf_words,
             py::arg("n") = 10)
        .def("get_word_idf", &TfidfVectorizer::get_word_idf,
             py::arg("word"))
        .def("get_word_df", &TfidfVectorizer::get_word_df,
             py::arg("word"))
        .def("search_words", &TfidfVectorizer::search_words,
             py::arg("pattern"), py::arg("max_results") = 100)
        .def("get_vocab_stats", &TfidfVectorizer::get_vocab_stats)
        .def("get_word_at_index", &TfidfVectorizer::get_word_at_index,
             py::arg("idx"))
        .def("has_word", &TfidfVectorizer::has_word,
             py::arg("word"))
        .def("get_words_in_idf_range", &TfidfVectorizer::get_words_in_idf_range,
             py::arg("min_idf"), py::arg("max_idf"), py::arg("max_results") = 1000)
        .def("get_words_in_df_range", &TfidfVectorizer::get_words_in_df_range,
             py::arg("min_df"), py::arg("max_df"), py::arg("max_results") = 1000)
        .def("get_vocab_size", &TfidfVectorizer::get_vocab_size)
        .def("get_total_docs", &TfidfVectorizer::get_total_docs)
        .def("export_vocabulary_with_idf", &TfidfVectorizer::export_vocabulary_with_idf)
        .def("get_random_words", &TfidfVectorizer::get_random_words,
             py::arg("n") = 10, py::arg("seed") = 42);

#ifdef USE_SIMD
    m.attr("simd_enabled") = true;
    m.attr("simd_type") = SIMD_TYPE;
#else
    m.attr("simd_enabled") = false;
#endif
}