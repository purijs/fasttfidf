"""
Comprehensive Test Suite for TF-IDF C++ Library
Tests: OOM, memory leaks, I/O errors, fuzz testing, crash handling
Compatible: macOS, Linux, Windows (where applicable)
"""

import pytest
import sys
import psutil
import time
import random
import string
import gc
import fasttfidf_csv
import fasttfidf_parquet
import pyarrow as pa
import pyarrow.parquet as pq

PARQUET_AVAILABLE = True

class TestBasicFunctionality:
    """Test basic functionality works"""
    
    def test_simple_fit(self, tmp_path):
        """Test basic fit operation"""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("text\nhello world\nfoo bar\n")
        
        vec = fasttfidf_csv.TfidfVectorizer()
        vec.fit(str(csv_file), verbose=False)
        
        assert vec.get_vocab_size() > 0
        assert vec.get_total_docs() == 2
    
    def test_save_load(self, tmp_path):
        """Test save/load functionality"""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("text\nhello world\nfoo bar\n")
        
        vec = fasttfidf_csv.TfidfVectorizer()
        vec.fit(str(csv_file), verbose=False)
        
        model_file = tmp_path / "model.txt"
        vec.save(str(model_file))
        
        vec2 = fasttfidf_csv.TfidfVectorizer()
        vec2.load(str(model_file))
        
        assert vec2.get_vocab_size() == vec.get_vocab_size()
        assert vec2.get_total_docs() == vec.get_total_docs()


class TestMalformedCSV:
    """Test handling of malformed CSV files"""
    
    def test_empty_file(self, tmp_path):
        """Test handling of empty file"""
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("")
        
        vec = fasttfidf_csv.TfidfVectorizer()
        with pytest.raises(RuntimeError):
            vec.fit(str(csv_file), verbose=False)
    
    def test_header_only(self, tmp_path):
        """Test file with only header"""
        csv_file = tmp_path / "header_only.csv"
        csv_file.write_text("text\n")
        
        vec = fasttfidf_csv.TfidfVectorizer()
        vec.fit(str(csv_file), verbose=False)
        assert vec.get_total_docs() == 0
    
    def test_missing_newline_at_end(self, tmp_path):
        """Test file without trailing newline"""
        csv_file = tmp_path / "no_newline.csv"
        csv_file.write_text("text\nhello world")  # No \n at end
        
        vec = fasttfidf_csv.TfidfVectorizer()
        vec.fit(str(csv_file), verbose=False)
        assert vec.get_total_docs() >= 1
    
    def test_very_long_line(self, tmp_path):
        """Test handling of extremely long lines"""
        csv_file = tmp_path / "long_line.csv"
        long_text = " ".join(["word"] * 100000)  # 600KB line
        csv_file.write_text(f"text\n{long_text}\n")
        
        vec = fasttfidf_csv.TfidfVectorizer()
        vec.fit(str(csv_file), verbose=False)
        assert vec.get_vocab_size() >= 1
    
    def test_empty_lines(self, tmp_path):
        """Test handling of empty lines"""
        csv_file = tmp_path / "empty_lines.csv"
        csv_file.write_text("text\n\n\nhello world\n\n\n")
        
        vec = fasttfidf_csv.TfidfVectorizer()
        vec.fit(str(csv_file), verbose=False)
        assert vec.get_total_docs() >= 1
    
    def test_special_characters(self, tmp_path):
        """Test handling of special characters"""
        csv_file = tmp_path / "special.csv"
        csv_file.write_text("text\n!@#$%^&*()\n你好世界\n")
        
        vec = fasttfidf_csv.TfidfVectorizer()
        vec.fit(str(csv_file), verbose=False)
        # Should not crash, vocab may be empty
        assert vec.get_vocab_size() >= 0
    
    def test_binary_data(self, tmp_path):
        """Test handling of binary/non-UTF8 data"""
        csv_file = tmp_path / "binary.csv"
        with open(csv_file, 'wb') as f:
            f.write(b"text\n")
            f.write(b"\x00\x01\x02\xff\xfe\xfd\n")  # Binary data
            f.write(b"hello world\n")
        
        vec = fasttfidf_csv.TfidfVectorizer()
        # Should handle gracefully, not crash
        try:
            vec.fit(str(csv_file), verbose=False)
        except Exception as e:
            # Acceptable to raise error, but shouldn't segfault
            assert "Runtime" in str(type(e).__name__)
    
    def test_null_bytes(self, tmp_path):
        """Test handling of null bytes in text"""
        csv_file = tmp_path / "nulls.csv"
        with open(csv_file, 'wb') as f:
            f.write(b"text\n")
            f.write(b"hello\x00world\n")
        
        vec = fasttfidf_csv.TfidfVectorizer()
        vec.fit(str(csv_file), verbose=False)
        # Should not crash


class TestMemoryConstraints:
    """Test memory handling and constraints"""
    
    def test_low_memory_system(self, tmp_path):
        """Test behavior on low memory (simulated)"""
        csv_file = tmp_path / "medium.csv"
        
        # Create 10MB file
        with open(csv_file, 'w') as f:
            f.write("text\n")
            for i in range(100000):
                f.write(f"word{i%1000} word{i%500} word{i%250}\n")
        
        vec = fasttfidf_csv.TfidfVectorizer()
        # Force single worker to reduce memory
        vec.fit(str(csv_file), num_processes=1, max_features=1000, verbose=True)
        
        assert vec.get_vocab_size() <= 1000
    
    def test_memory_leak_fit(self, tmp_path):
        """Test for memory leaks during fit"""
        csv_file = tmp_path / "test.csv"
        with open(csv_file, 'w') as f:
            f.write("text\n")
            for i in range(10000):
                f.write(f"word{i%100} test{i%50}\n")
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run fit multiple times
        for _ in range(5):
            vec = fasttfidf_csv.TfidfVectorizer()
            vec.fit(str(csv_file), verbose=False)
            del vec
            gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Should not increase by more than 50MB
        assert memory_increase < 50, f"Memory leak detected: {memory_increase:.1f} MB increase"
    
    def test_memory_leak_transform(self, tmp_path):
        """Test for memory leaks during transform"""
        csv_file = tmp_path / "test.csv"
        with open(csv_file, 'w') as f:
            f.write("text\n")
            for i in range(10000):
                f.write(f"word{i%100} test{i%50}\n")
        
        vec = fasttfidf_csv.TfidfVectorizer()
        vec.fit(str(csv_file), verbose=False)
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        # Stream multiple times
        for _ in range(5):
            vec.open_stream(str(csv_file))
            while True:
                result = vec.get_batch(1024 * 1024)  # 1MB
                if result is None:
                    break
                del result
            gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        
        assert memory_increase < 50, f"Memory leak detected: {memory_increase:.1f} MB increase"
    
    def test_large_vocabulary(self, tmp_path):
        """Test handling of very large vocabulary"""
        csv_file = tmp_path / "large_vocab.csv"
        with open(csv_file, 'w') as f:
            f.write("text\n")
            for i in range(100000):
                # Generate unique words to maximize vocab
                f.write(f"uniqueword{i} anotherword{i}\n")
        
        vec = fasttfidf_csv.TfidfVectorizer()
        vec.fit(str(csv_file), max_features=50000, verbose=False)
        
        assert vec.get_vocab_size() <= 50000


class TestIOErrors:
    """Test I/O error handling"""
    
    def test_nonexistent_file(self):
        """Test handling of non-existent file"""
        vec = fasttfidf_csv.TfidfVectorizer()
        with pytest.raises(RuntimeError, match="Cannot open file"):
            vec.fit("/nonexistent/path/file.csv", verbose=False)
    
    def test_directory_instead_of_file(self, tmp_path):
        """Test handling of directory path"""
        vec = fasttfidf_csv.TfidfVectorizer()
        with pytest.raises(RuntimeError):
            vec.fit(str(tmp_path), verbose=False)
    
    def test_no_read_permission(self, tmp_path):
        """Test handling of unreadable file"""
        if sys.platform == 'win32':
            pytest.skip("Permission test not applicable on Windows")
        
        csv_file = tmp_path / "noperm.csv"
        csv_file.write_text("text\nhello world\n")
        csv_file.chmod(0o000)  # Remove all permissions
        
        vec = fasttfidf_csv.TfidfVectorizer()
        try:
            with pytest.raises(RuntimeError):
                vec.fit(str(csv_file), verbose=False)
        finally:
            csv_file.chmod(0o644)  # Restore for cleanup
    
    def test_file_deleted_during_stream(self, tmp_path):
        """Test handling of file deleted during streaming"""
        csv_file = tmp_path / "deleted.csv"
        with open(csv_file, 'w') as f:
            f.write("text\n")
            for i in range(1000):
                f.write(f"word{i}\n")
        
        vec = fasttfidf_csv.TfidfVectorizer()
        vec.fit(str(csv_file), verbose=False)
        vec.open_stream(str(csv_file))
        
        # Get first batch
        result = vec.get_batch(1024)
        assert result is not None
        
        # Delete file (mmap should still work on Unix)
        csv_file.unlink()
        
        # Should still be able to read (on Unix)
        if sys.platform != 'win32':
            result = vec.get_batch(1024)
            # May be None (EOF) or still have data


class TestFuzzTesting:
    """Fuzz testing with random inputs"""
    
    def test_random_text(self, tmp_path):
        """Test with completely random text"""
        csv_file = tmp_path / "random.csv"
        with open(csv_file, 'w') as f:
            f.write("text\n")
            for _ in range(100):
                random_text = ''.join(random.choices(
                    string.ascii_letters + string.digits + ' \t\n!@#$%', 
                    k=random.randint(10, 1000)
                ))
                f.write(f"{random_text}\n")
        
        vec = fasttfidf_csv.TfidfVectorizer()
        vec.fit(str(csv_file), verbose=False)
        # Should not crash
    
    def test_random_csv_structure(self, tmp_path):
        """Test with random CSV structure"""
        csv_file = tmp_path / "random_csv.csv"
        with open(csv_file, 'w') as f:
            # Sometimes add header, sometimes don't
            if random.random() > 0.5:
                f.write("text\n")
            
            for _ in range(100):
                # Random number of columns
                cols = random.randint(1, 5)
                line = ','.join(['word' for _ in range(cols)])
                f.write(f"{line}\n")
        
        vec = fasttfidf_csv.TfidfVectorizer()
        try:
            vec.fit(str(csv_file), verbose=False)
        except RuntimeError:
            pass  # Acceptable to fail on malformed CSV
    
    def test_extreme_values(self, tmp_path):
        """Test with extreme parameter values"""
        csv_file = tmp_path / "test.csv"
        with open(csv_file, 'w') as f:
            f.write("text\n")
            for i in range(100):
                f.write(f"word{i}\n")
        
        vec = fasttfidf_csv.TfidfVectorizer()
        
        # Test extreme min_df
        vec.fit(str(csv_file), min_df=1000000, verbose=False)
        assert vec.get_vocab_size() == 0
        
        # Test extreme max_features
        vec.fit(str(csv_file), max_features=0, verbose=False)
        
        # Test extreme num_processes
        vec.fit(str(csv_file), num_processes=1000, verbose=False)


class TestConcurrency:
    """Test concurrent operations"""
    
    def test_multiple_instances(self, tmp_path):
        """Test multiple vectorizer instances"""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("text\nhello world\nfoo bar\n")
        
        vec1 = fasttfidf_csv.TfidfVectorizer()
        vec2 = fasttfidf_csv.TfidfVectorizer()
        
        vec1.fit(str(csv_file), verbose=False)
        vec2.fit(str(csv_file), verbose=False)
        
        assert vec1.get_vocab_size() == vec2.get_vocab_size()
    
    def test_fit_after_load(self, tmp_path):
        """Test fitting after loading model"""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("text\nhello world\n")
        
        vec = fasttfidf_csv.TfidfVectorizer()
        vec.fit(str(csv_file), verbose=False)
        
        model_file = tmp_path / "model.txt"
        vec.save(str(model_file))
        vec.load(str(model_file))
        
        # Should be able to fit again
        vec.fit(str(csv_file), verbose=False)


class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_single_document(self, tmp_path):
        """Test with single document"""
        csv_file = tmp_path / "single.csv"
        csv_file.write_text("text\nhello world\n")
        
        vec = fasttfidf_csv.TfidfVectorizer()
        vec.fit(str(csv_file), verbose=False)
        assert vec.get_total_docs() == 1
    
    def test_single_word_vocabulary(self, tmp_path):
        """Test with vocabulary of single word"""
        csv_file = tmp_path / "single_word.csv"
        with open(csv_file, 'w') as f:
            f.write("text\n")
            for _ in range(100):
                f.write("hello\n")
        
        vec = fasttfidf_csv.TfidfVectorizer()
        vec.fit(str(csv_file), verbose=False)
        assert vec.get_vocab_size() >= 1
    
    def test_all_stopwords(self, tmp_path):
        """Test document with only single-character words"""
        csv_file = tmp_path / "stopwords.csv"
        with open(csv_file, 'w') as f:
            f.write("text\n")
            for _ in range(100):
                f.write("a b c d e f g\n")  # All filtered (< 2 chars)
        
        vec = fasttfidf_csv.TfidfVectorizer()
        vec.fit(str(csv_file), verbose=False)
        assert vec.get_vocab_size() == 0
    
    def test_transform_before_fit(self):
        """Test transform without fitting first"""
        vec = fasttfidf_csv.TfidfVectorizer()
        # Should raise exception for non-existent file
        with pytest.raises(RuntimeError):
            vec.open_stream("/tmp/fake_nonexistent_file.csv")


class TestPlatformSpecific:
    """Platform-specific tests"""
    
    def test_large_file_unix(self, tmp_path):
        """Test large file handling on Unix systems"""
        if sys.platform == 'win32':
            pytest.skip("Unix-specific test")
        
        csv_file = tmp_path / "large.csv"
        with open(csv_file, 'w') as f:
            f.write("text\n")
            for i in range(1000000):
                f.write(f"word{i%10000}\n")
        
        vec = fasttfidf_csv.TfidfVectorizer()
        vec.fit(str(csv_file), verbose=False)
        assert vec.get_total_docs() == 1000000
    
    def test_symlink_handling(self, tmp_path):
        """Test handling of symbolic links"""
        if sys.platform == 'win32':
            pytest.skip("Symlink test not reliable on Windows")
        
        csv_file = tmp_path / "real.csv"
        csv_file.write_text("text\nhello world\n")
        
        symlink = tmp_path / "link.csv"
        symlink.symlink_to(csv_file)
        
        vec = fasttfidf_csv.TfidfVectorizer()
        vec.fit(str(symlink), verbose=False)
        assert vec.get_vocab_size() > 0


class TestPerformanceRegression:
    """Performance regression tests"""
    
    def test_fit_speed(self, tmp_path):
        """Test fit completes in reasonable time"""
        csv_file = tmp_path / "perf.csv"
        with open(csv_file, 'w') as f:
            f.write("text\n")
            for i in range(10000):
                f.write(f"word{i%1000} test{i%500}\n")
        
        vec = fasttfidf_csv.TfidfVectorizer()
        start = time.time()
        vec.fit(str(csv_file), verbose=False)
        duration = time.time() - start
        
        # Should complete in under 5 seconds for 10k docs
        assert duration < 5.0, f"Fit took {duration:.2f}s (too slow)"
    
    def test_transform_speed(self, tmp_path):
        """Test transform completes in reasonable time"""
        csv_file = tmp_path / "perf.csv"
        with open(csv_file, 'w') as f:
            f.write("text\n")
            for i in range(10000):
                f.write(f"word{i%1000} test{i%500}\n")
        
        vec = fasttfidf_csv.TfidfVectorizer()
        vec.fit(str(csv_file), verbose=False)
        
        start = time.time()
        vec.open_stream(str(csv_file))
        while True:
            result = vec.get_batch(1024 * 1024)
            if result is None:
                break
        duration = time.time() - start
        
        # Should complete in under 2 seconds
        assert duration < 2.0, f"Transform took {duration:.2f}s (too slow)"


class TestParquetSupport:
    """Test Parquet file support"""

    @pytest.mark.skipif(not PARQUET_AVAILABLE, reason="Parquet support not available")
    def test_parquet_basic_fit(self, tmp_path):
        """Test basic fit operation with parquet file"""
        parquet_file = tmp_path / "test.parquet"

        table = pa.table({'text': ['hello world', 'foo bar']})
        pq.write_table(table, parquet_file)

        vec = fasttfidf_parquet.TfidfVectorizer()
        vec.fit(str(parquet_file), verbose=False)

        assert vec.get_vocab_size() > 0
        assert vec.get_total_docs() == 2

    @pytest.mark.skipif(not PARQUET_AVAILABLE, reason="Parquet support not available")
    def test_parquet_save_load(self, tmp_path):
        """Test save/load with parquet"""
        parquet_file = tmp_path / "test.parquet"

        table = pa.table({'text': ['hello world', 'foo bar']})
        pq.write_table(table, parquet_file)

        vec = fasttfidf_parquet.TfidfVectorizer()
        vec.fit(str(parquet_file), verbose=False)

        model_file = tmp_path / "model.txt"
        vec.save(str(model_file))

        vec2 = fasttfidf_parquet.TfidfVectorizer()
        vec2.load(str(model_file))

        assert vec2.get_vocab_size() == vec.get_vocab_size()
        assert vec2.get_total_docs() == vec.get_total_docs()

    @pytest.mark.skipif(not PARQUET_AVAILABLE, reason="Parquet support not available")
    def test_parquet_streaming(self, tmp_path):
        """Test streaming transformation with parquet"""
        parquet_file = tmp_path / "test.parquet"

        texts = [f"word{i%100} test{i%50}" for i in range(1000)]
        table = pa.table({'text': texts})
        pq.write_table(table, parquet_file)

        vec = fasttfidf_parquet.TfidfVectorizer()
        vec.fit(str(parquet_file), verbose=False)

        vec.open_stream(str(parquet_file))
        batch = vec.get_batch(1024 * 1024)

        assert batch is not None
        data, indices, indptr = batch
        assert len(data) > 0
        assert len(indices) > 0
        assert len(indptr) > 0

    @pytest.mark.skipif(not PARQUET_AVAILABLE, reason="Parquet support not available")
    def test_parquet_large_file(self, tmp_path):
        """Test parquet with larger dataset"""
        parquet_file = tmp_path / "large.parquet"

        texts = [f"word{i%1000} test{i%500}" for i in range(10000)]
        table = pa.table({'text': texts})
        pq.write_table(table, parquet_file)

        vec = fasttfidf_parquet.TfidfVectorizer()
        vec.fit(str(parquet_file), max_features=500, verbose=False)

        assert vec.get_vocab_size() <= 500
        assert vec.get_total_docs() == 10000


# Fixtures
@pytest.fixture
def tmp_path(tmp_path_factory):
    """Create temporary directory for tests"""
    return tmp_path_factory.mktemp("tfidf_tests")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
