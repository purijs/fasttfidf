#pragma once

#include <string>
#include <unordered_set>
#include <unordered_map>
#include <cctype>
#include <cstring>
#include <algorithm>

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

constexpr size_t MAX_WORD_LENGTH = 127;

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

inline std::unordered_set<std::string> process_document_get_terms_avx2(const char* row_start, const char* row_end) {
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

inline std::unordered_set<std::string> process_document_get_terms_neon(const char* row_start, const char* row_end) {
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

inline std::unordered_set<std::string> process_document_get_terms_scalar(const char* row_start, const char* row_end) {
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