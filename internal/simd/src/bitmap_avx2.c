// Query Bitmap SIMD operations - AVX2 implementation
// Generated ASM using: c2goasm -a -c -f bitmap_avx2.c
// 
// These functions implement bitwise operations on uint64 arrays using AVX2.
// Each function processes 4 uint64 (256 bits) per SIMD iteration.

#include <stdint.h>
#include <immintrin.h>

// andWordsAVX2Asm performs dst[i] &= src[i] using AVX2
// Parameters: dst, src pointers, n = number of uint64 (must be multiple of 4)
void andWordsAVX2Asm(uint64_t* dst, const uint64_t* src, int64_t n) {
    for (int64_t i = 0; i < n; i += 4) {
        __m256i d = _mm256_loadu_si256((__m256i*)(dst + i));
        __m256i s = _mm256_loadu_si256((__m256i*)(src + i));
        __m256i r = _mm256_and_si256(d, s);
        _mm256_storeu_si256((__m256i*)(dst + i), r);
    }
}

// andNotWordsAVX2Asm performs dst[i] &= ~src[i] using AVX2
void andNotWordsAVX2Asm(uint64_t* dst, const uint64_t* src, int64_t n) {
    for (int64_t i = 0; i < n; i += 4) {
        __m256i d = _mm256_loadu_si256((__m256i*)(dst + i));
        __m256i s = _mm256_loadu_si256((__m256i*)(src + i));
        // ANDNOT is: ~s & d (note the operand order in AVX)
        __m256i r = _mm256_andnot_si256(s, d);
        _mm256_storeu_si256((__m256i*)(dst + i), r);
    }
}

// orWordsAVX2Asm performs dst[i] |= src[i] using AVX2
void orWordsAVX2Asm(uint64_t* dst, const uint64_t* src, int64_t n) {
    for (int64_t i = 0; i < n; i += 4) {
        __m256i d = _mm256_loadu_si256((__m256i*)(dst + i));
        __m256i s = _mm256_loadu_si256((__m256i*)(src + i));
        __m256i r = _mm256_or_si256(d, s);
        _mm256_storeu_si256((__m256i*)(dst + i), r);
    }
}

// xorWordsAVX2Asm performs dst[i] ^= src[i] using AVX2
void xorWordsAVX2Asm(uint64_t* dst, const uint64_t* src, int64_t n) {
    for (int64_t i = 0; i < n; i += 4) {
        __m256i d = _mm256_loadu_si256((__m256i*)(dst + i));
        __m256i s = _mm256_loadu_si256((__m256i*)(src + i));
        __m256i r = _mm256_xor_si256(d, s);
        _mm256_storeu_si256((__m256i*)(dst + i), r);
    }
}

// NOTE: popcountWords removed - Go's bits.OnesCount64 compiles to hardware
// POPCNT instruction and is faster than explicit SIMD due to reduced overhead.
// See: CRoaring and bits-and-blooms/bitset implementations.
