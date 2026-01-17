// Query Bitmap SIMD operations - AVX2 implementation
// 
// These functions implement bitwise operations on uint64 arrays using AVX2.
// Each function processes 4 uint64 (256 bits) per SIMD iteration with scalar tail.

#include <stdint.h>
#include <immintrin.h>

// andWordsAVX2Asm performs dst[i] &= src[i] using AVX2
void andWordsAVX2Asm(uint64_t* __restrict__ dst, const uint64_t* __restrict__ src, int64_t n) {
    int64_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256i d = _mm256_loadu_si256((__m256i*)(dst + i));
        __m256i s = _mm256_loadu_si256((__m256i*)(src + i));
        _mm256_storeu_si256((__m256i*)(dst + i), _mm256_and_si256(d, s));
    }
    for (; i < n; i++) {
        dst[i] &= src[i];
    }
}

// andNotWordsAVX2Asm performs dst[i] &= ~src[i] using AVX2
void andNotWordsAVX2Asm(uint64_t* __restrict__ dst, const uint64_t* __restrict__ src, int64_t n) {
    int64_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256i d = _mm256_loadu_si256((__m256i*)(dst + i));
        __m256i s = _mm256_loadu_si256((__m256i*)(src + i));
        // ANDNOT is: ~s & d (note the operand order in AVX)
        _mm256_storeu_si256((__m256i*)(dst + i), _mm256_andnot_si256(s, d));
    }
    for (; i < n; i++) {
        dst[i] &= ~src[i];
    }
}

// orWordsAVX2Asm performs dst[i] |= src[i] using AVX2
void orWordsAVX2Asm(uint64_t* __restrict__ dst, const uint64_t* __restrict__ src, int64_t n) {
    int64_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256i d = _mm256_loadu_si256((__m256i*)(dst + i));
        __m256i s = _mm256_loadu_si256((__m256i*)(src + i));
        _mm256_storeu_si256((__m256i*)(dst + i), _mm256_or_si256(d, s));
    }
    for (; i < n; i++) {
        dst[i] |= src[i];
    }
}

// xorWordsAVX2Asm performs dst[i] ^= src[i] using AVX2
void xorWordsAVX2Asm(uint64_t* __restrict__ dst, const uint64_t* __restrict__ src, int64_t n) {
    int64_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256i d = _mm256_loadu_si256((__m256i*)(dst + i));
        __m256i s = _mm256_loadu_si256((__m256i*)(src + i));
        _mm256_storeu_si256((__m256i*)(dst + i), _mm256_xor_si256(d, s));
    }
    for (; i < n; i++) {
        dst[i] ^= src[i];
    }
}

// NOTE: popcountWords removed - Go's bits.OnesCount64 compiles to hardware
// POPCNT instruction and is faster than explicit SIMD due to reduced overhead.
// See: CRoaring and bits-and-blooms/bitset implementations.
