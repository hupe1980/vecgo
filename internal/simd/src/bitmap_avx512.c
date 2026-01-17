// Query Bitmap SIMD operations - AVX-512 implementation
//
// These functions implement bitwise operations on uint64 arrays using AVX-512.
// Each function processes 8 uint64 (512 bits) per SIMD iteration with scalar tail.

#include <stdint.h>
#include <immintrin.h>

// andWordsAVX512Asm performs dst[i] &= src[i] using AVX-512
void andWordsAVX512Asm(uint64_t* __restrict__ dst, const uint64_t* __restrict__ src, int64_t n) {
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512i d = _mm512_loadu_si512((__m512i*)(dst + i));
        __m512i s = _mm512_loadu_si512((__m512i*)(src + i));
        _mm512_storeu_si512((__m512i*)(dst + i), _mm512_and_si512(d, s));
    }
    for (; i < n; i++) {
        dst[i] &= src[i];
    }
}

// andNotWordsAVX512Asm performs dst[i] &= ~src[i] using AVX-512
void andNotWordsAVX512Asm(uint64_t* __restrict__ dst, const uint64_t* __restrict__ src, int64_t n) {
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512i d = _mm512_loadu_si512((__m512i*)(dst + i));
        __m512i s = _mm512_loadu_si512((__m512i*)(src + i));
        // ANDNOT is: ~s & d
        _mm512_storeu_si512((__m512i*)(dst + i), _mm512_andnot_si512(s, d));
    }
    for (; i < n; i++) {
        dst[i] &= ~src[i];
    }
}

// orWordsAVX512Asm performs dst[i] |= src[i] using AVX-512
void orWordsAVX512Asm(uint64_t* __restrict__ dst, const uint64_t* __restrict__ src, int64_t n) {
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512i d = _mm512_loadu_si512((__m512i*)(dst + i));
        __m512i s = _mm512_loadu_si512((__m512i*)(src + i));
        _mm512_storeu_si512((__m512i*)(dst + i), _mm512_or_si512(d, s));
    }
    for (; i < n; i++) {
        dst[i] |= src[i];
    }
}

// xorWordsAVX512Asm performs dst[i] ^= src[i] using AVX-512
void xorWordsAVX512Asm(uint64_t* __restrict__ dst, const uint64_t* __restrict__ src, int64_t n) {
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512i d = _mm512_loadu_si512((__m512i*)(dst + i));
        __m512i s = _mm512_loadu_si512((__m512i*)(src + i));
        _mm512_storeu_si512((__m512i*)(dst + i), _mm512_xor_si512(d, s));
    }
    for (; i < n; i++) {
        dst[i] ^= src[i];
    }
}

// NOTE: popcountWords removed - Go's bits.OnesCount64 compiles to hardware
// POPCNT instruction and is faster than explicit SIMD due to reduced overhead.
// See: CRoaring and bits-and-blooms/bitset implementations.
