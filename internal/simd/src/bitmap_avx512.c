// Query Bitmap SIMD operations - AVX-512 implementation
// Generated ASM using: c2goasm -a -c -f bitmap_avx512.c
//
// These functions implement bitwise operations on uint64 arrays using AVX-512.
// Each function processes 8 uint64 (512 bits) per SIMD iteration.

#include <stdint.h>
#include <immintrin.h>

// andWordsAVX512Asm performs dst[i] &= src[i] using AVX-512
// Parameters: dst, src pointers, n = number of uint64 (must be multiple of 8)
void andWordsAVX512Asm(uint64_t* dst, const uint64_t* src, int64_t n) {
    for (int64_t i = 0; i < n; i += 8) {
        __m512i d = _mm512_loadu_si512((__m512i*)(dst + i));
        __m512i s = _mm512_loadu_si512((__m512i*)(src + i));
        __m512i r = _mm512_and_si512(d, s);
        _mm512_storeu_si512((__m512i*)(dst + i), r);
    }
}

// andNotWordsAVX512Asm performs dst[i] &= ~src[i] using AVX-512
void andNotWordsAVX512Asm(uint64_t* dst, const uint64_t* src, int64_t n) {
    for (int64_t i = 0; i < n; i += 8) {
        __m512i d = _mm512_loadu_si512((__m512i*)(dst + i));
        __m512i s = _mm512_loadu_si512((__m512i*)(src + i));
        // ANDNOT is: ~s & d
        __m512i r = _mm512_andnot_si512(s, d);
        _mm512_storeu_si512((__m512i*)(dst + i), r);
    }
}

// orWordsAVX512Asm performs dst[i] |= src[i] using AVX-512
void orWordsAVX512Asm(uint64_t* dst, const uint64_t* src, int64_t n) {
    for (int64_t i = 0; i < n; i += 8) {
        __m512i d = _mm512_loadu_si512((__m512i*)(dst + i));
        __m512i s = _mm512_loadu_si512((__m512i*)(src + i));
        __m512i r = _mm512_or_si512(d, s);
        _mm512_storeu_si512((__m512i*)(dst + i), r);
    }
}

// xorWordsAVX512Asm performs dst[i] ^= src[i] using AVX-512
void xorWordsAVX512Asm(uint64_t* dst, const uint64_t* src, int64_t n) {
    for (int64_t i = 0; i < n; i += 8) {
        __m512i d = _mm512_loadu_si512((__m512i*)(dst + i));
        __m512i s = _mm512_loadu_si512((__m512i*)(src + i));
        __m512i r = _mm512_xor_si512(d, s);
        _mm512_storeu_si512((__m512i*)(dst + i), r);
    }
}

// NOTE: popcountWords removed - Go's bits.OnesCount64 compiles to hardware
// POPCNT instruction and is faster than explicit SIMD due to reduced overhead.
// See: CRoaring and bits-and-blooms/bitset implementations.
