// Query Bitmap SIMD operations - NEON implementation
//
// These functions implement bitwise operations on uint64 arrays using NEON.
// NEON processes 2 uint64 (128 bits) per vector, we process 4 vectors (8 words) per loop.

#include <stdint.h>
#include <arm_neon.h>

// andWordsNEONAsm performs dst[i] &= src[i] using NEON
void andWordsNEONAsm(uint64_t* __restrict__ dst, const uint64_t* __restrict__ src, int64_t n) {
    int64_t i = 0;
    // SIMD loop: process 8 words at a time
    for (; i + 8 <= n; i += 8) {
        uint64x2_t d0 = vld1q_u64(dst + i);
        uint64x2_t d1 = vld1q_u64(dst + i + 2);
        uint64x2_t d2 = vld1q_u64(dst + i + 4);
        uint64x2_t d3 = vld1q_u64(dst + i + 6);
        
        uint64x2_t s0 = vld1q_u64(src + i);
        uint64x2_t s1 = vld1q_u64(src + i + 2);
        uint64x2_t s2 = vld1q_u64(src + i + 4);
        uint64x2_t s3 = vld1q_u64(src + i + 6);
        
        vst1q_u64(dst + i, vandq_u64(d0, s0));
        vst1q_u64(dst + i + 2, vandq_u64(d1, s1));
        vst1q_u64(dst + i + 4, vandq_u64(d2, s2));
        vst1q_u64(dst + i + 6, vandq_u64(d3, s3));
    }
    // Scalar tail
    for (; i < n; i++) {
        dst[i] &= src[i];
    }
}

// andNotWordsNEONAsm performs dst[i] &= ~src[i] using NEON
void andNotWordsNEONAsm(uint64_t* __restrict__ dst, const uint64_t* __restrict__ src, int64_t n) {
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint64x2_t d0 = vld1q_u64(dst + i);
        uint64x2_t d1 = vld1q_u64(dst + i + 2);
        uint64x2_t d2 = vld1q_u64(dst + i + 4);
        uint64x2_t d3 = vld1q_u64(dst + i + 6);
        
        uint64x2_t s0 = vld1q_u64(src + i);
        uint64x2_t s1 = vld1q_u64(src + i + 2);
        uint64x2_t s2 = vld1q_u64(src + i + 4);
        uint64x2_t s3 = vld1q_u64(src + i + 6);
        
        // BIC: d & ~s
        vst1q_u64(dst + i, vbicq_u64(d0, s0));
        vst1q_u64(dst + i + 2, vbicq_u64(d1, s1));
        vst1q_u64(dst + i + 4, vbicq_u64(d2, s2));
        vst1q_u64(dst + i + 6, vbicq_u64(d3, s3));
    }
    for (; i < n; i++) {
        dst[i] &= ~src[i];
    }
}

// orWordsNEONAsm performs dst[i] |= src[i] using NEON
void orWordsNEONAsm(uint64_t* __restrict__ dst, const uint64_t* __restrict__ src, int64_t n) {
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint64x2_t d0 = vld1q_u64(dst + i);
        uint64x2_t d1 = vld1q_u64(dst + i + 2);
        uint64x2_t d2 = vld1q_u64(dst + i + 4);
        uint64x2_t d3 = vld1q_u64(dst + i + 6);
        
        uint64x2_t s0 = vld1q_u64(src + i);
        uint64x2_t s1 = vld1q_u64(src + i + 2);
        uint64x2_t s2 = vld1q_u64(src + i + 4);
        uint64x2_t s3 = vld1q_u64(src + i + 6);
        
        vst1q_u64(dst + i, vorrq_u64(d0, s0));
        vst1q_u64(dst + i + 2, vorrq_u64(d1, s1));
        vst1q_u64(dst + i + 4, vorrq_u64(d2, s2));
        vst1q_u64(dst + i + 6, vorrq_u64(d3, s3));
    }
    for (; i < n; i++) {
        dst[i] |= src[i];
    }
}

// xorWordsNEONAsm performs dst[i] ^= src[i] using NEON
void xorWordsNEONAsm(uint64_t* __restrict__ dst, const uint64_t* __restrict__ src, int64_t n) {
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint64x2_t d0 = vld1q_u64(dst + i);
        uint64x2_t d1 = vld1q_u64(dst + i + 2);
        uint64x2_t d2 = vld1q_u64(dst + i + 4);
        uint64x2_t d3 = vld1q_u64(dst + i + 6);
        
        uint64x2_t s0 = vld1q_u64(src + i);
        uint64x2_t s1 = vld1q_u64(src + i + 2);
        uint64x2_t s2 = vld1q_u64(src + i + 4);
        uint64x2_t s3 = vld1q_u64(src + i + 6);
        
        vst1q_u64(dst + i, veorq_u64(d0, s0));
        vst1q_u64(dst + i + 2, veorq_u64(d1, s1));
        vst1q_u64(dst + i + 4, veorq_u64(d2, s2));
        vst1q_u64(dst + i + 6, veorq_u64(d3, s3));
    }
    for (; i < n; i++) {
        dst[i] ^= src[i];
    }
}

// NOTE: popcountWords removed - Go's bits.OnesCount64 compiles to hardware
// CNT instruction and is faster than explicit SIMD due to reduced overhead.
// See: CRoaring and bits-and-blooms/bitset implementations.
