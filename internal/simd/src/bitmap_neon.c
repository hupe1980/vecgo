// Query Bitmap SIMD operations - NEON implementation
// Generated ASM using: c2goasm -a -c -f bitmap_neon.c
//
// These functions implement bitwise operations on uint64 arrays using NEON.
// NEON processes 2 uint64 (128 bits) per vector, we process 4 vectors (8 words) per loop.

#include <stdint.h>
#include <arm_neon.h>

// andWordsNEONAsm performs dst[i] &= src[i] using NEON
// Parameters: dst, src pointers, n = number of uint64 (must be multiple of 8)
void andWordsNEONAsm(uint64_t* dst, const uint64_t* src, int64_t n) {
    for (int64_t i = 0; i < n; i += 8) {
        // Load 4 NEON registers (8 uint64 total)
        uint64x2_t d0 = vld1q_u64(dst + i);
        uint64x2_t d1 = vld1q_u64(dst + i + 2);
        uint64x2_t d2 = vld1q_u64(dst + i + 4);
        uint64x2_t d3 = vld1q_u64(dst + i + 6);
        
        uint64x2_t s0 = vld1q_u64(src + i);
        uint64x2_t s1 = vld1q_u64(src + i + 2);
        uint64x2_t s2 = vld1q_u64(src + i + 4);
        uint64x2_t s3 = vld1q_u64(src + i + 6);
        
        uint64x2_t r0 = vandq_u64(d0, s0);
        uint64x2_t r1 = vandq_u64(d1, s1);
        uint64x2_t r2 = vandq_u64(d2, s2);
        uint64x2_t r3 = vandq_u64(d3, s3);
        
        vst1q_u64(dst + i, r0);
        vst1q_u64(dst + i + 2, r1);
        vst1q_u64(dst + i + 4, r2);
        vst1q_u64(dst + i + 6, r3);
    }
}

// andNotWordsNEONAsm performs dst[i] &= ~src[i] using NEON
void andNotWordsNEONAsm(uint64_t* dst, const uint64_t* src, int64_t n) {
    for (int64_t i = 0; i < n; i += 8) {
        uint64x2_t d0 = vld1q_u64(dst + i);
        uint64x2_t d1 = vld1q_u64(dst + i + 2);
        uint64x2_t d2 = vld1q_u64(dst + i + 4);
        uint64x2_t d3 = vld1q_u64(dst + i + 6);
        
        uint64x2_t s0 = vld1q_u64(src + i);
        uint64x2_t s1 = vld1q_u64(src + i + 2);
        uint64x2_t s2 = vld1q_u64(src + i + 4);
        uint64x2_t s3 = vld1q_u64(src + i + 6);
        
        // BIC: d & ~s
        uint64x2_t r0 = vbicq_u64(d0, s0);
        uint64x2_t r1 = vbicq_u64(d1, s1);
        uint64x2_t r2 = vbicq_u64(d2, s2);
        uint64x2_t r3 = vbicq_u64(d3, s3);
        
        vst1q_u64(dst + i, r0);
        vst1q_u64(dst + i + 2, r1);
        vst1q_u64(dst + i + 4, r2);
        vst1q_u64(dst + i + 6, r3);
    }
}

// orWordsNEONAsm performs dst[i] |= src[i] using NEON
void orWordsNEONAsm(uint64_t* dst, const uint64_t* src, int64_t n) {
    for (int64_t i = 0; i < n; i += 8) {
        uint64x2_t d0 = vld1q_u64(dst + i);
        uint64x2_t d1 = vld1q_u64(dst + i + 2);
        uint64x2_t d2 = vld1q_u64(dst + i + 4);
        uint64x2_t d3 = vld1q_u64(dst + i + 6);
        
        uint64x2_t s0 = vld1q_u64(src + i);
        uint64x2_t s1 = vld1q_u64(src + i + 2);
        uint64x2_t s2 = vld1q_u64(src + i + 4);
        uint64x2_t s3 = vld1q_u64(src + i + 6);
        
        uint64x2_t r0 = vorrq_u64(d0, s0);
        uint64x2_t r1 = vorrq_u64(d1, s1);
        uint64x2_t r2 = vorrq_u64(d2, s2);
        uint64x2_t r3 = vorrq_u64(d3, s3);
        
        vst1q_u64(dst + i, r0);
        vst1q_u64(dst + i + 2, r1);
        vst1q_u64(dst + i + 4, r2);
        vst1q_u64(dst + i + 6, r3);
    }
}

// xorWordsNEONAsm performs dst[i] ^= src[i] using NEON
void xorWordsNEONAsm(uint64_t* dst, const uint64_t* src, int64_t n) {
    for (int64_t i = 0; i < n; i += 8) {
        uint64x2_t d0 = vld1q_u64(dst + i);
        uint64x2_t d1 = vld1q_u64(dst + i + 2);
        uint64x2_t d2 = vld1q_u64(dst + i + 4);
        uint64x2_t d3 = vld1q_u64(dst + i + 6);
        
        uint64x2_t s0 = vld1q_u64(src + i);
        uint64x2_t s1 = vld1q_u64(src + i + 2);
        uint64x2_t s2 = vld1q_u64(src + i + 4);
        uint64x2_t s3 = vld1q_u64(src + i + 6);
        
        uint64x2_t r0 = veorq_u64(d0, s0);
        uint64x2_t r1 = veorq_u64(d1, s1);
        uint64x2_t r2 = veorq_u64(d2, s2);
        uint64x2_t r3 = veorq_u64(d3, s3);
        
        vst1q_u64(dst + i, r0);
        vst1q_u64(dst + i + 2, r1);
        vst1q_u64(dst + i + 4, r2);
        vst1q_u64(dst + i + 6, r3);
    }
}

// NOTE: popcountWords removed - Go's bits.OnesCount64 compiles to hardware
// CNT instruction and is faster than explicit SIMD due to reduced overhead.
// See: CRoaring and bits-and-blooms/bitset implementations.
