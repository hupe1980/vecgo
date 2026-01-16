// Query Bitmap SIMD operations - SVE2 implementation
// Generated ASM using: c2goasm -a -c -f bitmap_sve2.c
//
// These functions implement bitwise operations on uint64 arrays using SVE2.
// SVE2 is scalable (128-2048 bits), so we use predicated loops.

#include <stdint.h>
#include <arm_sve.h>

// andWordsSVE2Asm performs dst[i] &= src[i] using SVE2
void andWordsSVE2Asm(uint64_t* dst, const uint64_t* src, int64_t n) {
    int64_t i = 0;
    svbool_t pg = svwhilelt_b64(i, n);
    
    do {
        svuint64_t d = svld1_u64(pg, dst + i);
        svuint64_t s = svld1_u64(pg, src + i);
        svuint64_t r = svand_u64_z(pg, d, s);
        svst1_u64(pg, dst + i, r);
        
        i += svcntd();
        pg = svwhilelt_b64(i, n);
    } while (svptest_any(svptrue_b64(), pg));
}

// andNotWordsSVE2Asm performs dst[i] &= ~src[i] using SVE2
void andNotWordsSVE2Asm(uint64_t* dst, const uint64_t* src, int64_t n) {
    int64_t i = 0;
    svbool_t pg = svwhilelt_b64(i, n);
    
    do {
        svuint64_t d = svld1_u64(pg, dst + i);
        svuint64_t s = svld1_u64(pg, src + i);
        // BIC: d & ~s
        svuint64_t r = svbic_u64_z(pg, d, s);
        svst1_u64(pg, dst + i, r);
        
        i += svcntd();
        pg = svwhilelt_b64(i, n);
    } while (svptest_any(svptrue_b64(), pg));
}

// orWordsSVE2Asm performs dst[i] |= src[i] using SVE2
void orWordsSVE2Asm(uint64_t* dst, const uint64_t* src, int64_t n) {
    int64_t i = 0;
    svbool_t pg = svwhilelt_b64(i, n);
    
    do {
        svuint64_t d = svld1_u64(pg, dst + i);
        svuint64_t s = svld1_u64(pg, src + i);
        svuint64_t r = svorr_u64_z(pg, d, s);
        svst1_u64(pg, dst + i, r);
        
        i += svcntd();
        pg = svwhilelt_b64(i, n);
    } while (svptest_any(svptrue_b64(), pg));
}

// xorWordsSVE2Asm performs dst[i] ^= src[i] using SVE2
void xorWordsSVE2Asm(uint64_t* dst, const uint64_t* src, int64_t n) {
    int64_t i = 0;
    svbool_t pg = svwhilelt_b64(i, n);
    
    do {
        svuint64_t d = svld1_u64(pg, dst + i);
        svuint64_t s = svld1_u64(pg, src + i);
        svuint64_t r = sveor_u64_z(pg, d, s);
        svst1_u64(pg, dst + i, r);
        
        i += svcntd();
        pg = svwhilelt_b64(i, n);
    } while (svptest_any(svptrue_b64(), pg));
}

// NOTE: popcountWords removed - Go's bits.OnesCount64 compiles to hardware
// CNT instruction and is faster than explicit SIMD due to reduced overhead.
// See: CRoaring and bits-and-blooms/bitset implementations.
