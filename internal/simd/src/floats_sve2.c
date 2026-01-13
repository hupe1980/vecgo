#ifdef __ARM_FEATURE_SVE2
#include <arm_sve.h>
#include <stdint.h>

// floats_sve2.c - SVE2 optimized vector operations
//
// SVE2 provides scalable vector lengths from 128-2048 bits, enabling
// efficient vectorization across different ARM implementations.

// dotProductSve2 computes dot product of two float32 vectors.
void dotProductSve2(float *__restrict a, float *__restrict b, int64_t n, float *__restrict result) {
    svfloat32_t sum = svdup_f32(0.0f);
    int64_t i = 0;
    
    svbool_t pg = svptrue_b32();
    int64_t vl = svcntw();
    
    // Main loop - process full SVE vectors
    for (; i + vl <= n; i += vl) {
        svfloat32_t va = svld1_f32(pg, a + i);
        svfloat32_t vb = svld1_f32(pg, b + i);
        sum = svmla_f32_x(pg, sum, va, vb);
    }
    
    // Handle tail with predication
    if (i < n) {
        svbool_t pg_tail = svwhilelt_b32_s64(i, n);
        svfloat32_t va = svld1_f32(pg_tail, a + i);
        svfloat32_t vb = svld1_f32(pg_tail, b + i);
        sum = svmla_f32_m(pg_tail, sum, va, vb);
    }
    
    // Horizontal sum
    *result = svaddv_f32(pg, sum);
}

// pqAdcLookupSve2 performs PQ ADC lookup with distance table.
void pqAdcLookupSve2(float *__restrict table, uint8_t *__restrict codes, int64_t m, float *__restrict result) {
    float sum = 0.0f;
    
    // PQ ADC lookup is inherently sequential due to table indexing
    // Use scalar loop with unrolling for efficiency
    int64_t i = 0;
    for (; i <= m - 4; i += 4) {
        sum += table[codes[i]];
        table += 256;
        sum += table[codes[i + 1]];
        table += 256;
        sum += table[codes[i + 2]];
        table += 256;
        sum += table[codes[i + 3]];
        table += 256;
    }
    
    for (; i < m; i++) {
        sum += table[codes[i]];
        table += 256;
    }
    
    *result = sum;
}

// squaredL2Sve2 computes squared L2 distance between two float32 vectors.
void squaredL2Sve2(float *__restrict a, float *__restrict b, int64_t n, float *__restrict result) {
    svfloat32_t sum = svdup_f32(0.0f);
    int64_t i = 0;
    
    svbool_t pg = svptrue_b32();
    int64_t vl = svcntw();
    
    // Main loop - process full SVE vectors
    for (; i + vl <= n; i += vl) {
        svfloat32_t va = svld1_f32(pg, a + i);
        svfloat32_t vb = svld1_f32(pg, b + i);
        svfloat32_t diff = svsub_f32_x(pg, va, vb);
        sum = svmla_f32_x(pg, sum, diff, diff);
    }
    
    // Handle tail with predication
    if (i < n) {
        svbool_t pg_tail = svwhilelt_b32_s64(i, n);
        svfloat32_t va = svld1_f32(pg_tail, a + i);
        svfloat32_t vb = svld1_f32(pg_tail, b + i);
        svfloat32_t diff = svsub_f32_x(pg_tail, va, vb);
        sum = svmla_f32_m(pg_tail, sum, diff, diff);
    }
    
    // Horizontal sum
    *result = svaddv_f32(pg, sum);
}

// scaleSve2 scales a vector by a scalar in place.
void scaleSve2(float *__restrict a, int64_t n, float *__restrict scalar) {
    if (n <= 0) {
        return;
    }
    
    float s = *scalar;
    svfloat32_t sv = svdup_f32(s);
    int64_t i = 0;
    
    svbool_t pg = svptrue_b32();
    int64_t vl = svcntw();
    
    // Main loop - process full SVE vectors
    for (; i + vl <= n; i += vl) {
        svfloat32_t va = svld1_f32(pg, a + i);
        svfloat32_t vr = svmul_f32_x(pg, va, sv);
        svst1_f32(pg, a + i, vr);
    }
    
    // Handle tail with predication
    if (i < n) {
        svbool_t pg_tail = svwhilelt_b32_s64(i, n);
        svfloat32_t va = svld1_f32(pg_tail, a + i);
        svfloat32_t vr = svmul_f32_x(pg_tail, va, sv);
        svst1_f32(pg_tail, a + i, vr);
    }
}

#endif // __ARM_FEATURE_SVE2
