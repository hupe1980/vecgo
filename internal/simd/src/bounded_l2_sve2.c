// Bounded L2 distance with early exit - ARM SVE2 implementation
#include <arm_sve.h>
#include <stdint.h>

// SquaredL2BoundedSve2 computes squared L2 with early exit when exceeding bound.
// SVE2 uses predicated operations with vector-length-agnostic code.
void squaredL2BoundedSve2(
    float* __restrict__ vec1,
    float* __restrict__ vec2,
    int64_t n,
    float bound,
    float* __restrict__ result,
    int32_t* __restrict__ exceeded
) {
    // Get the number of float elements per SVE vector
    int64_t vl = svcntw();  // Vector length in 32-bit words
    
    // Four accumulators for better ILP
    svfloat32_t sum1 = svdup_f32(0.0f);
    svfloat32_t sum2 = svdup_f32(0.0f);
    svfloat32_t sum3 = svdup_f32(0.0f);
    svfloat32_t sum4 = svdup_f32(0.0f);
    
    float total = 0.0f;
    int64_t i = 0;
    
    // Block size for early exit check (process ~64 elements then check)
    int64_t blockSize = vl * 4;
    int64_t checkInterval = (64 / blockSize > 0) ? (64 / blockSize) * blockSize : blockSize;
    int64_t processed = 0;
    
    svbool_t pg = svptrue_b32();
    
    // Main loop with 4x unrolling
    while (i + vl * 4 <= n) {
        // Load 4 vectors
        svfloat32_t v1_1 = svld1_f32(pg, vec1 + i);
        svfloat32_t v1_2 = svld1_f32(pg, vec1 + i + vl);
        svfloat32_t v1_3 = svld1_f32(pg, vec1 + i + vl * 2);
        svfloat32_t v1_4 = svld1_f32(pg, vec1 + i + vl * 3);
        
        svfloat32_t v2_1 = svld1_f32(pg, vec2 + i);
        svfloat32_t v2_2 = svld1_f32(pg, vec2 + i + vl);
        svfloat32_t v2_3 = svld1_f32(pg, vec2 + i + vl * 2);
        svfloat32_t v2_4 = svld1_f32(pg, vec2 + i + vl * 3);
        
        // Compute differences
        svfloat32_t diff1 = svsub_f32_x(pg, v1_1, v2_1);
        svfloat32_t diff2 = svsub_f32_x(pg, v1_2, v2_2);
        svfloat32_t diff3 = svsub_f32_x(pg, v1_3, v2_3);
        svfloat32_t diff4 = svsub_f32_x(pg, v1_4, v2_4);
        
        // FMA: sum += diff * diff
        sum1 = svmla_f32_x(pg, sum1, diff1, diff1);
        sum2 = svmla_f32_x(pg, sum2, diff2, diff2);
        sum3 = svmla_f32_x(pg, sum3, diff3, diff3);
        sum4 = svmla_f32_x(pg, sum4, diff4, diff4);
        
        i += vl * 4;
        processed += vl * 4;
        
        // Early exit check approximately every 64 elements
        if (processed >= checkInterval) {
            // Combine accumulators using SVE horizontal reduction
            svfloat32_t combined = svadd_f32_x(pg, 
                svadd_f32_x(pg, sum1, sum2), 
                svadd_f32_x(pg, sum3, sum4));
            total = svaddv_f32(pg, combined);
            
            if (total > bound) {
                *result = total;
                *exceeded = 1;
                return;
            }
            processed = 0;
        }
    }
    
    // Final reduction of 4x unrolled accumulators
    svfloat32_t combined = svadd_f32_x(pg, 
        svadd_f32_x(pg, sum1, sum2), 
        svadd_f32_x(pg, sum3, sum4));
    total = svaddv_f32(pg, combined);
    
    // Single accumulator for remaining full vectors
    svfloat32_t tail_sum = svdup_f32(0.0f);
    
    // Process remaining full vectors one at a time
    while (i + vl <= n) {
        svfloat32_t v1 = svld1_f32(pg, vec1 + i);
        svfloat32_t v2 = svld1_f32(pg, vec2 + i);
        svfloat32_t diff = svsub_f32_x(pg, v1, v2);
        tail_sum = svmla_f32_x(pg, tail_sum, diff, diff);
        i += vl;
    }
    
    // Add remaining full vectors to total
    total += svaddv_f32(pg, tail_sum);
    
    // Handle final tail elements with predicated load
    if (i < n) {
        svbool_t pg_tail = svwhilelt_b32_s64(i, n);
        svfloat32_t v1 = svld1_f32(pg_tail, vec1 + i);
        svfloat32_t v2 = svld1_f32(pg_tail, vec2 + i);
        svfloat32_t diff = svsub_f32_x(pg_tail, v1, v2);
        svfloat32_t sq = svmul_f32_x(pg_tail, diff, diff);
        total += svaddv_f32(pg_tail, sq);
    }
    
    *result = total;
    *exceeded = (total > bound) ? 1 : 0;
}
