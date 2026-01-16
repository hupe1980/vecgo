// Bounded L2 distance with early exit - ARM NEON implementation
#include <arm_neon.h>
#include <stdint.h>

// Helper: horizontal sum of float32x4_t
static inline float hsum_neon(float32x4_t v) {
    float32x2_t sum = vadd_f32(vget_low_f32(v), vget_high_f32(v));
    sum = vpadd_f32(sum, sum);
    return vget_lane_f32(sum, 0);
}

// SquaredL2BoundedNeon computes squared L2 with early exit when exceeding bound.
void squaredL2BoundedNeon(
    float* __restrict__ vec1,
    float* __restrict__ vec2,
    int64_t n,
    float bound,
    float* __restrict__ result,
    int32_t* __restrict__ exceeded
) {
    // Four accumulators for better ILP (16 floats per iteration = 4 x 4)
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);
    float32x4_t sum4 = vdupq_n_f32(0.0f);
    
    float total = 0.0f;
    int64_t i = 0;
    
    // Process in blocks of 64 elements (4 iterations of 16)
    // Check bound after each block
    const int64_t blockSize = 64;
    const int64_t innerBlock = 16;  // 4 x 4-wide NEON
    
    while (i + blockSize <= n) {
        // Process 64 elements
        for (int j = 0; j < 4; j++) {
            float32x4_t v1_1 = vld1q_f32(vec1 + i);
            float32x4_t v1_2 = vld1q_f32(vec1 + i + 4);
            float32x4_t v1_3 = vld1q_f32(vec1 + i + 8);
            float32x4_t v1_4 = vld1q_f32(vec1 + i + 12);
            
            float32x4_t v2_1 = vld1q_f32(vec2 + i);
            float32x4_t v2_2 = vld1q_f32(vec2 + i + 4);
            float32x4_t v2_3 = vld1q_f32(vec2 + i + 8);
            float32x4_t v2_4 = vld1q_f32(vec2 + i + 12);
            
            float32x4_t diff1 = vsubq_f32(v1_1, v2_1);
            float32x4_t diff2 = vsubq_f32(v1_2, v2_2);
            float32x4_t diff3 = vsubq_f32(v1_3, v2_3);
            float32x4_t diff4 = vsubq_f32(v1_4, v2_4);
            
            // FMA: sum += diff * diff
            sum1 = vfmaq_f32(sum1, diff1, diff1);
            sum2 = vfmaq_f32(sum2, diff2, diff2);
            sum3 = vfmaq_f32(sum3, diff3, diff3);
            sum4 = vfmaq_f32(sum4, diff4, diff4);
            
            i += innerBlock;
        }
        
        // Early exit check after 64 elements
        float32x4_t combined = vaddq_f32(vaddq_f32(sum1, sum2), vaddq_f32(sum3, sum4));
        total = hsum_neon(combined);
        
        if (total > bound) {
            *result = total;
            *exceeded = 1;
            return;
        }
    }
    
    // Process remaining blocks of 16
    while (i + innerBlock <= n) {
        float32x4_t v1_1 = vld1q_f32(vec1 + i);
        float32x4_t v1_2 = vld1q_f32(vec1 + i + 4);
        float32x4_t v1_3 = vld1q_f32(vec1 + i + 8);
        float32x4_t v1_4 = vld1q_f32(vec1 + i + 12);
        
        float32x4_t v2_1 = vld1q_f32(vec2 + i);
        float32x4_t v2_2 = vld1q_f32(vec2 + i + 4);
        float32x4_t v2_3 = vld1q_f32(vec2 + i + 8);
        float32x4_t v2_4 = vld1q_f32(vec2 + i + 12);
        
        float32x4_t diff1 = vsubq_f32(v1_1, v2_1);
        float32x4_t diff2 = vsubq_f32(v1_2, v2_2);
        float32x4_t diff3 = vsubq_f32(v1_3, v2_3);
        float32x4_t diff4 = vsubq_f32(v1_4, v2_4);
        
        sum1 = vfmaq_f32(sum1, diff1, diff1);
        sum2 = vfmaq_f32(sum2, diff2, diff2);
        sum3 = vfmaq_f32(sum3, diff3, diff3);
        sum4 = vfmaq_f32(sum4, diff4, diff4);
        
        i += innerBlock;
    }
    
    // Combine all accumulators
    float32x4_t combined = vaddq_f32(vaddq_f32(sum1, sum2), vaddq_f32(sum3, sum4));
    total = hsum_neon(combined);
    
    // Scalar cleanup
    for (; i < n; i++) {
        float diff = vec1[i] - vec2[i];
        total += diff * diff;
    }
    
    *result = total;
    *exceeded = (total > bound) ? 1 : 0;
}
