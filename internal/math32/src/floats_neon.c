#include <arm_neon.h>
#include <stdint.h>

void dotProductNeon(float *__restrict a, float *__restrict b, int64_t n, float *__restrict result)
{
    int64_t epoch = n / 16; // Process 16 elements per iteration (4 accumulators)
    int64_t remain = n % 16;
    
    // Four accumulators for better instruction-level parallelism
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);
    float32x4_t sum4 = vdupq_n_f32(0.0f);
    
    // Main vectorized loop: 16 floats per iteration
    for (int64_t i = 0; i < epoch; ++i)
    {
        // Software prefetch to hide memory latency
        __builtin_prefetch(a + 64);
        __builtin_prefetch(b + 64);

        float32x4_t v1 = vld1q_f32(a);
        float32x4_t v2 = vld1q_f32(b);
        sum1 = vfmaq_f32(sum1, v1, v2);
        
        v1 = vld1q_f32(a + 4);
        v2 = vld1q_f32(b + 4);
        sum2 = vfmaq_f32(sum2, v1, v2);
        
        v1 = vld1q_f32(a + 8);
        v2 = vld1q_f32(b + 8);
        sum3 = vfmaq_f32(sum3, v1, v2);
        
        v1 = vld1q_f32(a + 12);
        v2 = vld1q_f32(b + 12);
        sum4 = vfmaq_f32(sum4, v1, v2);
        
        a += 16;
        b += 16;
    }

    // Process remaining elements
    float remain_sum = 0.0f;
    for (int64_t i = 0; i < remain; ++i)
    {
        remain_sum += a[i] * b[i];
    }
    
    // Combine all four accumulators
    sum1 = vaddq_f32(sum1, sum2);
    sum3 = vaddq_f32(sum3, sum4);
    sum1 = vaddq_f32(sum1, sum3);
    
    // Portable horizontal reduction (compatible with all ARMv8-A)
    float32x2_t low = vget_low_f32(sum1);
    float32x2_t high = vget_high_f32(sum1);
    float32x2_t sum_pair = vadd_f32(low, high);
    sum_pair = vpadd_f32(sum_pair, sum_pair);
    
    *result = vget_lane_f32(sum_pair, 0) + remain_sum;
}

void pqAdcLookupNeon(float *__restrict table, uint8_t *__restrict codes, int64_t m, float *__restrict result)
{
    float sum = 0.0f;
    int64_t i;
    // Unroll 4 times
    for (i = 0; i <= m - 4; i += 4)
    {
        sum += table[codes[i]];
        table += 256;
        sum += table[codes[i+1]];
        table += 256;
        sum += table[codes[i+2]];
        table += 256;
        sum += table[codes[i+3]];
        table += 256;
    }
    
    for (; i < m; i++)
    {
        sum += table[codes[i]];
        table += 256;
    }
    *result = sum;
}

void squaredL2Neon(float *__restrict a, float *__restrict b, int64_t n, float *__restrict result)
{
    // Four accumulators for better throughput
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);
    float32x4_t sum4 = vdupq_n_f32(0.0f);
    
    int64_t epoch = n / 16;  // Process 16 elements per iteration
    int64_t remainder = n % 16;

    for (int64_t i = 0; i < epoch; ++i)
    {
        // Software prefetch
        __builtin_prefetch(a + 64);
        __builtin_prefetch(b + 64);

        float32x4_t diff1 = vsubq_f32(vld1q_f32(a), vld1q_f32(b));
        sum1 = vfmaq_f32(sum1, diff1, diff1);
        
        float32x4_t diff2 = vsubq_f32(vld1q_f32(a + 4), vld1q_f32(b + 4));
        sum2 = vfmaq_f32(sum2, diff2, diff2);
        
        float32x4_t diff3 = vsubq_f32(vld1q_f32(a + 8), vld1q_f32(b + 8));
        sum3 = vfmaq_f32(sum3, diff3, diff3);
        
        float32x4_t diff4 = vsubq_f32(vld1q_f32(a + 12), vld1q_f32(b + 12));
        sum4 = vfmaq_f32(sum4, diff4, diff4);
        
        a += 16;
        b += 16;
    }
    
    // Combine all four accumulators
    sum1 = vaddq_f32(sum1, sum2);
    sum3 = vaddq_f32(sum3, sum4);
    sum1 = vaddq_f32(sum1, sum3);
    
    // Portable horizontal reduction
    float32x2_t low = vget_low_f32(sum1);
    float32x2_t high = vget_high_f32(sum1);
    float32x2_t sum_pair = vadd_f32(low, high);
    sum_pair = vpadd_f32(sum_pair, sum_pair);
    
    float sum = vget_lane_f32(sum_pair, 0);

    // Scalar cleanup for remaining elements
    for (int64_t i = 0; i < remainder; ++i) 
    {
        float diff = *a++ - *b++;
        sum += diff * diff;
    }

    *result = sum;
}

// scaleNeon scales a by (*scalar) in place.
//
// Note: scalar is passed by pointer to avoid ABI differences for float arguments
// in the Go assembly generator pipeline.
void scaleNeon(float *__restrict a, int64_t n, float *__restrict scalar)
{
    if (n <= 0) {
        return;
    }

    float s = *scalar;
    float32x4_t sv = vdupq_n_f32(s);

    int64_t epoch = n / 16;
    int64_t remainder = n % 16;

    for (int64_t i = 0; i < epoch; ++i)
    {
        __builtin_prefetch(a + 64);

        float32x4_t v1 = vld1q_f32(a);
        float32x4_t v2 = vld1q_f32(a + 4);
        float32x4_t v3 = vld1q_f32(a + 8);
        float32x4_t v4 = vld1q_f32(a + 12);

        v1 = vmulq_f32(v1, sv);
        v2 = vmulq_f32(v2, sv);
        v3 = vmulq_f32(v3, sv);
        v4 = vmulq_f32(v4, sv);

        vst1q_f32(a, v1);
        vst1q_f32(a + 4, v2);
        vst1q_f32(a + 8, v3);
        vst1q_f32(a + 12, v4);

        a += 16;
    }

    for (int64_t i = 0; i < remainder; ++i)
    {
        a[i] *= s;
    }
}