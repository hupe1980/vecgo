#include <arm_neon.h>

void _dot_product_neon(float *a, float *b, long n, float* result)
{
    int epoch = n / 16; // Process 16 elements per iteration (4 accumulators)
    int remain = n % 16;
    
    // Four accumulators for better instruction-level parallelism
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);
    float32x4_t sum4 = vdupq_n_f32(0.0f);
    
    // Main vectorized loop: 16 floats per iteration
    for (int i = 0; i < epoch; ++i)
    {
        float32x4_t v1 = vld1q_f32(a);
        float32x4_t v2 = vld1q_f32(b);
        sum1 = vmlaq_f32(sum1, v1, v2);
        
        v1 = vld1q_f32(a + 4);
        v2 = vld1q_f32(b + 4);
        sum2 = vmlaq_f32(sum2, v1, v2);
        
        v1 = vld1q_f32(a + 8);
        v2 = vld1q_f32(b + 8);
        sum3 = vmlaq_f32(sum3, v1, v2);
        
        v1 = vld1q_f32(a + 12);
        v2 = vld1q_f32(b + 12);
        sum4 = vmlaq_f32(sum4, v1, v2);
        
        a += 16;
        b += 16;
    }

    // Process remaining elements
    float remain_sum = 0.0f;
    for (int i = 0; i < remain; ++i)
    {
        remain_sum += a[i] * b[i];
    }
    
    // Combine all four accumulators
    sum1 = vaddq_f32(sum1, sum2);
    sum3 = vaddq_f32(sum3, sum4);
    sum1 = vaddq_f32(sum1, sum3);
    
    // Horizontal reduction
    sum1 = vpaddq_f32(sum1, sum1);
    sum1 = vpaddq_f32(sum1, sum1);
    
    *result = vgetq_lane_f32(sum1, 0) + remain_sum;
}

void _squared_l2_neon(float *a, float *b, long n, float *result)
{
    // Four accumulators for better throughput
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);
    float32x4_t sum4 = vdupq_n_f32(0.0f);
    
    long epoch = n / 16;  // Process 16 elements per iteration
    long remainder = n % 16;

    for (long i = 0; i < epoch; ++i)
    {
        float32x4_t diff1 = vsubq_f32(vld1q_f32(a), vld1q_f32(b));
        sum1 = vmlaq_f32(sum1, diff1, diff1);
        
        float32x4_t diff2 = vsubq_f32(vld1q_f32(a + 4), vld1q_f32(b + 4));
        sum2 = vmlaq_f32(sum2, diff2, diff2);
        
        float32x4_t diff3 = vsubq_f32(vld1q_f32(a + 8), vld1q_f32(b + 8));
        sum3 = vmlaq_f32(sum3, diff3, diff3);
        
        float32x4_t diff4 = vsubq_f32(vld1q_f32(a + 12), vld1q_f32(b + 12));
        sum4 = vmlaq_f32(sum4, diff4, diff4);
        
        a += 16;
        b += 16;
    }
    
    // Combine all four accumulators
    sum1 = vaddq_f32(sum1, sum2);
    sum3 = vaddq_f32(sum3, sum4);
    sum1 = vaddq_f32(sum1, sum3);
    
    // Horizontal reduction
    sum1 = vpaddq_f32(sum1, sum1);
    sum1 = vpaddq_f32(sum1, sum1);
    
    float sum = vgetq_lane_f32(sum1, 0);

    // Scalar cleanup for remaining elements
    for (long i = 0; i < remainder; ++i) 
    {
        float diff = *a++ - *b++;
        sum += diff * diff;
    }

    *result = sum;
}

// _scale_neon scales a by (*scalar) in place.
//
// Note: scalar is passed by pointer to avoid ABI differences for float arguments
// in the Go assembly generator pipeline.
void _scale_neon(float *a, long n, float *scalar)
{
    if (n <= 0) {
        return;
    }

    float s = *scalar;
    float32x4_t sv = vdupq_n_f32(s);

    long epoch = n / 16;
    long remainder = n % 16;

    for (long i = 0; i < epoch; ++i)
    {
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

    for (long i = 0; i < remainder; ++i)
    {
        a[i] *= s;
    }
}
