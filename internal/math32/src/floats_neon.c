#include <arm_neon.h>

void _vdot_neon(float *a, float *b, long n, float* ret) {
    int epoch = n / 8; // Number of full vectors (size 4) to process with unroll factor of 2
    int remain = n % 8; // Number of elements left after processing full vectors
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);

    // Vectorized computation loop with loop unrolling
    for (int i = 0; i < epoch; i++) {
        float32x4_t v1_1 = vld1q_f32(a);
        float32x4_t v2_1 = vld1q_f32(b);
        float32x4_t v1_2 = vld1q_f32(a + 4);
        float32x4_t v2_2 = vld1q_f32(b + 4);

        sum1 = vmlaq_f32(sum1, v1_1, v2_1);
        sum2 = vmlaq_f32(sum2, v1_2, v2_2);

        a += 8;
        b += 8;
    }

    // Process remaining elements
    for (int i = 0; i < remain; i++) {
        sum1 = vsetq_lane_f32(vgetq_lane_f32(sum1, 0) + a[i] * b[i], sum1, 0);
    }

    // Horizontal sum of the vectors
    sum1 = vaddq_f32(sum1, sum2);
    sum1 = vpaddq_f32(sum1, sum1);
    sum1 = vpaddq_f32(sum1, sum1);
    float32x2_t sum2_lanes = vget_low_f32(sum1);
    *ret = vget_lane_f32(sum2_lanes, 0);
}

void _squared_l2_neon(float *a, float *b, long n, float *result) {
    float32x4_t sum = vdupq_n_f32(0.0f);

    int epoch = n / 4;  // Number of full vectors (size 4) to process
    int remain = n % 4; // Number of elements left after processing full vectors

    // Process complete epochs (4 elements at a time)
    for (int i = 0; i < epoch; i++) {
        float32x4_t va = vld1q_f32(a + i * 4);
        float32x4_t vb = vld1q_f32(b + i * 4);
        
        float32x4_t diff = vsubq_f32(va, vb);
        float32x4_t square = vmulq_f32(diff, diff);
        
        sum = vaddq_f32(sum, square);
    }

    // Handle remaining elements
    if (remain > 0) {
        float32x4_t va_remain = vld1q_f32(a + epoch * 4);
        float32x4_t vb_remain = vld1q_f32(b + epoch * 4);

        // Calculate difference and square for remaining elements
        float32x4_t diff_remain = vsubq_f32(va_remain, vb_remain);
        float32x4_t square_remain = vmulq_f32(diff_remain, diff_remain);

        // Accumulate the squares of remaining elements
        
        // For remain = 1, extract the first element
        float32_t square_element_0 = vgetq_lane_f32(square_remain, 0);
        sum = vsetq_lane_f32(vgetq_lane_f32(sum, 0) + square_element_0, sum, 0);

        // For remain = 2, extract the second element
        if (remain > 1) {
            float32_t square_element_1 = vgetq_lane_f32(square_remain, 1);
            sum = vsetq_lane_f32(vgetq_lane_f32(sum, 1) + square_element_1, sum, 1);
        }

        // For remain = 3, extract the third element
        if (remain > 2) {
            float32_t square_element_2 = vgetq_lane_f32(square_remain, 2);
            sum = vsetq_lane_f32(vgetq_lane_f32(sum, 2) + square_element_2, sum, 2);
        }
    }

    // Reduce sum to a scalar value
    float32x2_t sum_low = vget_low_f32(sum);
    float32x2_t sum_high = vget_high_f32(sum);
    float32x2_t total_sum = vadd_f32(sum_low, sum_high);
    total_sum = vpadd_f32(total_sum, total_sum);

    // Extract the final result
    float32_t final_result;
    vst1_lane_f32(&final_result, total_sum, 0);
    *result = final_result;
}