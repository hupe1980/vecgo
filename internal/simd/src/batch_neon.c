#include <arm_neon.h>
#include <stdint.h>

// SquaredL2BatchNeon computes squared L2 distance for a batch of vectors using NEON.
// query: pointer to the query vector (dim floats)
// targets: pointer to the flattened target vectors (n * dim floats)
// dim: dimension of the vectors
// n: number of target vectors
// out: pointer to the output array (n floats)
void squaredL2BatchNeon(float *__restrict__ query, float *__restrict__ targets, int64_t dim, int64_t n, float *__restrict__ out) {
    for (int64_t i = 0; i < n; i++) {
        float *target = targets + i * dim;
        
        float32x4_t sum1 = vdupq_n_f32(0.0f);
        float32x4_t sum2 = vdupq_n_f32(0.0f);
        float32x4_t sum3 = vdupq_n_f32(0.0f);
        float32x4_t sum4 = vdupq_n_f32(0.0f);

        int64_t j = 0;
        // Unrolled loop (16 floats per step)
        for (; j <= dim - 16; j += 16) {
            // Prefetch
            __builtin_prefetch(query + j + 64);
            __builtin_prefetch(target + j + 64);

            float32x4_t q1 = vld1q_f32(query + j);
            float32x4_t q2 = vld1q_f32(query + j + 4);
            float32x4_t q3 = vld1q_f32(query + j + 8);
            float32x4_t q4 = vld1q_f32(query + j + 12);

            float32x4_t t1 = vld1q_f32(target + j);
            float32x4_t t2 = vld1q_f32(target + j + 4);
            float32x4_t t3 = vld1q_f32(target + j + 8);
            float32x4_t t4 = vld1q_f32(target + j + 12);

            float32x4_t d1 = vsubq_f32(q1, t1);
            float32x4_t d2 = vsubq_f32(q2, t2);
            float32x4_t d3 = vsubq_f32(q3, t3);
            float32x4_t d4 = vsubq_f32(q4, t4);

            sum1 = vfmaq_f32(sum1, d1, d1);
            sum2 = vfmaq_f32(sum2, d2, d2);
            sum3 = vfmaq_f32(sum3, d3, d3);
            sum4 = vfmaq_f32(sum4, d4, d4);
        }

        // Reduce accumulators
        sum1 = vaddq_f32(sum1, sum2);
        sum3 = vaddq_f32(sum3, sum4);
        sum1 = vaddq_f32(sum1, sum3);

        // Handle remaining 4-blocks
        for (; j <= dim - 4; j += 4) {
            float32x4_t q = vld1q_f32(query + j);
            float32x4_t t = vld1q_f32(target + j);
            float32x4_t d = vsubq_f32(q, t);
            sum1 = vfmaq_f32(sum1, d, d);
        }

        // Horizontal sum
        float total = vaddvq_f32(sum1);

        // Handle scalar remainder
        for (; j < dim; j++) {
            float d = query[j] - target[j];
            total += d * d;
        }

        out[i] = total;
    }
}

// DotBatchNeon computes dot product for a batch of vectors using NEON.
void dotBatchNeon(float *__restrict__ query, float *__restrict__ targets, int64_t dim, int64_t n, float *__restrict__ out) {
    for (int64_t i = 0; i < n; i++) {
        float *target = targets + i * dim;
        
        float32x4_t sum1 = vdupq_n_f32(0.0f);
        float32x4_t sum2 = vdupq_n_f32(0.0f);
        float32x4_t sum3 = vdupq_n_f32(0.0f);
        float32x4_t sum4 = vdupq_n_f32(0.0f);

        int64_t j = 0;
        // Unrolled loop (16 floats per step)
        for (; j <= dim - 16; j += 16) {
            // Prefetch
            __builtin_prefetch(query + j + 64);
            __builtin_prefetch(target + j + 64);

            float32x4_t q1 = vld1q_f32(query + j);
            float32x4_t q2 = vld1q_f32(query + j + 4);
            float32x4_t q3 = vld1q_f32(query + j + 8);
            float32x4_t q4 = vld1q_f32(query + j + 12);

            float32x4_t t1 = vld1q_f32(target + j);
            float32x4_t t2 = vld1q_f32(target + j + 4);
            float32x4_t t3 = vld1q_f32(target + j + 8);
            float32x4_t t4 = vld1q_f32(target + j + 12);

            sum1 = vfmaq_f32(sum1, q1, t1);
            sum2 = vfmaq_f32(sum2, q2, t2);
            sum3 = vfmaq_f32(sum3, q3, t3);
            sum4 = vfmaq_f32(sum4, q4, t4);
        }

        // Reduce accumulators
        sum1 = vaddq_f32(sum1, sum2);
        sum3 = vaddq_f32(sum3, sum4);
        sum1 = vaddq_f32(sum1, sum3);

        // Handle remaining 4-blocks
        for (; j <= dim - 4; j += 4) {
            float32x4_t q = vld1q_f32(query + j);
            float32x4_t t = vld1q_f32(target + j);
            sum1 = vfmaq_f32(sum1, q, t);
        }

        // Horizontal sum
        float total = vaddvq_f32(sum1);

        // Handle scalar remainder
        for (; j < dim; j++) {
            total += query[j] * target[j];
        }

        out[i] = total;
    }
}
