#include <immintrin.h>
#include <stdint.h>

// SquaredL2BatchAvx512 computes squared L2 distance for a batch of vectors using AVX-512.
// query: pointer to the query vector (dim floats)
// targets: pointer to the flattened target vectors (n * dim floats)
// dim: dimension of the vectors
// n: number of target vectors
// out: pointer to the output array (n floats)
void squaredL2BatchAvx512(float *__restrict__ query, float *__restrict__ targets, int64_t dim, int64_t n, float *__restrict__ out) {
    for (int64_t i = 0; i < n; i++) {
        float *target = targets + i * dim;
        
        __m512 sum1 = _mm512_setzero_ps();
        __m512 sum2 = _mm512_setzero_ps();
        __m512 sum3 = _mm512_setzero_ps();
        __m512 sum4 = _mm512_setzero_ps();

        int64_t j = 0;
        // Unrolled loop (64 floats per step)
        for (; j <= dim - 64; j += 64) {
            __m512 q1 = _mm512_loadu_ps(query + j);
            __m512 q2 = _mm512_loadu_ps(query + j + 16);
            __m512 q3 = _mm512_loadu_ps(query + j + 32);
            __m512 q4 = _mm512_loadu_ps(query + j + 48);

            __m512 t1 = _mm512_loadu_ps(target + j);
            __m512 t2 = _mm512_loadu_ps(target + j + 16);
            __m512 t3 = _mm512_loadu_ps(target + j + 32);
            __m512 t4 = _mm512_loadu_ps(target + j + 48);

            __m512 d1 = _mm512_sub_ps(q1, t1);
            __m512 d2 = _mm512_sub_ps(q2, t2);
            __m512 d3 = _mm512_sub_ps(q3, t3);
            __m512 d4 = _mm512_sub_ps(q4, t4);

            sum1 = _mm512_fmadd_ps(d1, d1, sum1);
            sum2 = _mm512_fmadd_ps(d2, d2, sum2);
            sum3 = _mm512_fmadd_ps(d3, d3, sum3);
            sum4 = _mm512_fmadd_ps(d4, d4, sum4);
        }

        // Reduce accumulators
        sum1 = _mm512_add_ps(sum1, sum2);
        sum3 = _mm512_add_ps(sum3, sum4);
        sum1 = _mm512_add_ps(sum1, sum3);

        // Handle remaining 16-blocks
        for (; j <= dim - 16; j += 16) {
            __m512 q = _mm512_loadu_ps(query + j);
            __m512 t = _mm512_loadu_ps(target + j);
            __m512 d = _mm512_sub_ps(q, t);
            sum1 = _mm512_fmadd_ps(d, d, sum1);
        }

        // Horizontal sum
        float total = _mm512_reduce_add_ps(sum1);

        // Handle scalar remainder
        for (; j < dim; j++) {
            float d = query[j] - target[j];
            total += d * d;
        }

        out[i] = total;
    }
}

// DotBatchAvx512 computes dot product for a batch of vectors using AVX-512.
void dotBatchAvx512(float *__restrict__ query, float *__restrict__ targets, int64_t dim, int64_t n, float *__restrict__ out) {
    for (int64_t i = 0; i < n; i++) {
        float *target = targets + i * dim;
        
        __m512 sum1 = _mm512_setzero_ps();
        __m512 sum2 = _mm512_setzero_ps();
        __m512 sum3 = _mm512_setzero_ps();
        __m512 sum4 = _mm512_setzero_ps();

        int64_t j = 0;
        // Unrolled loop (64 floats per step)
        for (; j <= dim - 64; j += 64) {
            __m512 q1 = _mm512_loadu_ps(query + j);
            __m512 q2 = _mm512_loadu_ps(query + j + 16);
            __m512 q3 = _mm512_loadu_ps(query + j + 32);
            __m512 q4 = _mm512_loadu_ps(query + j + 48);

            __m512 t1 = _mm512_loadu_ps(target + j);
            __m512 t2 = _mm512_loadu_ps(target + j + 16);
            __m512 t3 = _mm512_loadu_ps(target + j + 32);
            __m512 t4 = _mm512_loadu_ps(target + j + 48);

            sum1 = _mm512_fmadd_ps(q1, t1, sum1);
            sum2 = _mm512_fmadd_ps(q2, t2, sum2);
            sum3 = _mm512_fmadd_ps(q3, t3, sum3);
            sum4 = _mm512_fmadd_ps(q4, t4, sum4);
        }

        // Reduce accumulators
        sum1 = _mm512_add_ps(sum1, sum2);
        sum3 = _mm512_add_ps(sum3, sum4);
        sum1 = _mm512_add_ps(sum1, sum3);

        // Handle remaining 16-blocks
        for (; j <= dim - 16; j += 16) {
            __m512 q = _mm512_loadu_ps(query + j);
            __m512 t = _mm512_loadu_ps(target + j);
            sum1 = _mm512_fmadd_ps(q, t, sum1);
        }

        // Horizontal sum
        float total = _mm512_reduce_add_ps(sum1);

        // Handle scalar remainder
        for (; j < dim; j++) {
            total += query[j] * target[j];
        }

        out[i] = total;
    }
}
