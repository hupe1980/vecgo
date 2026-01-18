// AVX2 batch distance kernels
// Optimizations:
//   - 4-way accumulator unrolling for ILP
//   - Software prefetching for cache efficiency
//   - Optimized horizontal sum (no store/reload)
//   - FMA for fused multiply-add
#include <immintrin.h>
#include <stdint.h>

// Prefetch distance in bytes (4 cache lines = 256 bytes ahead)
#define PREFETCH_AHEAD 256

// Optimized horizontal sum: avoids store/load round-trip
static inline float hsum256_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);           // 4 floats
    sum = _mm_hadd_ps(sum, sum);               // 2 floats
    sum = _mm_hadd_ps(sum, sum);               // 1 float
    return _mm_cvtss_f32(sum);
}

// SquaredL2BatchAvx2 computes squared L2 distance for a batch of vectors using AVX2.
// query: pointer to the query vector (dim floats)
// targets: pointer to the flattened target vectors (n * dim floats)
// dim: dimension of the vectors
// n: number of target vectors
// out: pointer to the output array (n floats)
void squaredL2BatchAvx2(float *__restrict__ query, float *__restrict__ targets, int64_t dim, int64_t n, float *__restrict__ out) {
    for (int64_t i = 0; i < n; i++) {
        float *target = targets + i * dim;
        
        // Prefetch next target vector (critical for batch processing)
        if (i + 1 < n) {
            _mm_prefetch((const char*)(targets + (i + 1) * dim), _MM_HINT_T0);
        }
        
        __m256 sum1 = _mm256_setzero_ps();
        __m256 sum2 = _mm256_setzero_ps();
        __m256 sum3 = _mm256_setzero_ps();
        __m256 sum4 = _mm256_setzero_ps();

        int64_t j = 0;
        // Unrolled loop (32 floats per step)
        for (; j <= dim - 32; j += 32) {
            // Prefetch ahead within current vector
            _mm_prefetch((const char*)(target + j + PREFETCH_AHEAD/4), _MM_HINT_T0);
            
            __m256 q1 = _mm256_loadu_ps(query + j);
            __m256 q2 = _mm256_loadu_ps(query + j + 8);
            __m256 q3 = _mm256_loadu_ps(query + j + 16);
            __m256 q4 = _mm256_loadu_ps(query + j + 24);

            __m256 t1 = _mm256_loadu_ps(target + j);
            __m256 t2 = _mm256_loadu_ps(target + j + 8);
            __m256 t3 = _mm256_loadu_ps(target + j + 16);
            __m256 t4 = _mm256_loadu_ps(target + j + 24);

            __m256 d1 = _mm256_sub_ps(q1, t1);
            __m256 d2 = _mm256_sub_ps(q2, t2);
            __m256 d3 = _mm256_sub_ps(q3, t3);
            __m256 d4 = _mm256_sub_ps(q4, t4);

            sum1 = _mm256_fmadd_ps(d1, d1, sum1);
            sum2 = _mm256_fmadd_ps(d2, d2, sum2);
            sum3 = _mm256_fmadd_ps(d3, d3, sum3);
            sum4 = _mm256_fmadd_ps(d4, d4, sum4);
        }

        // Reduce accumulators
        sum1 = _mm256_add_ps(sum1, sum2);
        sum3 = _mm256_add_ps(sum3, sum4);
        sum1 = _mm256_add_ps(sum1, sum3);

        // Handle remaining 8-blocks
        for (; j <= dim - 8; j += 8) {
            __m256 q = _mm256_loadu_ps(query + j);
            __m256 t = _mm256_loadu_ps(target + j);
            __m256 d = _mm256_sub_ps(q, t);
            sum1 = _mm256_fmadd_ps(d, d, sum1);
        }

        // Optimized horizontal sum (register-only, no memory round-trip)
        float total = hsum256_ps(sum1);

        // Handle scalar remainder
        for (; j < dim; j++) {
            float d = query[j] - target[j];
            total += d * d;
        }

        out[i] = total;
    }
}

// DotBatchAvx2 computes dot product for a batch of vectors using AVX2.
void dotBatchAvx2(float *__restrict__ query, float *__restrict__ targets, int64_t dim, int64_t n, float *__restrict__ out) {
    for (int64_t i = 0; i < n; i++) {
        float *target = targets + i * dim;
        
        // Prefetch next target vector
        if (i + 1 < n) {
            _mm_prefetch((const char*)(targets + (i + 1) * dim), _MM_HINT_T0);
        }
        
        __m256 sum1 = _mm256_setzero_ps();
        __m256 sum2 = _mm256_setzero_ps();
        __m256 sum3 = _mm256_setzero_ps();
        __m256 sum4 = _mm256_setzero_ps();

        int64_t j = 0;
        // Unrolled loop (32 floats per step)
        for (; j <= dim - 32; j += 32) {
            // Prefetch ahead
            _mm_prefetch((const char*)(target + j + PREFETCH_AHEAD/4), _MM_HINT_T0);
            
            __m256 q1 = _mm256_loadu_ps(query + j);
            __m256 q2 = _mm256_loadu_ps(query + j + 8);
            __m256 q3 = _mm256_loadu_ps(query + j + 16);
            __m256 q4 = _mm256_loadu_ps(query + j + 24);

            __m256 t1 = _mm256_loadu_ps(target + j);
            __m256 t2 = _mm256_loadu_ps(target + j + 8);
            __m256 t3 = _mm256_loadu_ps(target + j + 16);
            __m256 t4 = _mm256_loadu_ps(target + j + 24);

            sum1 = _mm256_fmadd_ps(q1, t1, sum1);
            sum2 = _mm256_fmadd_ps(q2, t2, sum2);
            sum3 = _mm256_fmadd_ps(q3, t3, sum3);
            sum4 = _mm256_fmadd_ps(q4, t4, sum4);
        }

        // Reduce accumulators
        sum1 = _mm256_add_ps(sum1, sum2);
        sum3 = _mm256_add_ps(sum3, sum4);
        sum1 = _mm256_add_ps(sum1, sum3);

        // Handle remaining 8-blocks
        for (; j <= dim - 8; j += 8) {
            __m256 q = _mm256_loadu_ps(query + j);
            __m256 t = _mm256_loadu_ps(target + j);
            sum1 = _mm256_fmadd_ps(q, t, sum1);
        }

        // Optimized horizontal sum (register-only)
        float total = hsum256_ps(sum1);

        // Handle scalar remainder
        for (; j < dim; j++) {
            total += query[j] * target[j];
        }

        out[i] = total;
    }
}
