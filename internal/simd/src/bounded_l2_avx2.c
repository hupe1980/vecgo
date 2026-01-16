// Bounded L2 distance with early exit - AVX2 implementation
// Returns 1 if distance exceeded bound (early exit), 0 otherwise
#include <immintrin.h>
#include <stdint.h>

// Helper: horizontal sum of __m256
static inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

// SquaredL2BoundedAvx2 computes squared L2 with early exit when exceeding bound.
// Returns: exceeded (1 if distance > bound, 0 otherwise)
// Writes actual distance to *result
void squaredL2BoundedAvx2(
    float* __restrict__ vec1,
    float* __restrict__ vec2,
    int64_t n,
    float bound,
    float* __restrict__ result,
    int32_t* __restrict__ exceeded
) {
    // Four accumulators for better ILP (same as regular L2)
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();
    __m256 sum4 = _mm256_setzero_ps();
    
    float total = 0.0f;
    int64_t i = 0;
    
    // Process in blocks of 32 elements (4 x 8-wide AVX2)
    // Check bound after each block for early exit
    const int64_t blockSize = 32;
    
    while (i + blockSize <= n) {
        // Load and compute differences for 32 elements
        __m256 v1_1 = _mm256_loadu_ps(vec1 + i);
        __m256 v1_2 = _mm256_loadu_ps(vec1 + i + 8);
        __m256 v1_3 = _mm256_loadu_ps(vec1 + i + 16);
        __m256 v1_4 = _mm256_loadu_ps(vec1 + i + 24);
        
        __m256 v2_1 = _mm256_loadu_ps(vec2 + i);
        __m256 v2_2 = _mm256_loadu_ps(vec2 + i + 8);
        __m256 v2_3 = _mm256_loadu_ps(vec2 + i + 16);
        __m256 v2_4 = _mm256_loadu_ps(vec2 + i + 24);
        
        __m256 diff1 = _mm256_sub_ps(v1_1, v2_1);
        __m256 diff2 = _mm256_sub_ps(v1_2, v2_2);
        __m256 diff3 = _mm256_sub_ps(v1_3, v2_3);
        __m256 diff4 = _mm256_sub_ps(v1_4, v2_4);
        
        // FMA: sum += diff * diff
        sum1 = _mm256_fmadd_ps(diff1, diff1, sum1);
        sum2 = _mm256_fmadd_ps(diff2, diff2, sum2);
        sum3 = _mm256_fmadd_ps(diff3, diff3, sum3);
        sum4 = _mm256_fmadd_ps(diff4, diff4, sum4);
        
        i += blockSize;
        
        // Early exit check every 64 elements (2 blocks)
        // This balances check overhead vs early exit opportunity
        if ((i % 64) == 0) {
            // Combine accumulators and check
            __m256 combined = _mm256_add_ps(_mm256_add_ps(sum1, sum2), _mm256_add_ps(sum3, sum4));
            float blockSum = hsum256(combined);
            total = blockSum;
            
            if (total > bound) {
                *result = total;
                *exceeded = 1;
                return;
            }
        }
    }
    
    // Combine all accumulators
    __m256 combined = _mm256_add_ps(_mm256_add_ps(sum1, sum2), _mm256_add_ps(sum3, sum4));
    total = hsum256(combined);
    
    // Scalar cleanup for remaining elements
    for (; i < n; i++) {
        float diff = vec1[i] - vec2[i];
        total += diff * diff;
    }
    
    *result = total;
    *exceeded = (total > bound) ? 1 : 0;
}
