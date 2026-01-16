// Bounded L2 distance with early exit - AVX-512 implementation
#include <immintrin.h>
#include <stdint.h>

// Helper: horizontal sum of __m512
static inline float hsum512(__m512 v) {
    __m256 lo = _mm512_castps512_ps256(v);
    __m256 hi = _mm512_extractf32x8_ps(v, 1);
    __m256 sum256 = _mm256_add_ps(lo, hi);
    __m128 lo128 = _mm256_castps256_ps128(sum256);
    __m128 hi128 = _mm256_extractf128_ps(sum256, 1);
    __m128 sum128 = _mm_add_ps(lo128, hi128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    return _mm_cvtss_f32(sum128);
}

// SquaredL2BoundedAvx512 computes squared L2 with early exit when exceeding bound.
void squaredL2BoundedAvx512(
    float* __restrict__ vec1,
    float* __restrict__ vec2,
    int64_t n,
    float bound,
    float* __restrict__ result,
    int32_t* __restrict__ exceeded
) {
    // Four accumulators for better ILP (64 floats per iteration = 4 x 16)
    __m512 sum1 = _mm512_setzero_ps();
    __m512 sum2 = _mm512_setzero_ps();
    __m512 sum3 = _mm512_setzero_ps();
    __m512 sum4 = _mm512_setzero_ps();
    
    float total = 0.0f;
    int64_t i = 0;
    
    // Process in blocks of 64 elements (4 x 16-wide AVX-512)
    const int64_t blockSize = 64;
    
    while (i + blockSize <= n) {
        // Software prefetch next cache lines
        _mm_prefetch((const char*)(vec1 + i + 64), _MM_HINT_T0);
        _mm_prefetch((const char*)(vec2 + i + 64), _MM_HINT_T0);
        
        // Load and compute differences for 64 elements
        __m512 v1_1 = _mm512_loadu_ps(vec1 + i);
        __m512 v1_2 = _mm512_loadu_ps(vec1 + i + 16);
        __m512 v1_3 = _mm512_loadu_ps(vec1 + i + 32);
        __m512 v1_4 = _mm512_loadu_ps(vec1 + i + 48);
        
        __m512 v2_1 = _mm512_loadu_ps(vec2 + i);
        __m512 v2_2 = _mm512_loadu_ps(vec2 + i + 16);
        __m512 v2_3 = _mm512_loadu_ps(vec2 + i + 32);
        __m512 v2_4 = _mm512_loadu_ps(vec2 + i + 48);
        
        __m512 diff1 = _mm512_sub_ps(v1_1, v2_1);
        __m512 diff2 = _mm512_sub_ps(v1_2, v2_2);
        __m512 diff3 = _mm512_sub_ps(v1_3, v2_3);
        __m512 diff4 = _mm512_sub_ps(v1_4, v2_4);
        
        // FMA: sum += diff * diff
        sum1 = _mm512_fmadd_ps(diff1, diff1, sum1);
        sum2 = _mm512_fmadd_ps(diff2, diff2, sum2);
        sum3 = _mm512_fmadd_ps(diff3, diff3, sum3);
        sum4 = _mm512_fmadd_ps(diff4, diff4, sum4);
        
        i += blockSize;
        
        // Early exit check every block (64 elements)
        // Combine accumulators and check
        __m512 combined = _mm512_add_ps(_mm512_add_ps(sum1, sum2), _mm512_add_ps(sum3, sum4));
        float blockSum = hsum512(combined);
        total = blockSum;
        
        if (total > bound) {
            *result = total;
            *exceeded = 1;
            return;
        }
    }
    
    // Combine all accumulators
    __m512 combined = _mm512_add_ps(_mm512_add_ps(sum1, sum2), _mm512_add_ps(sum3, sum4));
    total = hsum512(combined);
    
    // Handle remaining elements with AVX2
    for (; i + 8 <= n; i += 8) {
        __m256 v1 = _mm256_loadu_ps(vec1 + i);
        __m256 v2 = _mm256_loadu_ps(vec2 + i);
        __m256 diff = _mm256_sub_ps(v1, v2);
        __m256 sq = _mm256_mul_ps(diff, diff);
        
        __m128 lo = _mm256_castps256_ps128(sq);
        __m128 hi = _mm256_extractf128_ps(sq, 1);
        __m128 sum128 = _mm_add_ps(lo, hi);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        total += _mm_cvtss_f32(sum128);
    }
    
    // Scalar cleanup
    for (; i < n; i++) {
        float diff = vec1[i] - vec2[i];
        total += diff * diff;
    }
    
    *result = total;
    *exceeded = (total > bound) ? 1 : 0;
}
