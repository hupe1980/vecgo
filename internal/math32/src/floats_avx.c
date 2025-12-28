// Optimized AVX2 kernels with 4 accumulators + FMA for maximum ILP
#include <immintrin.h> // AVX/AVX2 intrinsics

// _dot_product_avx computes dot product with 4-way accumulator unrolling + FMA.
// Processes 32 floats per iteration (4 accumulators × 8-wide AVX).
// Requires AVX2 for FMA support.
void _dot_product_avx(float *a, float *b, long n, float *res)
{
    // Four accumulators for better instruction-level parallelism
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();
    __m256 sum4 = _mm256_setzero_ps();
    
    long epoch = n / 32;  // Process 32 elements per iteration
    long i;
    
    for (i = 0; i < epoch; i++)
    {
        long offset = i * 32;
        
        // Load 32 floats (4 × 8-wide vectors)
        __m256 a1 = _mm256_loadu_ps(a + offset);
        __m256 a2 = _mm256_loadu_ps(a + offset + 8);
        __m256 a3 = _mm256_loadu_ps(a + offset + 16);
        __m256 a4 = _mm256_loadu_ps(a + offset + 24);
        
        __m256 b1 = _mm256_loadu_ps(b + offset);
        __m256 b2 = _mm256_loadu_ps(b + offset + 8);
        __m256 b3 = _mm256_loadu_ps(b + offset + 16);
        __m256 b4 = _mm256_loadu_ps(b + offset + 24);
        
        // Use FMA (fused multiply-add): sum = sum + a * b
        // This reduces instruction count by ~50% compared to separate mul + add
        sum1 = _mm256_fmadd_ps(a1, b1, sum1);
        sum2 = _mm256_fmadd_ps(a2, b2, sum2);
        sum3 = _mm256_fmadd_ps(a3, b3, sum3);
        sum4 = _mm256_fmadd_ps(a4, b4, sum4);
    }
    
    // Combine all four accumulators
    sum1 = _mm256_add_ps(sum1, sum2);
    sum3 = _mm256_add_ps(sum3, sum4);
    sum1 = _mm256_add_ps(sum1, sum3);
    
    // Horizontal reduction (sum all 8 lanes)
    float temp[8];
    _mm256_storeu_ps(temp, sum1);
    float result = temp[0] + temp[1] + temp[2] + temp[3] + 
                   temp[4] + temp[5] + temp[6] + temp[7];
    
    // Scalar cleanup for remaining elements
    long remainder = n % 32;
    for (i = n - remainder; i < n; i++)
    {
        result += a[i] * b[i];
    }
    
    *res = result;
}

// _squared_l2_avx computes squared L2 distance with 4-way accumulator unrolling + FMA.
// Processes 32 floats per iteration (4 accumulators × 8-wide AVX).
void _squared_l2_avx(float *vec1, float *vec2, long n, float *result)
{
    // Four accumulators for better instruction-level parallelism
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();
    __m256 sum4 = _mm256_setzero_ps();
    
    long epoch = n / 32;
    long i;
    
    for (i = 0; i < epoch; i++)
    {
        long offset = i * 32;
        
        // Load vectors
        __m256 v1_1 = _mm256_loadu_ps(vec1 + offset);
        __m256 v1_2 = _mm256_loadu_ps(vec1 + offset + 8);
        __m256 v1_3 = _mm256_loadu_ps(vec1 + offset + 16);
        __m256 v1_4 = _mm256_loadu_ps(vec1 + offset + 24);
        
        __m256 v2_1 = _mm256_loadu_ps(vec2 + offset);
        __m256 v2_2 = _mm256_loadu_ps(vec2 + offset + 8);
        __m256 v2_3 = _mm256_loadu_ps(vec2 + offset + 16);
        __m256 v2_4 = _mm256_loadu_ps(vec2 + offset + 24);
        
        // Compute differences
        __m256 diff1 = _mm256_sub_ps(v1_1, v2_1);
        __m256 diff2 = _mm256_sub_ps(v1_2, v2_2);
        __m256 diff3 = _mm256_sub_ps(v1_3, v2_3);
        __m256 diff4 = _mm256_sub_ps(v1_4, v2_4);
        
        // Use FMA to accumulate squared differences: sum = sum + diff * diff
        sum1 = _mm256_fmadd_ps(diff1, diff1, sum1);
        sum2 = _mm256_fmadd_ps(diff2, diff2, sum2);
        sum3 = _mm256_fmadd_ps(diff3, diff3, sum3);
        sum4 = _mm256_fmadd_ps(diff4, diff4, sum4);
    }
    
    // Combine all four accumulators
    sum1 = _mm256_add_ps(sum1, sum2);
    sum3 = _mm256_add_ps(sum3, sum4);
    sum1 = _mm256_add_ps(sum1, sum3);
    
    // Horizontal reduction
    float temp[8];
    _mm256_storeu_ps(temp, sum1);
    float sum = temp[0] + temp[1] + temp[2] + temp[3] + 
                temp[4] + temp[5] + temp[6] + temp[7];
    
    // Scalar cleanup for remaining elements
    long remainder = n % 32;
    for (i = n - remainder; i < n; i++)
    {
        float diff = vec1[i] - vec2[i];
        sum += diff * diff;
    }
    
    *result = sum;
}