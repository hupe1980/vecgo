// Optimized AVX-512 kernels with 4 accumulators + FMA + horizontal reduction intrinsic
#include <immintrin.h>

// _dot_product_avx512 computes dot product with 4-way accumulator unrolling + FMA.
// Processes 64 floats per iteration (4 accumulators × 16-wide AVX-512).
// Uses _mm512_reduce_add_ps() for efficient horizontal reduction.
void _dot_product_avx512(float* vec1, float* vec2, long n, float* result)
{
    // Four accumulators for better instruction-level parallelism
    __m512 sum1 = _mm512_setzero_ps();
    __m512 sum2 = _mm512_setzero_ps();
    __m512 sum3 = _mm512_setzero_ps();
    __m512 sum4 = _mm512_setzero_ps();
    
    long epoch = n / 64;  // Process 64 elements per iteration
    long i;
    
    for (i = 0; i < epoch; i++)
    {
        long offset = i * 64;
        
        // Load 64 floats (4 × 16-wide vectors)
        __m512 v1_1 = _mm512_loadu_ps(vec1 + offset);
        __m512 v1_2 = _mm512_loadu_ps(vec1 + offset + 16);
        __m512 v1_3 = _mm512_loadu_ps(vec1 + offset + 32);
        __m512 v1_4 = _mm512_loadu_ps(vec1 + offset + 48);
        
        __m512 v2_1 = _mm512_loadu_ps(vec2 + offset);
        __m512 v2_2 = _mm512_loadu_ps(vec2 + offset + 16);
        __m512 v2_3 = _mm512_loadu_ps(vec2 + offset + 32);
        __m512 v2_4 = _mm512_loadu_ps(vec2 + offset + 48);
        
        // Use FMA (fused multiply-add): sum = sum + v1 * v2
        sum1 = _mm512_fmadd_ps(v1_1, v2_1, sum1);
        sum2 = _mm512_fmadd_ps(v1_2, v2_2, sum2);
        sum3 = _mm512_fmadd_ps(v1_3, v2_3, sum3);
        sum4 = _mm512_fmadd_ps(v1_4, v2_4, sum4);
    }
    
    // Combine all four accumulators
    sum1 = _mm512_add_ps(sum1, sum2);
    sum3 = _mm512_add_ps(sum3, sum4);
    sum1 = _mm512_add_ps(sum1, sum3);
    
    // Use AVX-512 horizontal reduction intrinsic (cleaner than manual)
    float total = _mm512_reduce_add_ps(sum1);
    
    // Scalar cleanup for remaining elements
    long remainder = n % 64;
    for (i = n - remainder; i < n; i++)
    {
        total += vec1[i] * vec2[i];
    }
    
    *result = total;
}

// _squared_l2_avx512 computes squared L2 distance with 4-way accumulator unrolling + FMA.
// Processes 64 floats per iteration (4 accumulators × 16-wide AVX-512).
void _squared_l2_avx512(float* vec1, float* vec2, long n, float* result)
{
    // Four accumulators for better instruction-level parallelism
    __m512 sum1 = _mm512_setzero_ps();
    __m512 sum2 = _mm512_setzero_ps();
    __m512 sum3 = _mm512_setzero_ps();
    __m512 sum4 = _mm512_setzero_ps();
    
    long epoch = n / 64;
    long i;
    
    for (i = 0; i < epoch; i++)
    {
        long offset = i * 64;
        
        // Load vectors
        __m512 v1_1 = _mm512_loadu_ps(vec1 + offset);
        __m512 v1_2 = _mm512_loadu_ps(vec1 + offset + 16);
        __m512 v1_3 = _mm512_loadu_ps(vec1 + offset + 32);
        __m512 v1_4 = _mm512_loadu_ps(vec1 + offset + 48);
        
        __m512 v2_1 = _mm512_loadu_ps(vec2 + offset);
        __m512 v2_2 = _mm512_loadu_ps(vec2 + offset + 16);
        __m512 v2_3 = _mm512_loadu_ps(vec2 + offset + 32);
        __m512 v2_4 = _mm512_loadu_ps(vec2 + offset + 48);
        
        // Compute differences
        __m512 diff1 = _mm512_sub_ps(v1_1, v2_1);
        __m512 diff2 = _mm512_sub_ps(v1_2, v2_2);
        __m512 diff3 = _mm512_sub_ps(v1_3, v2_3);
        __m512 diff4 = _mm512_sub_ps(v1_4, v2_4);
        
        // Use FMA to accumulate squared differences
        sum1 = _mm512_fmadd_ps(diff1, diff1, sum1);
        sum2 = _mm512_fmadd_ps(diff2, diff2, sum2);
        sum3 = _mm512_fmadd_ps(diff3, diff3, sum3);
        sum4 = _mm512_fmadd_ps(diff4, diff4, sum4);
    }
    
    // Combine all four accumulators
    sum1 = _mm512_add_ps(sum1, sum2);
    sum3 = _mm512_add_ps(sum3, sum4);
    sum1 = _mm512_add_ps(sum1, sum3);
    
    // Use AVX-512 horizontal reduction intrinsic
    float total = _mm512_reduce_add_ps(sum1);
    
    // Scalar cleanup for remaining elements
    long remainder = n % 64;
    for (i = n - remainder; i < n; i++)
    {
        float diff = vec1[i] - vec2[i];
        total += diff * diff;
    }
    
    *result = total;
}
