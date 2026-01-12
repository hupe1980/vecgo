// Optimized AVX-512 kernels with 4 accumulators + FMA + horizontal reduction intrinsic
#include <immintrin.h>
#include <stdint.h>

// Prefetch distance (in cache lines ahead). Tuned for L2 cache latency.
// 512 bytes = 8 cache lines = optimal for AVX-512 with larger vectors.
#define PREFETCH_AHEAD 512

// DotProductAvx512 computes dot product with 4-way accumulator unrolling + FMA.
// Processes 64 floats per iteration (4 accumulators × 16-wide AVX-512).
// Uses _mm512_reduce_add_ps() for efficient horizontal reduction.
void dotProductAvx512(float* __restrict vec1, float* __restrict vec2, int64_t n, float* __restrict result)
{
    // Four accumulators for better instruction-level parallelism
    __m512 sum1 = _mm512_setzero_ps();
    __m512 sum2 = _mm512_setzero_ps();
    __m512 sum3 = _mm512_setzero_ps();
    __m512 sum4 = _mm512_setzero_ps();
    
    int64_t epoch = n / 64;  // Process 64 elements per iteration
    int64_t i;
    
    for (i = 0; i < epoch; i++)
    {
        int64_t offset = i * 64;
        
        // Prefetch next iteration's data into L1 cache
        _mm_prefetch((const char*)(vec1 + offset + PREFETCH_AHEAD/4), _MM_HINT_T0);
        _mm_prefetch((const char*)(vec2 + offset + PREFETCH_AHEAD/4), _MM_HINT_T0);
        
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
    int64_t remainder = n % 64;
    for (i = n - remainder; i < n; i++)
    {
        total += vec1[i] * vec2[i];
    }
    
    *result = total;
}

// SquaredL2Avx512 computes squared L2 distance with 4-way accumulator unrolling + FMA.
// Processes 64 floats per iteration (4 accumulators × 16-wide AVX-512).
void squaredL2Avx512(float* __restrict vec1, float* __restrict vec2, int64_t n, float* __restrict result)
{
    // Four accumulators for better instruction-level parallelism
    __m512 sum1 = _mm512_setzero_ps();
    __m512 sum2 = _mm512_setzero_ps();
    __m512 sum3 = _mm512_setzero_ps();
    __m512 sum4 = _mm512_setzero_ps();
    
    int64_t epoch = n / 64;
    int64_t i;
    
    for (i = 0; i < epoch; i++)
    {
        int64_t offset = i * 64;
        
        // Prefetch next iteration's data into L1 cache
        _mm_prefetch((const char*)(vec1 + offset + PREFETCH_AHEAD/4), _MM_HINT_T0);
        _mm_prefetch((const char*)(vec2 + offset + PREFETCH_AHEAD/4), _MM_HINT_T0);
        
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
    int64_t remainder = n % 64;
    for (i = n - remainder; i < n; i++)
    {
        float diff = vec1[i] - vec2[i];
        total += diff * diff;
    }
    
    *result = total;
}
// PqAdcLookupAvx512 computes the sum of distances from a precomputed table.
// table: M x 256 floats (flattened)
// codes: M bytes
// m: number of subvectors
// result: pointer to float result
void pqAdcLookupAvx512(float *__restrict table, uint8_t *__restrict codes, int64_t m, float *__restrict result, const __m512i *offsets_ptr)
{
    __m512 sum_vec = _mm512_setzero_ps();
    
    // Load offsets via pointer to avoid .rodata constants (generator requires no relocations).
    __m512i offsets = _mm512_loadu_si512(offsets_ptr);
    
    int64_t i;
    for (i = 0; i <= m - 16; i += 16)
    {
        // Load 16 codes
        __m128i c_vec = _mm_loadu_si128((__m128i*)(codes + i));
        __m512i c_ints = _mm512_cvtepu8_epi32(c_vec);
        
        __m512i indices = _mm512_add_epi32(offsets, c_ints);
        
        __m512 vals = _mm512_i32gather_ps(indices, table, 4);
        
        sum_vec = _mm512_add_ps(sum_vec, vals);
        
        table += 4096; // 16 * 256
    }
    
    float total = _mm512_reduce_add_ps(sum_vec);
    
    for (; i < m; i++)
    {
        total += table[codes[i]];
        table += 256;
    }
    
    *result = total;
}