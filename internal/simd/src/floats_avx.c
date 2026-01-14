// Optimized AVX2 kernels with 4 accumulators + FMA for maximum ILP
#include <immintrin.h> // AVX/AVX2 intrinsics
#include <stdint.h>

// Prefetch distance (in cache lines ahead). Tuned for L2 cache latency.
// 256 bytes = 4 cache lines = optimal for most x86 processors.
#define PREFETCH_AHEAD 256

// DotProductAvx computes dot product with 4-way accumulator unrolling + FMA.
// Processes 32 floats per iteration (4 accumulators × 8-wide AVX).
// Requires AVX2 for FMA support.
void dotProductAvx(float *__restrict a, float *__restrict b, int64_t n, float *__restrict res)
{
    // Four accumulators for better instruction-level parallelism
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();
    __m256 sum4 = _mm256_setzero_ps();
    
    int64_t epoch = n / 32;  // Process 32 elements per iteration
    int64_t i;
    
    for (i = 0; i < epoch; i++)
    {
        int64_t offset = i * 32;
        
        // Prefetch next iteration's data into L1 cache
        _mm_prefetch((const char*)(a + offset + PREFETCH_AHEAD/4), _MM_HINT_T0);
        _mm_prefetch((const char*)(b + offset + PREFETCH_AHEAD/4), _MM_HINT_T0);
        
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
    int64_t remainder = n % 32;
    for (i = n - remainder; i < n; i++)
    {
        result += a[i] * b[i];
    }
    
    *res = result;
}

// SquaredL2Avx computes squared L2 distance with 4-way accumulator unrolling + FMA.
// Processes 32 floats per iteration (4 accumulators × 8-wide AVX).
void squaredL2Avx(float *__restrict vec1, float *__restrict vec2, int64_t n, float *__restrict result)
{
    // Four accumulators for better instruction-level parallelism
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();
    __m256 sum4 = _mm256_setzero_ps();
    
    int64_t epoch = n / 32;
    int64_t i;
    
    for (i = 0; i < epoch; i++)
    {
        int64_t offset = i * 32;
        
        // Prefetch next iteration's data into L1 cache
        _mm_prefetch((const char*)(vec1 + offset + PREFETCH_AHEAD/4), _MM_HINT_T0);
        _mm_prefetch((const char*)(vec2 + offset + PREFETCH_AHEAD/4), _MM_HINT_T0);
        
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
    int64_t remainder = n % 32;
    for (i = n - remainder; i < n; i++)
    {
        float diff = vec1[i] - vec2[i];
        sum += diff * diff;
    }
    
    *result = sum;
}

// PqAdcLookupAvx computes the sum of distances from a precomputed table.
// table: M x 256 floats (flattened)
// codes: M bytes
// m: number of subvectors
// result: pointer to float result
void pqAdcLookupAvx(float *__restrict table, uint8_t *__restrict codes, int64_t m, float *__restrict result, const __m256i *offsets_ptr)
{
    __m256 sum_vec = _mm256_setzero_ps();
    
    // Load offsets via pointer to avoid .rodata constants (generator requires no relocations).
    __m256i offsets = _mm256_loadu_si256(offsets_ptr);
    
    int64_t i;
    for (i = 0; i <= m - 8; i += 8)
    {
        // Load 8 codes (bytes)
        long long c = *(long long*)(codes + i);
        __m128i c_vec = _mm_cvtsi64_si128(c);
        __m256i c_ints = _mm256_cvtepu8_epi32(c_vec);
        
        // Add offsets
        __m256i indices = _mm256_add_epi32(offsets, c_ints);
        
        // Gather
        __m256 vals = _mm256_i32gather_ps(table, indices, 4);
        
        sum_vec = _mm256_add_ps(sum_vec, vals);
        
        table += 2048; // 8 * 256
    }
    
    // Horizontal sum
    float temp[8];
    _mm256_storeu_ps(temp, sum_vec);
    float total = temp[0] + temp[1] + temp[2] + temp[3] + 
                  temp[4] + temp[5] + temp[6] + temp[7];
                  
    // Remainder
    for (; i < m; i++)
    {
        total += table[codes[i]];
        table += 256;
    }
    
    *result = total;
}

// ScaleAvx scales array a by (*scalar) in place.
// Processes 32 floats per iteration (4 × 8-wide AVX).
//
// Note: scalar is passed by pointer to avoid ABI differences for float arguments
// in the Go assembly generator pipeline.
void scaleAvx(float *__restrict a, int64_t n, float *__restrict scalar)
{
    if (n <= 0) {
        return;
    }

    float s = *scalar;
    __m256 sv = _mm256_set1_ps(s);

    int64_t epoch = n / 32;
    int64_t i;

    for (i = 0; i < epoch; i++)
    {
        int64_t offset = i * 32;

        // Prefetch next iteration's data
        _mm_prefetch((const char*)(a + offset + PREFETCH_AHEAD/4), _MM_HINT_T0);

        // Load 32 floats
        __m256 v1 = _mm256_loadu_ps(a + offset);
        __m256 v2 = _mm256_loadu_ps(a + offset + 8);
        __m256 v3 = _mm256_loadu_ps(a + offset + 16);
        __m256 v4 = _mm256_loadu_ps(a + offset + 24);

        // Multiply by scalar
        v1 = _mm256_mul_ps(v1, sv);
        v2 = _mm256_mul_ps(v2, sv);
        v3 = _mm256_mul_ps(v3, sv);
        v4 = _mm256_mul_ps(v4, sv);

        // Store back
        _mm256_storeu_ps(a + offset, v1);
        _mm256_storeu_ps(a + offset + 8, v2);
        _mm256_storeu_ps(a + offset + 16, v3);
        _mm256_storeu_ps(a + offset + 24, v4);
    }

    // Scalar cleanup for remaining elements
    int64_t remainder = n % 32;
    for (i = n - remainder; i < n; i++)
    {
        a[i] *= s;
    }
}