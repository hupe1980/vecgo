// SIMD filter operations for numeric indexing
// Compile with: clang -O3 -mavx2 -S -o filter_avx.s filter_avx.c
// Or for AVX-512: clang -O3 -mavx512f -mavx512bw -S -o filter_avx512.s filter_avx512.c

#include <immintrin.h>
#include <stdint.h>

// FilterRangeF64AVX2 - Check if values are in range [minVal, maxVal]
// Processes 4 float64 values at a time using AVX2
// 
// Arguments:
//   values: pointer to float64 array
//   n: number of elements
//   minVal: minimum value (inclusive)
//   maxVal: maximum value (inclusive)  
//   dst: output byte array (1 = in range, 0 = out of range)
void filterRangeF64Avx2(
    const double* __restrict__ values,
    int64_t n,
    double minVal,
    double maxVal,
    uint8_t* __restrict__ dst
) {
    // Broadcast min and max to all lanes
    __m256d vmin = _mm256_set1_pd(minVal);
    __m256d vmax = _mm256_set1_pd(maxVal);
    
    int64_t i = 0;
    
    // Process 4 doubles at a time (256 bits / 64 bits = 4)
    for (; i + 4 <= n; i += 4) {
        __m256d v = _mm256_loadu_pd(&values[i]);
        
        // Compare: v >= minVal AND v <= maxVal
        __m256d cmp_ge = _mm256_cmp_pd(v, vmin, _CMP_GE_OQ);  // v >= min
        __m256d cmp_le = _mm256_cmp_pd(v, vmax, _CMP_LE_OQ);  // v <= max
        __m256d in_range = _mm256_and_pd(cmp_ge, cmp_le);
        
        // Extract mask bits (4 bits, one per double)
        int mask = _mm256_movemask_pd(in_range);
        
        // Store individual bytes
        dst[i]     = (mask >> 0) & 1;
        dst[i + 1] = (mask >> 1) & 1;
        dst[i + 2] = (mask >> 2) & 1;
        dst[i + 3] = (mask >> 3) & 1;
    }
    
    // Handle remainder
    for (; i < n; i++) {
        dst[i] = (values[i] >= minVal && values[i] <= maxVal) ? 1 : 0;
    }
}

// FilterRangeF64IndicesAVX2 - Return indices of values in range [minVal, maxVal]
// More efficient when you need indices directly
// Writes count of matching indices to *countOut
void filterRangeF64IndicesAvx2(
    const double* __restrict__ values,
    int64_t n,
    double minVal,
    double maxVal,
    int32_t* __restrict__ dst,
    int64_t* __restrict__ countOut
) {
    __m256d vmin = _mm256_set1_pd(minVal);
    __m256d vmax = _mm256_set1_pd(maxVal);
    
    int64_t count = 0;
    int64_t i = 0;
    
    for (; i + 4 <= n; i += 4) {
        __m256d v = _mm256_loadu_pd(&values[i]);
        
        __m256d cmp_ge = _mm256_cmp_pd(v, vmin, _CMP_GE_OQ);
        __m256d cmp_le = _mm256_cmp_pd(v, vmax, _CMP_LE_OQ);
        __m256d in_range = _mm256_and_pd(cmp_ge, cmp_le);
        
        int mask = _mm256_movemask_pd(in_range);
        
        // Unrolled extraction of matching indices
        if (mask & 1) dst[count++] = (int32_t)(i);
        if (mask & 2) dst[count++] = (int32_t)(i + 1);
        if (mask & 4) dst[count++] = (int32_t)(i + 2);
        if (mask & 8) dst[count++] = (int32_t)(i + 3);
    }
    
    // Handle remainder
    for (; i < n; i++) {
        if (values[i] >= minVal && values[i] <= maxVal) {
            dst[count++] = (int32_t)i;
        }
    }
    
    *countOut = count;
}

// CountRangeF64AVX2 - Count values in range [minVal, maxVal]
// Optimized for just counting without storing indices
// Writes count to *countOut
void countRangeF64Avx2(
    const double* __restrict__ values,
    int64_t n,
    double minVal,
    double maxVal,
    int64_t* __restrict__ countOut
) {
    __m256d vmin = _mm256_set1_pd(minVal);
    __m256d vmax = _mm256_set1_pd(maxVal);
    
    int64_t count = 0;
    int64_t i = 0;
    
    for (; i + 4 <= n; i += 4) {
        __m256d v = _mm256_loadu_pd(&values[i]);
        
        __m256d cmp_ge = _mm256_cmp_pd(v, vmin, _CMP_GE_OQ);
        __m256d cmp_le = _mm256_cmp_pd(v, vmax, _CMP_LE_OQ);
        __m256d in_range = _mm256_and_pd(cmp_ge, cmp_le);
        
        int mask = _mm256_movemask_pd(in_range);
        count += __builtin_popcount(mask);
    }
    
    // Handle remainder
    for (; i < n; i++) {
        if (values[i] >= minVal && values[i] <= maxVal) {
            count++;
        }
    }
    
    *countOut = count;
}

// GatherU32AVX2 - Gather uint32 values at specified indices
// Useful for collecting rowIDs that matched a filter
void gatherU32Avx2(
    const uint32_t* __restrict__ src,
    const int32_t* __restrict__ indices,
    int64_t n,
    uint32_t* __restrict__ dst
) {
    int64_t i = 0;
    
    // Process 8 indices at a time using AVX2 gather
    for (; i + 8 <= n; i += 8) {
        __m256i idx = _mm256_loadu_si256((const __m256i*)&indices[i]);
        __m256i gathered = _mm256_i32gather_epi32((const int*)src, idx, 4);
        _mm256_storeu_si256((__m256i*)&dst[i], gathered);
    }
    
    // Handle remainder
    for (; i < n; i++) {
        dst[i] = src[indices[i]];
    }
}
