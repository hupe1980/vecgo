// SIMD filter operations using AVX-512 for numeric indexing
// Compile with: clang -O3 -mavx512f -mavx512bw -mavx512dq -S -o filter_avx512.s filter_avx512.c

#include <immintrin.h>
#include <stdint.h>

// FilterRangeF64AVX512 - Check if values are in range [minVal, maxVal]
// Processes 8 float64 values at a time using AVX-512
//
// Arguments:
//   values: pointer to float64 array
//   n: number of elements
//   minVal: minimum value (inclusive)
//   maxVal: maximum value (inclusive)
//   dst: output byte array (1 = in range, 0 = out of range)
void filterRangeF64Avx512(
    const double* __restrict__ values,
    int64_t n,
    double minVal,
    double maxVal,
    uint8_t* __restrict__ dst
) {
    // Broadcast min and max to all 8 lanes
    __m512d vmin = _mm512_set1_pd(minVal);
    __m512d vmax = _mm512_set1_pd(maxVal);
    
    int64_t i = 0;
    
    // Process 8 doubles at a time (512 bits / 64 bits = 8)
    for (; i + 8 <= n; i += 8) {
        __m512d v = _mm512_loadu_pd(&values[i]);
        
        // AVX-512 comparison returns a mask register directly
        __mmask8 cmp_ge = _mm512_cmp_pd_mask(v, vmin, _CMP_GE_OQ);  // v >= min
        __mmask8 cmp_le = _mm512_cmp_pd_mask(v, vmax, _CMP_LE_OQ);  // v <= max
        __mmask8 in_range = cmp_ge & cmp_le;
        
        // Extract individual bits to bytes
        dst[i]     = (in_range >> 0) & 1;
        dst[i + 1] = (in_range >> 1) & 1;
        dst[i + 2] = (in_range >> 2) & 1;
        dst[i + 3] = (in_range >> 3) & 1;
        dst[i + 4] = (in_range >> 4) & 1;
        dst[i + 5] = (in_range >> 5) & 1;
        dst[i + 6] = (in_range >> 6) & 1;
        dst[i + 7] = (in_range >> 7) & 1;
    }
    
    // Handle remainder
    for (; i < n; i++) {
        dst[i] = (values[i] >= minVal && values[i] <= maxVal) ? 1 : 0;
    }
}

// FilterRangeF64IndicesAVX512 - Return indices of values in range
// Uses AVX-512 compress instruction for efficient index extraction
// Writes count of matching indices to *countOut
void filterRangeF64IndicesAvx512(
    const double* __restrict__ values,
    int64_t n,
    double minVal,
    double maxVal,
    int32_t* __restrict__ dst,
    int64_t* __restrict__ countOut
) {
    __m512d vmin = _mm512_set1_pd(minVal);
    __m512d vmax = _mm512_set1_pd(maxVal);
    
    int64_t count = 0;
    int64_t i = 0;
    
    for (; i + 8 <= n; i += 8) {
        __m512d v = _mm512_loadu_pd(&values[i]);
        
        __mmask8 cmp_ge = _mm512_cmp_pd_mask(v, vmin, _CMP_GE_OQ);
        __mmask8 cmp_le = _mm512_cmp_pd_mask(v, vmax, _CMP_LE_OQ);
        __mmask8 in_range = cmp_ge & cmp_le;
        
        // Create index vector [i, i+1, i+2, i+3, i+4, i+5, i+6, i+7]
        __m256i indices = _mm256_setr_epi32(
            (int32_t)i, (int32_t)(i+1), (int32_t)(i+2), (int32_t)(i+3),
            (int32_t)(i+4), (int32_t)(i+5), (int32_t)(i+6), (int32_t)(i+7)
        );
        
        // Compress store - only store matching indices
        _mm256_mask_compressstoreu_epi32(&dst[count], in_range, indices);
        count += __builtin_popcount(in_range);
    }
    
    // Handle remainder
    for (; i < n; i++) {
        if (values[i] >= minVal && values[i] <= maxVal) {
            dst[count++] = (int32_t)i;
        }
    }
    
    *countOut = count;
}

// CountRangeF64AVX512 - Count values in range [minVal, maxVal]
// Writes count to *countOut
void countRangeF64Avx512(
    const double* __restrict__ values,
    int64_t n,
    double minVal,
    double maxVal,
    int64_t* __restrict__ countOut
) {
    __m512d vmin = _mm512_set1_pd(minVal);
    __m512d vmax = _mm512_set1_pd(maxVal);
    
    int64_t count = 0;
    int64_t i = 0;
    
    for (; i + 8 <= n; i += 8) {
        __m512d v = _mm512_loadu_pd(&values[i]);
        
        __mmask8 cmp_ge = _mm512_cmp_pd_mask(v, vmin, _CMP_GE_OQ);
        __mmask8 cmp_le = _mm512_cmp_pd_mask(v, vmax, _CMP_LE_OQ);
        __mmask8 in_range = cmp_ge & cmp_le;
        
        count += __builtin_popcount(in_range);
    }
    
    // Handle remainder
    for (; i < n; i++) {
        if (values[i] >= minVal && values[i] <= maxVal) {
            count++;
        }
    }
    
    *countOut = count;
}

// GatherU32AVX512 - Gather uint32 values at specified indices
// Uses AVX-512 for 16-wide gather
void gatherU32Avx512(
    const uint32_t* __restrict__ src,
    const int32_t* __restrict__ indices,
    int64_t n,
    uint32_t* __restrict__ dst
) {
    int64_t i = 0;
    
    // Process 16 indices at a time using AVX-512 gather
    for (; i + 16 <= n; i += 16) {
        __m512i idx = _mm512_loadu_si512((const __m512i*)&indices[i]);
        __m512i gathered = _mm512_i32gather_epi32(idx, src, 4);
        _mm512_storeu_si512((__m512i*)&dst[i], gathered);
    }
    
    // Handle remainder with scalar
    for (; i < n; i++) {
        dst[i] = src[indices[i]];
    }
}
