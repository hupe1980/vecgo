#include <immintrin.h>
#include <stdint.h>

// INT4 (4-bit) quantization SIMD kernels for AVX2
//
// INT4 is nibble-packed: high nibble = first value, low nibble = second value
// Dequantization: val[i] = (quant[i] / 15.0) * diff[i] + minVal[i]
//
// For precomputed lookup tables:
// lookupTable[i*16 + q] = (q / 15.0) * diff[i] + minVal[i]

// int4L2DistanceAvx2 computes squared L2 distance between query and INT4 code.
// Uses per-dimension min/diff for dequantization.
//
// query: float32 query vector (dim)
// code: packed INT4 codes ((dim+1)/2 bytes)
// dim: dimension of vectors
// min: per-dimension minimum values
// diff: per-dimension scale (max - min)
// out: output distance
void int4L2DistanceAvx2(const float *__restrict__ query,
                       const uint8_t *__restrict__ code,
                       int64_t dim,
                       const float *__restrict__ minVal,
                       const float *__restrict__ diff,
                       float *__restrict__ out) {
    __m256 sum = _mm256_setzero_ps();
    
    // Avoid literal pool: construct 1/15 from integer bit pattern
    // 1.0f/15.0f = 0x3c888889 in IEEE 754
    __m256i scale_bits = _mm256_set1_epi32(0x3c888889);
    __m256 scale = _mm256_castsi256_ps(scale_bits);
    
    // Construct 0x0F0F0F0F... mask from smaller values
    __m128i nibble_mask_128 = _mm_set1_epi32(0x0F0F0F0F);

    int64_t i = 0;
    // Process 16 dimensions at a time (8 bytes of packed INT4)
    for (; i <= dim - 16; i += 16) {
        // Load 8 bytes of packed INT4 data (16 values)
        __m128i packed = _mm_loadl_epi64((__m128i const *)(code + i / 2));
        
        // Unpack high nibbles (first values): shift right by 4, then mask
        __m128i high_nibbles = _mm_and_si128(_mm_srli_epi16(packed, 4), nibble_mask_128);
        
        // Unpack low nibbles (second values)
        __m128i low_nibbles = _mm_and_si128(packed, nibble_mask_128);
        
        // Interleave: we need [h0,l0,h1,l1,h2,l2,...] pattern
        // First 8 values: h0,l0,h1,l1,h2,l2,h3,l3
        __m128i interleaved_lo = _mm_unpacklo_epi8(high_nibbles, low_nibbles);
        // Next 8 values: h4,l4,h5,l5,h6,l6,h7,l7
        __m128i interleaved_hi = _mm_unpackhi_epi8(high_nibbles, low_nibbles);
        
        // Expand to 32-bit integers (first 8)
        __m256i vals_lo = _mm256_cvtepu8_epi32(interleaved_lo);
        // Expand to 32-bit integers (next 8)
        __m256i vals_hi = _mm256_cvtepu8_epi32(_mm_srli_si128(interleaved_lo, 8));
        
        // Convert to float
        __m256 f_lo = _mm256_cvtepi32_ps(vals_lo);
        __m256 f_hi = _mm256_cvtepi32_ps(vals_hi);
        
        // Dequantize: val = (quant / 15.0) * diff + min
        __m256 diff_lo = _mm256_loadu_ps(diff + i);
        __m256 diff_hi = _mm256_loadu_ps(diff + i + 8);
        __m256 minVal_lo = _mm256_loadu_ps(minVal + i);
        __m256 minVal_hi = _mm256_loadu_ps(minVal + i + 8);
        
        __m256 dequant_lo = _mm256_fmadd_ps(_mm256_mul_ps(f_lo, scale), diff_lo, minVal_lo);
        __m256 dequant_hi = _mm256_fmadd_ps(_mm256_mul_ps(f_hi, scale), diff_hi, minVal_hi);
        
        // Load query values
        __m256 q_lo = _mm256_loadu_ps(query + i);
        __m256 q_hi = _mm256_loadu_ps(query + i + 8);
        
        // Compute differences
        __m256 d_lo = _mm256_sub_ps(q_lo, dequant_lo);
        __m256 d_hi = _mm256_sub_ps(q_hi, dequant_hi);
        
        // Accumulate squared differences
        sum = _mm256_fmadd_ps(d_lo, d_lo, sum);
        sum = _mm256_fmadd_ps(d_hi, d_hi, sum);
    }
    
    // Horizontal sum
    __m128 sum_lo = _mm256_castps256_ps128(sum);
    __m128 sum_hi = _mm256_extractf128_ps(sum, 1);
    __m128 sum_128 = _mm_add_ps(sum_lo, sum_hi);
    sum_128 = _mm_hadd_ps(sum_128, sum_128);
    sum_128 = _mm_hadd_ps(sum_128, sum_128);
    float total = _mm_cvtss_f32(sum_128);
    
    // Scalar tail
    for (; i < dim; i += 2) {
        uint8_t packed_byte = code[i / 2];
        uint8_t q1 = (packed_byte >> 4) & 0x0F;
        uint8_t q2 = packed_byte & 0x0F;
        
        float val1 = ((float)q1 / 15.0f) * diff[i] + minVal[i];
        float d1 = query[i] - val1;
        total += d1 * d1;
        
        if (i + 1 < dim) {
            float val2 = ((float)q2 / 15.0f) * diff[i + 1] + minVal[i + 1];
            float d2 = query[i + 1] - val2;
            total += d2 * d2;
        }
    }
    
    *out = total;
}

// int4L2DistancePrecomputedAvx2 uses precomputed lookup tables for faster distance.
// lookupTable is 16 * dim floats: lookupTable[i*16 + q] = dequantized value for dim i, quant q
void int4L2DistancePrecomputedAvx2(const float *__restrict__ query,
                                   const uint8_t *__restrict__ code,
                                   int64_t dim,
                                   const float *__restrict__ lookupTable,
                                   float *__restrict__ out) {
    __m256 sum = _mm256_setzero_ps();
    
    int64_t i = 0;
    // Process 8 dimensions at a time
    for (; i <= dim - 8; i += 8) {
        // Load 4 bytes (8 nibbles)
        uint32_t packed = *(uint32_t *)(code + i / 2);
        
        // Manual lookup for 8 values (AVX gather is expensive for small tables)
        float vals[8];
        for (int k = 0; k < 8; k += 2) {
            uint8_t byte = (packed >> (k * 4)) & 0xFF;
            uint8_t q1 = (byte >> 4) & 0x0F;
            uint8_t q2 = byte & 0x0F;
            vals[k] = lookupTable[(i + k) * 16 + q1];
            vals[k + 1] = lookupTable[(i + k + 1) * 16 + q2];
        }
        
        __m256 dequant = _mm256_loadu_ps(vals);
        __m256 q = _mm256_loadu_ps(query + i);
        __m256 d = _mm256_sub_ps(q, dequant);
        sum = _mm256_fmadd_ps(d, d, sum);
    }
    
    // Horizontal sum
    __m128 sum_lo = _mm256_castps256_ps128(sum);
    __m128 sum_hi = _mm256_extractf128_ps(sum, 1);
    __m128 sum_128 = _mm_add_ps(sum_lo, sum_hi);
    sum_128 = _mm_hadd_ps(sum_128, sum_128);
    sum_128 = _mm_hadd_ps(sum_128, sum_128);
    float total = _mm_cvtss_f32(sum_128);
    
    // Scalar tail
    for (; i < dim; i += 2) {
        uint8_t packed_byte = code[i / 2];
        uint8_t q1 = (packed_byte >> 4) & 0x0F;
        float val1 = lookupTable[i * 16 + q1];
        float d1 = query[i] - val1;
        total += d1 * d1;
        
        if (i + 1 < dim) {
            uint8_t q2 = packed_byte & 0x0F;
            float val2 = lookupTable[(i + 1) * 16 + q2];
            float d2 = query[i + 1] - val2;
            total += d2 * d2;
        }
    }
    
    *out = total;
}

// int4L2DistanceBatchAvx2 computes L2 distances for multiple INT4 codes.
// codes is a flattened array of n codes, each of size (dim+1)/2.
void int4L2DistanceBatchAvx2(const float *__restrict__ query,
                            const uint8_t *__restrict__ codes,
                            int64_t dim,
                            int64_t n,
                            const float *__restrict__ minVal,
                            const float *__restrict__ diff,
                            float *__restrict__ out) {
    int64_t codeSize = (dim + 1) / 2;
    
    for (int64_t j = 0; j < n; j++) {
        int4L2DistanceAvx2(query, codes + j * codeSize, dim, minVal, diff, out + j);
    }
}
