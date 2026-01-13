#include <immintrin.h>
#include <stdint.h>

// INT4 (4-bit) quantization SIMD kernels for AVX-512
//
// INT4 is nibble-packed: high nibble = first value, low nibble = second value
// Dequantization: val[i] = (quant[i] / 15.0) * diff[i] + min[i]

// int4L2DistanceAvx512 computes squared L2 distance between query and INT4 code.
void int4L2DistanceAvx512(const float *__restrict__ query,
                          const uint8_t *__restrict__ code,
                          int64_t dim,
                          const float *__restrict__ min,
                          const float *__restrict__ diff,
                          float *__restrict__ out) {
    __m512 sum = _mm512_setzero_ps();
    const __m512 scale = _mm512_set1_ps(1.0f / 15.0f);

    int64_t i = 0;
    // Process 32 dimensions at a time (16 bytes of packed INT4)
    for (; i <= dim - 32; i += 32) {
        // Load 16 bytes of packed INT4 data (32 values)
        __m128i packed = _mm_loadu_si128((__m128i const *)(code + i / 2));
        
        // Unpack high nibbles (shift right by 4, mask)
        __m128i high_nibbles = _mm_and_si128(_mm_srli_epi16(packed, 4), _mm_set1_epi8(0x0F));
        // Unpack low nibbles
        __m128i low_nibbles = _mm_and_si128(packed, _mm_set1_epi8(0x0F));
        
        // Interleave to get correct order: [h0,l0,h1,l1,...]
        __m128i interleaved_lo = _mm_unpacklo_epi8(high_nibbles, low_nibbles);
        __m128i interleaved_hi = _mm_unpackhi_epi8(high_nibbles, low_nibbles);
        
        // Expand first 16 values to 32-bit
        __m512i vals_0 = _mm512_cvtepu8_epi32(interleaved_lo);
        // Expand next 16 values
        __m512i vals_1 = _mm512_cvtepu8_epi32(interleaved_hi);
        
        // Convert to float
        __m512 f_0 = _mm512_cvtepi32_ps(vals_0);
        __m512 f_1 = _mm512_cvtepi32_ps(vals_1);
        
        // Dequantize: val = (quant / 15.0) * diff + min
        __m512 diff_0 = _mm512_loadu_ps(diff + i);
        __m512 diff_1 = _mm512_loadu_ps(diff + i + 16);
        __m512 min_0 = _mm512_loadu_ps(min + i);
        __m512 min_1 = _mm512_loadu_ps(min + i + 16);
        
        __m512 dequant_0 = _mm512_fmadd_ps(_mm512_mul_ps(f_0, scale), diff_0, min_0);
        __m512 dequant_1 = _mm512_fmadd_ps(_mm512_mul_ps(f_1, scale), diff_1, min_1);
        
        // Load query values
        __m512 q_0 = _mm512_loadu_ps(query + i);
        __m512 q_1 = _mm512_loadu_ps(query + i + 16);
        
        // Compute differences
        __m512 d_0 = _mm512_sub_ps(q_0, dequant_0);
        __m512 d_1 = _mm512_sub_ps(q_1, dequant_1);
        
        // Accumulate squared differences
        sum = _mm512_fmadd_ps(d_0, d_0, sum);
        sum = _mm512_fmadd_ps(d_1, d_1, sum);
    }
    
    // Reduce 512-bit sum to scalar
    float total = _mm512_reduce_add_ps(sum);
    
    // Scalar tail
    for (; i < dim; i += 2) {
        uint8_t packed_byte = code[i / 2];
        uint8_t q1 = (packed_byte >> 4) & 0x0F;
        uint8_t q2 = packed_byte & 0x0F;
        
        float val1 = ((float)q1 / 15.0f) * diff[i] + min[i];
        float d1 = query[i] - val1;
        total += d1 * d1;
        
        if (i + 1 < dim) {
            float val2 = ((float)q2 / 15.0f) * diff[i + 1] + min[i + 1];
            float d2 = query[i + 1] - val2;
            total += d2 * d2;
        }
    }
    
    *out = total;
}

// int4L2DistancePrecomputedAvx512 uses precomputed lookup tables.
void int4L2DistancePrecomputedAvx512(const float *__restrict__ query,
                                      const uint8_t *__restrict__ code,
                                      int64_t dim,
                                      const float *__restrict__ lookupTable,
                                      float *__restrict__ out) {
    __m512 sum = _mm512_setzero_ps();
    
    int64_t i = 0;
    // Process 16 dimensions at a time
    for (; i <= dim - 16; i += 16) {
        // Load 8 bytes (16 nibbles)
        uint64_t packed = *(uint64_t *)(code + i / 2);
        
        // Manual lookup for 16 values
        float vals[16];
        for (int k = 0; k < 16; k += 2) {
            uint8_t byte = (packed >> (k * 4)) & 0xFF;
            uint8_t q1 = (byte >> 4) & 0x0F;
            uint8_t q2 = byte & 0x0F;
            vals[k] = lookupTable[(i + k) * 16 + q1];
            vals[k + 1] = lookupTable[(i + k + 1) * 16 + q2];
        }
        
        __m512 dequant = _mm512_loadu_ps(vals);
        __m512 q = _mm512_loadu_ps(query + i);
        __m512 d = _mm512_sub_ps(q, dequant);
        sum = _mm512_fmadd_ps(d, d, sum);
    }
    
    float total = _mm512_reduce_add_ps(sum);
    
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

// int4L2DistanceBatchAvx512 computes L2 distances for multiple INT4 codes.
void int4L2DistanceBatchAvx512(const float *__restrict__ query,
                               const uint8_t *__restrict__ codes,
                               int64_t dim,
                               int64_t n,
                               const float *__restrict__ min,
                               const float *__restrict__ diff,
                               float *__restrict__ out) {
    int64_t codeSize = (dim + 1) / 2;
    
    for (int64_t j = 0; j < n; j++) {
        int4L2DistanceAvx512(query, codes + j * codeSize, dim, min, diff, out + j);
    }
}
