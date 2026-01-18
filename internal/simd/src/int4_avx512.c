// INT4 (4-bit) quantization SIMD kernels for AVX-512
//
// Optimizations:
//   - Inline assembly for constant construction (no rodata relocations)
//   - Dual accumulators for ILP  
//   - Software prefetching
//   - AVX-512 native reduce operations
//
// INT4 is nibble-packed: high nibble = first value, low nibble = second value
// Dequantization: val[i] = (quant[i] / 15.0) * diff[i] + minVal[i]

#include <immintrin.h>
#include <stdint.h>

#define PREFETCH_AHEAD 512

// Helper: load 1/15.0f into scalar XMM from immediate (no rodata)
static inline float load_inv15_scalar_avx512(void) {
    uint32_t bits = 0x3d888889;
    float result;
    __asm__ ("vmovd %1, %%xmm15; vmovss %%xmm15, %0" 
             : "=m"(result) : "r"(bits) : "xmm15");
    return result;
}

// int4L2DistanceAvx512 computes squared L2 distance between query and INT4 code.
void int4L2DistanceAvx512(const float *__restrict__ query,
                          const uint8_t *__restrict__ code,
                          int64_t dim,
                          const float *__restrict__ minVal,
                          const float *__restrict__ diff,
                          float *__restrict__ out) {
    // Dual accumulators for better ILP
    __m512 sum1 = _mm512_setzero_ps();
    __m512 sum2 = _mm512_setzero_ps();
    
    // Construct 1/15.0f (0x3d888889) using mov immediate + broadcast
    uint32_t inv15_bits = 0x3d888889;
    __m128 inv15_scalar;
    __asm__ ("vmovd %1, %0" : "=x"(inv15_scalar) : "r"(inv15_bits));
    __m512 scale = _mm512_broadcastss_ps(inv15_scalar);
    
    // Construct 0x0F mask using mov immediate + broadcast
    uint32_t mask_bits = 0x0F0F0F0F;
    __m128i mask_scalar;
    __asm__ ("vmovd %1, %0" : "=x"(mask_scalar) : "r"(mask_bits));
    __m128i nibble_mask_128 = _mm_broadcastd_epi32(mask_scalar);

    int64_t i = 0;
    
    // 2x unrolled: Process 64 dimensions at a time
    for (; i <= dim - 64; i += 64) {
        // Prefetch ahead
        _mm_prefetch((const char*)(code + i / 2 + PREFETCH_AHEAD/2), _MM_HINT_T0);
        _mm_prefetch((const char*)(query + i + PREFETCH_AHEAD/4), _MM_HINT_T0);
        
        // First 32 dimensions
        __m128i packed1 = _mm_loadu_si128((__m128i const *)(code + i / 2));
        __m128i high1 = _mm_and_si128(_mm_srli_epi16(packed1, 4), nibble_mask_128);
        __m128i low1 = _mm_and_si128(packed1, nibble_mask_128);
        __m512i vals1_0 = _mm512_cvtepu8_epi32(_mm_unpacklo_epi8(high1, low1));
        __m512i vals1_1 = _mm512_cvtepu8_epi32(_mm_unpackhi_epi8(high1, low1));
        __m512 f1_0 = _mm512_cvtepi32_ps(vals1_0);
        __m512 f1_1 = _mm512_cvtepi32_ps(vals1_1);
        
        // Second 32 dimensions
        __m128i packed2 = _mm_loadu_si128((__m128i const *)(code + i / 2 + 16));
        __m128i high2 = _mm_and_si128(_mm_srli_epi16(packed2, 4), nibble_mask_128);
        __m128i low2 = _mm_and_si128(packed2, nibble_mask_128);
        __m512i vals2_0 = _mm512_cvtepu8_epi32(_mm_unpacklo_epi8(high2, low2));
        __m512i vals2_1 = _mm512_cvtepu8_epi32(_mm_unpackhi_epi8(high2, low2));
        __m512 f2_0 = _mm512_cvtepi32_ps(vals2_0);
        __m512 f2_1 = _mm512_cvtepi32_ps(vals2_1);
        
        // Dequantize all
        __m512 dq1_0 = _mm512_fmadd_ps(_mm512_mul_ps(f1_0, scale), _mm512_loadu_ps(diff + i), _mm512_loadu_ps(minVal + i));
        __m512 dq1_1 = _mm512_fmadd_ps(_mm512_mul_ps(f1_1, scale), _mm512_loadu_ps(diff + i + 16), _mm512_loadu_ps(minVal + i + 16));
        __m512 dq2_0 = _mm512_fmadd_ps(_mm512_mul_ps(f2_0, scale), _mm512_loadu_ps(diff + i + 32), _mm512_loadu_ps(minVal + i + 32));
        __m512 dq2_1 = _mm512_fmadd_ps(_mm512_mul_ps(f2_1, scale), _mm512_loadu_ps(diff + i + 48), _mm512_loadu_ps(minVal + i + 48));
        
        // Compute and accumulate
        __m512 d1_0 = _mm512_sub_ps(_mm512_loadu_ps(query + i), dq1_0);
        __m512 d1_1 = _mm512_sub_ps(_mm512_loadu_ps(query + i + 16), dq1_1);
        __m512 d2_0 = _mm512_sub_ps(_mm512_loadu_ps(query + i + 32), dq2_0);
        __m512 d2_1 = _mm512_sub_ps(_mm512_loadu_ps(query + i + 48), dq2_1);
        
        sum1 = _mm512_fmadd_ps(d1_0, d1_0, sum1);
        sum1 = _mm512_fmadd_ps(d1_1, d1_1, sum1);
        sum2 = _mm512_fmadd_ps(d2_0, d2_0, sum2);
        sum2 = _mm512_fmadd_ps(d2_1, d2_1, sum2);
    }
    
    // Process remaining 32-element blocks
    for (; i <= dim - 32; i += 32) {
        __m128i packed = _mm_loadu_si128((__m128i const *)(code + i / 2));
        __m128i high_nibbles = _mm_and_si128(_mm_srli_epi16(packed, 4), nibble_mask_128);
        __m128i low_nibbles = _mm_and_si128(packed, nibble_mask_128);
        
        __m512i vals_0 = _mm512_cvtepu8_epi32(_mm_unpacklo_epi8(high_nibbles, low_nibbles));
        __m512i vals_1 = _mm512_cvtepu8_epi32(_mm_unpackhi_epi8(high_nibbles, low_nibbles));
        __m512 f_0 = _mm512_cvtepi32_ps(vals_0);
        __m512 f_1 = _mm512_cvtepi32_ps(vals_1);
        
        __m512 dequant_0 = _mm512_fmadd_ps(_mm512_mul_ps(f_0, scale), _mm512_loadu_ps(diff + i), _mm512_loadu_ps(minVal + i));
        __m512 dequant_1 = _mm512_fmadd_ps(_mm512_mul_ps(f_1, scale), _mm512_loadu_ps(diff + i + 16), _mm512_loadu_ps(minVal + i + 16));
        
        __m512 d_0 = _mm512_sub_ps(_mm512_loadu_ps(query + i), dequant_0);
        __m512 d_1 = _mm512_sub_ps(_mm512_loadu_ps(query + i + 16), dequant_1);
        
        sum1 = _mm512_fmadd_ps(d_0, d_0, sum1);
        sum1 = _mm512_fmadd_ps(d_1, d_1, sum1);
    }
    
    // Combine accumulators and reduce
    sum1 = _mm512_add_ps(sum1, sum2);
    float total = _mm512_reduce_add_ps(sum1);
    
    // Scalar tail
    float inv15_f = load_inv15_scalar_avx512();
    
    for (; i < dim; i += 2) {
        uint8_t packed_byte = code[i / 2];
        uint8_t q1 = (packed_byte >> 4) & 0x0F;
        uint8_t q2 = packed_byte & 0x0F;
        
        float val1 = ((float)q1 * inv15_f) * diff[i] + minVal[i];
        float d1 = query[i] - val1;
        total += d1 * d1;
        
        if (i + 1 < dim) {
            float val2 = ((float)q2 * inv15_f) * diff[i + 1] + minVal[i + 1];
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
// Fully inlined to avoid function call relocations.
void int4L2DistanceBatchAvx512(const float *__restrict__ query,
                               const uint8_t *__restrict__ codes,
                               int64_t dim,
                               int64_t n,
                               const float *__restrict__ minVal,
                               const float *__restrict__ diff,
                               float *__restrict__ out) {
    int64_t codeSize = (dim + 1) / 2;
    
    // Construct constants once
    uint32_t inv15_bits = 0x3d888889;
    __m128 inv15_scalar;
    __asm__ ("vmovd %1, %0" : "=x"(inv15_scalar) : "r"(inv15_bits));
    __m512 scale = _mm512_broadcastss_ps(inv15_scalar);
    
    uint32_t mask_bits = 0x0F0F0F0F;
    __m128i mask_scalar;
    __asm__ ("vmovd %1, %0" : "=x"(mask_scalar) : "r"(mask_bits));
    __m128i nibble_mask_128 = _mm_broadcastd_epi32(mask_scalar);
    
    float inv15_f;
    __asm__ ("vmovd %1, %%xmm15; vmovss %%xmm15, %0" : "=m"(inv15_f) : "r"(inv15_bits) : "xmm15");
    
    for (int64_t j = 0; j < n; j++) {
        const uint8_t *code = codes + j * codeSize;
        
        // Prefetch next code
        if (j + 1 < n) {
            _mm_prefetch((const char*)(codes + (j + 1) * codeSize), _MM_HINT_T0);
        }
        
        __m512 sum1 = _mm512_setzero_ps();
        __m512 sum2 = _mm512_setzero_ps();
        
        int64_t i = 0;
        
        // 2x unrolled: 64 dimensions per iteration
        for (; i <= dim - 64; i += 64) {
            __m128i packed1 = _mm_loadu_si128((__m128i const *)(code + i / 2));
            __m128i high1 = _mm_and_si128(_mm_srli_epi16(packed1, 4), nibble_mask_128);
            __m128i low1 = _mm_and_si128(packed1, nibble_mask_128);
            __m512 f1_0 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_unpacklo_epi8(high1, low1)));
            __m512 f1_1 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_unpackhi_epi8(high1, low1)));
            
            __m128i packed2 = _mm_loadu_si128((__m128i const *)(code + i / 2 + 16));
            __m128i high2 = _mm_and_si128(_mm_srli_epi16(packed2, 4), nibble_mask_128);
            __m128i low2 = _mm_and_si128(packed2, nibble_mask_128);
            __m512 f2_0 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_unpacklo_epi8(high2, low2)));
            __m512 f2_1 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_unpackhi_epi8(high2, low2)));
            
            __m512 dq1_0 = _mm512_fmadd_ps(_mm512_mul_ps(f1_0, scale), _mm512_loadu_ps(diff + i), _mm512_loadu_ps(minVal + i));
            __m512 dq1_1 = _mm512_fmadd_ps(_mm512_mul_ps(f1_1, scale), _mm512_loadu_ps(diff + i + 16), _mm512_loadu_ps(minVal + i + 16));
            __m512 dq2_0 = _mm512_fmadd_ps(_mm512_mul_ps(f2_0, scale), _mm512_loadu_ps(diff + i + 32), _mm512_loadu_ps(minVal + i + 32));
            __m512 dq2_1 = _mm512_fmadd_ps(_mm512_mul_ps(f2_1, scale), _mm512_loadu_ps(diff + i + 48), _mm512_loadu_ps(minVal + i + 48));
            
            __m512 d1_0 = _mm512_sub_ps(_mm512_loadu_ps(query + i), dq1_0);
            __m512 d1_1 = _mm512_sub_ps(_mm512_loadu_ps(query + i + 16), dq1_1);
            __m512 d2_0 = _mm512_sub_ps(_mm512_loadu_ps(query + i + 32), dq2_0);
            __m512 d2_1 = _mm512_sub_ps(_mm512_loadu_ps(query + i + 48), dq2_1);
            
            sum1 = _mm512_fmadd_ps(d1_0, d1_0, sum1);
            sum1 = _mm512_fmadd_ps(d1_1, d1_1, sum1);
            sum2 = _mm512_fmadd_ps(d2_0, d2_0, sum2);
            sum2 = _mm512_fmadd_ps(d2_1, d2_1, sum2);
        }
        
        // 32-element cleanup
        for (; i <= dim - 32; i += 32) {
            __m128i packed = _mm_loadu_si128((__m128i const *)(code + i / 2));
            __m128i high = _mm_and_si128(_mm_srli_epi16(packed, 4), nibble_mask_128);
            __m128i low = _mm_and_si128(packed, nibble_mask_128);
            __m512 f_0 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_unpacklo_epi8(high, low)));
            __m512 f_1 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_unpackhi_epi8(high, low)));
            
            __m512 dq_0 = _mm512_fmadd_ps(_mm512_mul_ps(f_0, scale), _mm512_loadu_ps(diff + i), _mm512_loadu_ps(minVal + i));
            __m512 dq_1 = _mm512_fmadd_ps(_mm512_mul_ps(f_1, scale), _mm512_loadu_ps(diff + i + 16), _mm512_loadu_ps(minVal + i + 16));
            
            __m512 d_0 = _mm512_sub_ps(_mm512_loadu_ps(query + i), dq_0);
            __m512 d_1 = _mm512_sub_ps(_mm512_loadu_ps(query + i + 16), dq_1);
            
            sum1 = _mm512_fmadd_ps(d_0, d_0, sum1);
            sum1 = _mm512_fmadd_ps(d_1, d_1, sum1);
        }
        
        sum1 = _mm512_add_ps(sum1, sum2);
        float total = _mm512_reduce_add_ps(sum1);
        
        // Scalar tail
        for (; i < dim; i += 2) {
            uint8_t packed_byte = code[i / 2];
            uint8_t q1 = (packed_byte >> 4) & 0x0F;
            uint8_t q2 = packed_byte & 0x0F;
            
            float val1 = ((float)q1 * inv15_f) * diff[i] + minVal[i];
            float d1 = query[i] - val1;
            total += d1 * d1;
            
            if (i + 1 < dim) {
                float val2 = ((float)q2 * inv15_f) * diff[i + 1] + minVal[i + 1];
                float d2 = query[i + 1] - val2;
                total += d2 * d2;
            }
        }
        
        out[j] = total;
    }
}
