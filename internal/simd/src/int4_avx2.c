// INT4 (4-bit) quantization SIMD kernels for AVX2
//
// Optimizations:
//   - Inline assembly for constant construction (no rodata relocations)
//   - 2-way loop unrolling with dual accumulators for ILP
//   - Software prefetching for cache efficiency
//   - Optimized horizontal sum (register-only)
//   - FMA for fused multiply-add

#include <immintrin.h>
#include <stdint.h>

// Prefetch distance in bytes
#define PREFETCH_AHEAD 256

// Optimized horizontal sum: avoids store/load round-trip
static inline float hsum256_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

// Helper: load 1/15.0f into scalar XMM from immediate (no rodata)
static inline float load_inv15_scalar(void) {
    uint32_t bits = 0x3d888889;
    float result;
    __asm__ ("vmovd %1, %%xmm15; vmovss %%xmm15, %0" 
             : "=m"(result) : "r"(bits) : "xmm15");
    return result;
}

// int4L2DistanceAvx2 computes squared L2 distance between query and INT4 code.
void int4L2DistanceAvx2(const float *__restrict__ query,
                       const uint8_t *__restrict__ code,
                       int64_t dim,
                       const float *__restrict__ minVal,
                       const float *__restrict__ diff,
                       float *__restrict__ out) {
    // Dual accumulators for better ILP
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    
    // Construct 1/15.0f (0x3d888889) using mov immediate + broadcast
    uint32_t inv15_bits = 0x3d888889;
    __m128 inv15_scalar;
    __asm__ ("vmovd %1, %0" : "=x"(inv15_scalar) : "r"(inv15_bits));
    __m256 inv15 = _mm256_broadcastss_ps(inv15_scalar);
    
    // Construct 0x0F0F0F0F mask using mov immediate + broadcast
    uint32_t mask_bits = 0x0F0F0F0F;
    __m128i mask_scalar;
    __asm__ ("vmovd %1, %0" : "=x"(mask_scalar) : "r"(mask_bits));
    __m128i nibble_mask = _mm_broadcastd_epi32(mask_scalar);

    int64_t i = 0;
    
    // 2x unrolled: Process 32 dimensions at a time (16 bytes of packed INT4)
    for (; i + 32 <= dim; i += 32) {
        // Prefetch ahead
        _mm_prefetch((const char*)(code + i / 2 + PREFETCH_AHEAD/2), _MM_HINT_T0);
        _mm_prefetch((const char*)(query + i + PREFETCH_AHEAD/4), _MM_HINT_T0);
        
        // First 16 dimensions
        __m128i packed1 = _mm_loadl_epi64((__m128i const *)(code + i / 2));
        __m128i high1 = _mm_srli_epi16(packed1, 4);
        __m128i interleaved1 = _mm_unpacklo_epi8(high1, packed1);
        interleaved1 = _mm_and_si128(interleaved1, nibble_mask);
        
        __m128i lo1_128 = interleaved1;
        __m128i hi1_128 = _mm_srli_si128(interleaved1, 8);
        __m256i vals1_lo = _mm256_cvtepu8_epi32(lo1_128);
        __m256i vals1_hi = _mm256_cvtepu8_epi32(hi1_128);
        __m256 f1_lo = _mm256_cvtepi32_ps(vals1_lo);
        __m256 f1_hi = _mm256_cvtepi32_ps(vals1_hi);
        
        // Second 16 dimensions
        __m128i packed2 = _mm_loadl_epi64((__m128i const *)(code + i / 2 + 8));
        __m128i high2 = _mm_srli_epi16(packed2, 4);
        __m128i interleaved2 = _mm_unpacklo_epi8(high2, packed2);
        interleaved2 = _mm_and_si128(interleaved2, nibble_mask);
        
        __m128i lo2_128 = interleaved2;
        __m128i hi2_128 = _mm_srli_si128(interleaved2, 8);
        __m256i vals2_lo = _mm256_cvtepu8_epi32(lo2_128);
        __m256i vals2_hi = _mm256_cvtepu8_epi32(hi2_128);
        __m256 f2_lo = _mm256_cvtepi32_ps(vals2_lo);
        __m256 f2_hi = _mm256_cvtepi32_ps(vals2_hi);
        
        // Load all diff and minVal at once
        __m256 diff1_lo = _mm256_loadu_ps(diff + i);
        __m256 diff1_hi = _mm256_loadu_ps(diff + i + 8);
        __m256 diff2_lo = _mm256_loadu_ps(diff + i + 16);
        __m256 diff2_hi = _mm256_loadu_ps(diff + i + 24);
        __m256 min1_lo = _mm256_loadu_ps(minVal + i);
        __m256 min1_hi = _mm256_loadu_ps(minVal + i + 8);
        __m256 min2_lo = _mm256_loadu_ps(minVal + i + 16);
        __m256 min2_hi = _mm256_loadu_ps(minVal + i + 24);
        
        // Dequantize all
        __m256 dequant1_lo = _mm256_fmadd_ps(_mm256_mul_ps(f1_lo, inv15), diff1_lo, min1_lo);
        __m256 dequant1_hi = _mm256_fmadd_ps(_mm256_mul_ps(f1_hi, inv15), diff1_hi, min1_hi);
        __m256 dequant2_lo = _mm256_fmadd_ps(_mm256_mul_ps(f2_lo, inv15), diff2_lo, min2_lo);
        __m256 dequant2_hi = _mm256_fmadd_ps(_mm256_mul_ps(f2_hi, inv15), diff2_hi, min2_hi);
        
        // Load query and compute differences
        __m256 q1_lo = _mm256_loadu_ps(query + i);
        __m256 q1_hi = _mm256_loadu_ps(query + i + 8);
        __m256 q2_lo = _mm256_loadu_ps(query + i + 16);
        __m256 q2_hi = _mm256_loadu_ps(query + i + 24);
        
        __m256 d1_lo = _mm256_sub_ps(q1_lo, dequant1_lo);
        __m256 d1_hi = _mm256_sub_ps(q1_hi, dequant1_hi);
        __m256 d2_lo = _mm256_sub_ps(q2_lo, dequant2_lo);
        __m256 d2_hi = _mm256_sub_ps(q2_hi, dequant2_hi);
        
        // Accumulate to separate accumulators
        sum1 = _mm256_fmadd_ps(d1_lo, d1_lo, sum1);
        sum1 = _mm256_fmadd_ps(d1_hi, d1_hi, sum1);
        sum2 = _mm256_fmadd_ps(d2_lo, d2_lo, sum2);
        sum2 = _mm256_fmadd_ps(d2_hi, d2_hi, sum2);
    }
    
    // Handle remaining 16-element blocks
    for (; i + 16 <= dim; i += 16) {
        __m128i packed = _mm_loadl_epi64((__m128i const *)(code + i / 2));
        __m128i high_nibbles = _mm_srli_epi16(packed, 4);
        __m128i interleaved = _mm_unpacklo_epi8(high_nibbles, packed);
        interleaved = _mm_and_si128(interleaved, nibble_mask);
        
        __m128i lo_128 = interleaved;
        __m128i hi_128 = _mm_srli_si128(interleaved, 8);
        __m256i vals_lo = _mm256_cvtepu8_epi32(lo_128);
        __m256i vals_hi = _mm256_cvtepu8_epi32(hi_128);
        __m256 f_lo = _mm256_cvtepi32_ps(vals_lo);
        __m256 f_hi = _mm256_cvtepi32_ps(vals_hi);
        
        __m256 diff_lo = _mm256_loadu_ps(diff + i);
        __m256 diff_hi = _mm256_loadu_ps(diff + i + 8);
        __m256 min_lo = _mm256_loadu_ps(minVal + i);
        __m256 min_hi = _mm256_loadu_ps(minVal + i + 8);
        
        __m256 dequant_lo = _mm256_fmadd_ps(_mm256_mul_ps(f_lo, inv15), diff_lo, min_lo);
        __m256 dequant_hi = _mm256_fmadd_ps(_mm256_mul_ps(f_hi, inv15), diff_hi, min_hi);
        
        __m256 q_lo = _mm256_loadu_ps(query + i);
        __m256 q_hi = _mm256_loadu_ps(query + i + 8);
        __m256 d_lo = _mm256_sub_ps(q_lo, dequant_lo);
        __m256 d_hi = _mm256_sub_ps(q_hi, dequant_hi);
        
        sum1 = _mm256_fmadd_ps(d_lo, d_lo, sum1);
        sum1 = _mm256_fmadd_ps(d_hi, d_hi, sum1);
    }
    
    // Combine accumulators and horizontal sum
    sum1 = _mm256_add_ps(sum1, sum2);
    float total = hsum256_ps(sum1);
    
    // Scalar tail - get 1/15 from inline asm to avoid rodata
    float inv15_f = load_inv15_scalar();
    
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

// int4L2DistancePrecomputedAvx2 uses precomputed lookup tables for faster distance.
void int4L2DistancePrecomputedAvx2(const float *__restrict__ query,
                                   const uint8_t *__restrict__ code,
                                   int64_t dim,
                                   const float *__restrict__ lookupTable,
                                   float *__restrict__ out) {
    __m256 sum = _mm256_setzero_ps();
    
    int64_t i = 0;
    // Process 8 dimensions at a time
    for (; i + 8 <= dim; i += 8) {
        // Load 4 bytes (8 nibbles)
        uint32_t packed = *(uint32_t *)(code + i / 2);
        
        // Manual lookup for 8 values
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
// Note: Fully inlined to avoid function call relocations in generated assembly.
void int4L2DistanceBatchAvx2(const float *__restrict__ query,
                            const uint8_t *__restrict__ codes,
                            int64_t dim,
                            int64_t n,
                            const float *__restrict__ minVal,
                            const float *__restrict__ diff,
                            float *__restrict__ out) {
    int64_t codeSize = (dim + 1) / 2;
    
    // Construct constants once outside loop
    uint32_t inv15_bits = 0x3d888889;
    __m128 inv15_scalar;
    __asm__ ("vmovd %1, %0" : "=x"(inv15_scalar) : "r"(inv15_bits));
    __m256 inv15 = _mm256_broadcastss_ps(inv15_scalar);
    
    uint32_t mask_bits = 0x0F0F0F0F;
    __m128i mask_scalar;
    __asm__ ("vmovd %1, %0" : "=x"(mask_scalar) : "r"(mask_bits));
    __m128i nibble_mask = _mm_broadcastd_epi32(mask_scalar);
    
    float inv15_f;
    __asm__ ("vmovd %1, %%xmm15; vmovss %%xmm15, %0" : "=m"(inv15_f) : "r"(inv15_bits) : "xmm15");
    
    for (int64_t j = 0; j < n; j++) {
        const uint8_t *code = codes + j * codeSize;
        
        // Prefetch next code
        if (j + 1 < n) {
            _mm_prefetch((const char*)(codes + (j + 1) * codeSize), _MM_HINT_T0);
        }
        
        __m256 sum1 = _mm256_setzero_ps();
        __m256 sum2 = _mm256_setzero_ps();

        int64_t i = 0;
        
        // 2x unrolled: 32 dimensions per iteration
        for (; i + 32 <= dim; i += 32) {
            __m128i packed1 = _mm_loadl_epi64((__m128i const *)(code + i / 2));
            __m128i high1 = _mm_srli_epi16(packed1, 4);
            __m128i interleaved1 = _mm_and_si128(_mm_unpacklo_epi8(high1, packed1), nibble_mask);
            __m256 f1_lo = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(interleaved1));
            __m256 f1_hi = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(interleaved1, 8)));
            
            __m128i packed2 = _mm_loadl_epi64((__m128i const *)(code + i / 2 + 8));
            __m128i high2 = _mm_srli_epi16(packed2, 4);
            __m128i interleaved2 = _mm_and_si128(_mm_unpacklo_epi8(high2, packed2), nibble_mask);
            __m256 f2_lo = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(interleaved2));
            __m256 f2_hi = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(interleaved2, 8)));
            
            __m256 dq1_lo = _mm256_fmadd_ps(_mm256_mul_ps(f1_lo, inv15), _mm256_loadu_ps(diff + i), _mm256_loadu_ps(minVal + i));
            __m256 dq1_hi = _mm256_fmadd_ps(_mm256_mul_ps(f1_hi, inv15), _mm256_loadu_ps(diff + i + 8), _mm256_loadu_ps(minVal + i + 8));
            __m256 dq2_lo = _mm256_fmadd_ps(_mm256_mul_ps(f2_lo, inv15), _mm256_loadu_ps(diff + i + 16), _mm256_loadu_ps(minVal + i + 16));
            __m256 dq2_hi = _mm256_fmadd_ps(_mm256_mul_ps(f2_hi, inv15), _mm256_loadu_ps(diff + i + 24), _mm256_loadu_ps(minVal + i + 24));
            
            __m256 d1_lo = _mm256_sub_ps(_mm256_loadu_ps(query + i), dq1_lo);
            __m256 d1_hi = _mm256_sub_ps(_mm256_loadu_ps(query + i + 8), dq1_hi);
            __m256 d2_lo = _mm256_sub_ps(_mm256_loadu_ps(query + i + 16), dq2_lo);
            __m256 d2_hi = _mm256_sub_ps(_mm256_loadu_ps(query + i + 24), dq2_hi);
            
            sum1 = _mm256_fmadd_ps(d1_lo, d1_lo, sum1);
            sum1 = _mm256_fmadd_ps(d1_hi, d1_hi, sum1);
            sum2 = _mm256_fmadd_ps(d2_lo, d2_lo, sum2);
            sum2 = _mm256_fmadd_ps(d2_hi, d2_hi, sum2);
        }
        
        // 16-element cleanup
        for (; i + 16 <= dim; i += 16) {
            __m128i packed = _mm_loadl_epi64((__m128i const *)(code + i / 2));
            __m128i interleaved = _mm_and_si128(_mm_unpacklo_epi8(_mm_srli_epi16(packed, 4), packed), nibble_mask);
            __m256 f_lo = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(interleaved));
            __m256 f_hi = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(interleaved, 8)));
            
            __m256 dq_lo = _mm256_fmadd_ps(_mm256_mul_ps(f_lo, inv15), _mm256_loadu_ps(diff + i), _mm256_loadu_ps(minVal + i));
            __m256 dq_hi = _mm256_fmadd_ps(_mm256_mul_ps(f_hi, inv15), _mm256_loadu_ps(diff + i + 8), _mm256_loadu_ps(minVal + i + 8));
            
            __m256 d_lo = _mm256_sub_ps(_mm256_loadu_ps(query + i), dq_lo);
            __m256 d_hi = _mm256_sub_ps(_mm256_loadu_ps(query + i + 8), dq_hi);
            
            sum1 = _mm256_fmadd_ps(d_lo, d_lo, sum1);
            sum1 = _mm256_fmadd_ps(d_hi, d_hi, sum1);
        }
        
        sum1 = _mm256_add_ps(sum1, sum2);
        float total = hsum256_ps(sum1);
        
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
