#include <immintrin.h>
#include <stdint.h>

// Sq8L2BatchAvx512 computes squared L2 distance between a float32 query and SQ8 encoded vectors.
void sq8L2BatchAvx512(const float *__restrict__ query, const int8_t *__restrict__ codes, 
                          const float *__restrict__ scales, const float *__restrict__ biases,
                          int64_t dim, int64_t n, float *__restrict__ out) {
    
    for (int64_t i = 0; i < n; i++) {
        const int8_t *code = codes + i * dim;
        float scale = scales[i];
        float bias = biases[i];
        
        __m512 v_scale = _mm512_set1_ps(scale);
        __m512 v_bias = _mm512_set1_ps(bias);
        __m512 sum = _mm512_setzero_ps();
        
        int64_t j = 0;
        for (; j <= dim - 16; j += 16) {
            // Load 16 int8 codes
            __m128i v_i8 = _mm_loadu_si128((__m128i const*)(code + j));
            // Sign extend to int32 (AVX-512 supports cvtepi8_epi32 from xmm to zmm)
            __m512i v_i32 = _mm512_cvtepi8_epi32(v_i8);
            // Convert to float
            __m512 v_f32 = _mm512_cvtepi32_ps(v_i32);
            
            // Reconstruct: val = code * scale + bias
            __m512 v_rec = _mm512_fmadd_ps(v_f32, v_scale, v_bias);
            
            // Load query
            __m512 v_q = _mm512_loadu_ps(query + j);
            
            // Diff
            __m512 v_diff = _mm512_sub_ps(v_q, v_rec);
            
            // Accumulate squared diff
            sum = _mm512_fmadd_ps(v_diff, v_diff, sum);
        }
        
        // Horizontal sum
        float total = _mm512_reduce_add_ps(sum);
        
        // Remainder
        for (; j < dim; j++) {
            float rec = (float)code[j] * scale + bias;
            float diff = query[j] - rec;
            total += diff * diff;
        }
        
        out[i] = total;
    }
}

// Sq8uL2BatchPerDimensionAvx512 computes squared L2 distance with per-dimension scaling.
void sq8uL2BatchPerDimensionAvx512(const float *__restrict__ query, const uint8_t *__restrict__ codes,
                                       const float *__restrict__ mins, const float *__restrict__ invScales,
                                       int64_t dim, int64_t n, float *__restrict__ out) {
    for (int64_t i = 0; i < n; i++) {
        const uint8_t *code = codes + i * dim;
        __m512 sum = _mm512_setzero_ps();

        int64_t j = 0;
        for (; j <= dim - 16; j += 16) {
            // Load 16 uint8 codes
            __m128i v_u8 = _mm_loadu_si128((__m128i const*)(code + j));
            // Zero extend to int32 (AVX-512 supports cvtepu8_epi32)
            __m512i v_i32 = _mm512_cvtepu8_epi32(v_u8);
            // Convert to float
            __m512 v_f32 = _mm512_cvtepi32_ps(v_i32);

            // Load per-dimension min and invScale
            __m512 v_min = _mm512_loadu_ps(mins + j);
            __m512 v_invScale = _mm512_loadu_ps(invScales + j);

            // Reconstruct: val = min + code * invScale
            // FMA: a * b + c -> v_f32 * v_invScale + v_min
            __m512 v_rec = _mm512_fmadd_ps(v_f32, v_invScale, v_min);

            // Load query
            __m512 v_q = _mm512_loadu_ps(query + j);

            // Diff
            __m512 v_diff = _mm512_sub_ps(v_q, v_rec);

            // Accumulate squared diff
            sum = _mm512_fmadd_ps(v_diff, v_diff, sum);
        }

        float total = _mm512_reduce_add_ps(sum);

        // Remainder
        for (; j < dim; j++) {
            float rec = mins[j] + (float)code[j] * invScales[j];
            float diff = query[j] - rec;
            total += diff * diff;
        }

        out[i] = total;
    }
}