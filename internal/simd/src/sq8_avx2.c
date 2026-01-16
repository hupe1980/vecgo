#include <immintrin.h>
#include <stdint.h>

// Sq8L2BatchAvx2 computes squared L2 distance between a float32 query and SQ8 encoded vectors.
// query: float32 query vector (dim)
// codes: int8 encoded vectors (n * dim)
// scales: scale factor per vector (n)
// biases: bias per vector (n)
// dim: dimension
// n: number of vectors
// out: output distances (n)
void sq8L2BatchAvx2(const float *__restrict__ query, const int8_t *__restrict__ codes, 
                       const float *__restrict__ scales, const float *__restrict__ biases,
                       int64_t dim, int64_t n, float *__restrict__ out) {
    
    for (int64_t i = 0; i < n; i++) {
        const int8_t *code = codes + i * dim;
        float scale = scales[i];
        float bias = biases[i];
        
        __m256 v_scale = _mm256_set1_ps(scale);
        __m256 v_bias = _mm256_set1_ps(bias);
        __m256 sum = _mm256_setzero_ps();
        
        int64_t j = 0;
        for (; j <= dim - 8; j += 8) {
            // Load 8 int8 codes
            __m128i v_i8 = _mm_loadl_epi64((__m128i const*)(code + j));
            // Sign extend to int32
            __m256i v_i32 = _mm256_cvtepi8_epi32(v_i8);
            // Convert to float
            __m256 v_f32 = _mm256_cvtepi32_ps(v_i32);
            
            // Reconstruct: val = code * scale + bias
            // Use FMA: val = (code * scale) + bias
            __m256 v_rec = _mm256_fmadd_ps(v_f32, v_scale, v_bias);
            
            // Load query
            __m256 v_q = _mm256_loadu_ps(query + j);
            
            // Diff
            __m256 v_diff = _mm256_sub_ps(v_q, v_rec);
            
            // Accumulate squared diff
            sum = _mm256_fmadd_ps(v_diff, v_diff, sum);
        }
        
        // Horizontal sum
        float temp[8];
        _mm256_storeu_ps(temp, sum);
        float total = 0;
        for (int k = 0; k < 8; k++) total += temp[k];
        
        // Remainder
        for (; j < dim; j++) {
            float rec = (float)code[j] * scale + bias;
            float diff = query[j] - rec;
            total += diff * diff;
        }
        
        out[i] = total;
    }
}

// Sq8uL2BatchPerDimensionAvx2 computes squared L2 distance with per-dimension scaling.
// query: float32 query vector (dim)
// codes: uint8 encoded vectors (n * dim)
// mins: per-dimension minimum values (dim)
// invScales: per-dimension inverse scales (dim)
// dim: dimension
// n: number of vectors
// out: output distances (n)
void sq8uL2BatchPerDimensionAvx2(const float *__restrict__ query, const uint8_t *__restrict__ codes,
                                    const float *__restrict__ mins, const float *__restrict__ invScales,
                                    int64_t dim, int64_t n, float *__restrict__ out) {
    for (int64_t i = 0; i < n; i++) {
        const uint8_t *code = codes + i * dim;
        
        // Four accumulators for better ILP
        __m256 sum1 = _mm256_setzero_ps();
        __m256 sum2 = _mm256_setzero_ps();
        __m256 sum3 = _mm256_setzero_ps();
        __m256 sum4 = _mm256_setzero_ps();

        int64_t j = 0;
        for (; j <= dim - 32; j += 32) {
            // Load 32 uint8 codes in 4 batches of 8
            __m128i v_u8_1 = _mm_loadl_epi64((__m128i const*)(code + j));
            __m128i v_u8_2 = _mm_loadl_epi64((__m128i const*)(code + j + 8));
            __m128i v_u8_3 = _mm_loadl_epi64((__m128i const*)(code + j + 16));
            __m128i v_u8_4 = _mm_loadl_epi64((__m128i const*)(code + j + 24));
            
            // Zero extend to int32
            __m256i v_i32_1 = _mm256_cvtepu8_epi32(v_u8_1);
            __m256i v_i32_2 = _mm256_cvtepu8_epi32(v_u8_2);
            __m256i v_i32_3 = _mm256_cvtepu8_epi32(v_u8_3);
            __m256i v_i32_4 = _mm256_cvtepu8_epi32(v_u8_4);
            
            // Convert to float
            __m256 v_f32_1 = _mm256_cvtepi32_ps(v_i32_1);
            __m256 v_f32_2 = _mm256_cvtepi32_ps(v_i32_2);
            __m256 v_f32_3 = _mm256_cvtepi32_ps(v_i32_3);
            __m256 v_f32_4 = _mm256_cvtepi32_ps(v_i32_4);

            // Load per-dimension min and invScale
            __m256 v_min_1 = _mm256_loadu_ps(mins + j);
            __m256 v_min_2 = _mm256_loadu_ps(mins + j + 8);
            __m256 v_min_3 = _mm256_loadu_ps(mins + j + 16);
            __m256 v_min_4 = _mm256_loadu_ps(mins + j + 24);
            
            __m256 v_invScale_1 = _mm256_loadu_ps(invScales + j);
            __m256 v_invScale_2 = _mm256_loadu_ps(invScales + j + 8);
            __m256 v_invScale_3 = _mm256_loadu_ps(invScales + j + 16);
            __m256 v_invScale_4 = _mm256_loadu_ps(invScales + j + 24);

            // Reconstruct: val = min + code * invScale
            __m256 v_rec_1 = _mm256_fmadd_ps(v_f32_1, v_invScale_1, v_min_1);
            __m256 v_rec_2 = _mm256_fmadd_ps(v_f32_2, v_invScale_2, v_min_2);
            __m256 v_rec_3 = _mm256_fmadd_ps(v_f32_3, v_invScale_3, v_min_3);
            __m256 v_rec_4 = _mm256_fmadd_ps(v_f32_4, v_invScale_4, v_min_4);

            // Load query
            __m256 v_q_1 = _mm256_loadu_ps(query + j);
            __m256 v_q_2 = _mm256_loadu_ps(query + j + 8);
            __m256 v_q_3 = _mm256_loadu_ps(query + j + 16);
            __m256 v_q_4 = _mm256_loadu_ps(query + j + 24);

            // Diff
            __m256 v_diff_1 = _mm256_sub_ps(v_q_1, v_rec_1);
            __m256 v_diff_2 = _mm256_sub_ps(v_q_2, v_rec_2);
            __m256 v_diff_3 = _mm256_sub_ps(v_q_3, v_rec_3);
            __m256 v_diff_4 = _mm256_sub_ps(v_q_4, v_rec_4);

            // Accumulate squared diff
            sum1 = _mm256_fmadd_ps(v_diff_1, v_diff_1, sum1);
            sum2 = _mm256_fmadd_ps(v_diff_2, v_diff_2, sum2);
            sum3 = _mm256_fmadd_ps(v_diff_3, v_diff_3, sum3);
            sum4 = _mm256_fmadd_ps(v_diff_4, v_diff_4, sum4);
        }

        // Process remaining 8-element chunks
        for (; j <= dim - 8; j += 8) {
            __m128i v_u8 = _mm_loadl_epi64((__m128i const*)(code + j));
            __m256i v_i32 = _mm256_cvtepu8_epi32(v_u8);
            __m256 v_f32 = _mm256_cvtepi32_ps(v_i32);

            __m256 v_min = _mm256_loadu_ps(mins + j);
            __m256 v_invScale = _mm256_loadu_ps(invScales + j);
            __m256 v_rec = _mm256_fmadd_ps(v_f32, v_invScale, v_min);

            __m256 v_q = _mm256_loadu_ps(query + j);
            __m256 v_diff = _mm256_sub_ps(v_q, v_rec);
            sum1 = _mm256_fmadd_ps(v_diff, v_diff, sum1);
        }

        // Combine accumulators
        sum1 = _mm256_add_ps(sum1, sum2);
        sum3 = _mm256_add_ps(sum3, sum4);
        sum1 = _mm256_add_ps(sum1, sum3);

        // Horizontal sum
        float temp[8];
        _mm256_storeu_ps(temp, sum1);
        float total = temp[0] + temp[1] + temp[2] + temp[3] + 
                      temp[4] + temp[5] + temp[6] + temp[7];

        // Remainder
        for (; j < dim; j++) {
            float rec = mins[j] + (float)code[j] * invScales[j];
            float diff = query[j] - rec;
            total += diff * diff;
        }

        out[i] = total;
    }
}
