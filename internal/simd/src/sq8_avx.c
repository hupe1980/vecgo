#include <immintrin.h>
#include <stdint.h>

// Sq8L2BatchAvx computes squared L2 distance between a float32 query and SQ8 encoded vectors.
// query: float32 query vector (dim)
// codes: int8 encoded vectors (n * dim)
// scales: scale factor per vector (n)
// biases: bias per vector (n)
// dim: dimension
// n: number of vectors
// out: output distances (n)
void sq8L2BatchAvx(const float *__restrict__ query, const int8_t *__restrict__ codes, 
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
