#include <immintrin.h>
#include <stdint.h>

// squaredL2Int8DequantizedAvx computes squared L2 distance between a float32 query
// and an int8 code vector after dequantization:
//   rec = code[i]*scale + offset
//
// query: float32 query vector (subdim)
// code:  int8 code vector (subdim)
// subdim: length of vectors
// scale/offset: dequantization params
// out: output distance
void squaredL2Int8DequantizedAvx(const float *__restrict__ query,
                                const int8_t *__restrict__ code,
                                int64_t subdim,
                                const float *__restrict__ scale,
                                const float *__restrict__ offset,
                                float *__restrict__ out) {
    float s = *scale;
    float o = *offset;
    __m256 v_scale = _mm256_set1_ps(s);
    __m256 v_offset = _mm256_set1_ps(o);
    __m256 sum = _mm256_setzero_ps();

    int64_t j = 0;
    for (; j <= subdim - 8; j += 8) {
        __m128i v_i8 = _mm_loadl_epi64((__m128i const *)(code + j));
        __m256i v_i32 = _mm256_cvtepi8_epi32(v_i8);
        __m256 v_f32 = _mm256_cvtepi32_ps(v_i32);

        __m256 v_rec = _mm256_fmadd_ps(v_f32, v_scale, v_offset);
        __m256 v_q = _mm256_loadu_ps(query + j);
        __m256 v_diff = _mm256_sub_ps(v_q, v_rec);
        sum = _mm256_fmadd_ps(v_diff, v_diff, sum);
    }

    float tmp[8];
    _mm256_storeu_ps(tmp, sum);
    float total = 0;
    for (int k = 0; k < 8; k++) {
        total += tmp[k];
    }

    for (; j < subdim; j++) {
        float rec = (float)code[j] * s + o;
        float diff = query[j] - rec;
        total += diff * diff;
    }

    *out = total;
}

// buildDistanceTableInt8Avx fills out[0:256] with distances between query and all
// 256 int8 centroids in codebook.
// codebook is laid out as 256 consecutive centroids, each of length subdim.
void buildDistanceTableInt8Avx(const float *__restrict__ query,
                              const int8_t *__restrict__ codebook,
                              int64_t subdim,
                              const float *__restrict__ scale,
                              const float *__restrict__ offset,
                              float *__restrict__ out) {
    float s = *scale;
    float o = *offset;
    __m256 v_scale = _mm256_set1_ps(s);
    __m256 v_offset = _mm256_set1_ps(o);

    for (int c = 0; c < 256; c++) {
        const int8_t *code = codebook + (int64_t)c * subdim;
        __m256 sum = _mm256_setzero_ps();

        int64_t j = 0;
        for (; j <= subdim - 8; j += 8) {
            __m128i v_i8 = _mm_loadl_epi64((__m128i const *)(code + j));
            __m256i v_i32 = _mm256_cvtepi8_epi32(v_i8);
            __m256 v_f32 = _mm256_cvtepi32_ps(v_i32);

            __m256 v_rec = _mm256_fmadd_ps(v_f32, v_scale, v_offset);
            __m256 v_q = _mm256_loadu_ps(query + j);
            __m256 v_diff = _mm256_sub_ps(v_q, v_rec);
            sum = _mm256_fmadd_ps(v_diff, v_diff, sum);
        }

        float tmp[8];
        _mm256_storeu_ps(tmp, sum);
        float total = 0;
        for (int k = 0; k < 8; k++) {
            total += tmp[k];
        }

        for (; j < subdim; j++) {
            float rec = (float)code[j] * s + o;
            float diff = query[j] - rec;
            total += diff * diff;
        }

        out[c] = total;
    }
}

// findNearestCentroidInt8Avx returns the index (0..255) of the closest centroid.
void findNearestCentroidInt8Avx(const float *__restrict__ query,
                               const int8_t *__restrict__ codebook,
                               int64_t subdim,
                               const float *__restrict__ scale,
                               const float *__restrict__ offset,
                               int64_t *__restrict__ outIndex) {
    float s = *scale;
    float o = *offset;
    __m256 v_scale = _mm256_set1_ps(s);
    __m256 v_offset = _mm256_set1_ps(o);

    float best = 0;
    int bestInit = 0;
    int64_t bestIdx = 0;

    for (int c = 0; c < 256; c++) {
        const int8_t *code = codebook + (int64_t)c * subdim;
        __m256 sum = _mm256_setzero_ps();

        int64_t j = 0;
        for (; j <= subdim - 8; j += 8) {
            __m128i v_i8 = _mm_loadl_epi64((__m128i const *)(code + j));
            __m256i v_i32 = _mm256_cvtepi8_epi32(v_i8);
            __m256 v_f32 = _mm256_cvtepi32_ps(v_i32);

            __m256 v_rec = _mm256_fmadd_ps(v_f32, v_scale, v_offset);
            __m256 v_q = _mm256_loadu_ps(query + j);
            __m256 v_diff = _mm256_sub_ps(v_q, v_rec);
            sum = _mm256_fmadd_ps(v_diff, v_diff, sum);
        }

        float tmp[8];
        _mm256_storeu_ps(tmp, sum);
        float total = 0;
        for (int k = 0; k < 8; k++) {
            total += tmp[k];
        }

        for (; j < subdim; j++) {
            float rec = (float)code[j] * s + o;
            float diff = query[j] - rec;
            total += diff * diff;
        }

        if (!bestInit || total < best) {
            best = total;
            bestIdx = (int64_t)c;
            bestInit = 1;
        }
    }

    *outIndex = bestIdx;
}
