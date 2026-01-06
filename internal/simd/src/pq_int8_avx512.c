#include <immintrin.h>
#include <stdint.h>

// squaredL2Int8DequantizedAvx512 computes squared L2 distance between a float32 query
// and an int8 code vector after dequantization:
//   rec = code[i]*(*scale) + (*offset)
//
// query: float32 query vector (subdim)
// code:  int8 code vector (subdim)
// subdim: length of vectors
// scale/offset: pointers to dequantization params
// out: output distance
void squaredL2Int8DequantizedAvx512(const float *__restrict__ query,
                                   const int8_t *__restrict__ code,
                                   int64_t subdim,
                                   const float *__restrict__ scale,
                                   const float *__restrict__ offset,
                                   float *__restrict__ out) {
    float s = *scale;
    float o = *offset;

    __m512 v_scale = _mm512_set1_ps(s);
    __m512 v_offset = _mm512_set1_ps(o);
    __m512 sum = _mm512_setzero_ps();

    int64_t j = 0;
    for (; j <= subdim - 16; j += 16) {
        __m128i v_i8 = _mm_loadu_si128((__m128i const *)(code + j));
        __m512i v_i32 = _mm512_cvtepi8_epi32(v_i8);
        __m512 v_f32 = _mm512_cvtepi32_ps(v_i32);

        __m512 v_rec = _mm512_fmadd_ps(v_f32, v_scale, v_offset);
        __m512 v_q = _mm512_loadu_ps(query + j);
        __m512 v_diff = _mm512_sub_ps(v_q, v_rec);
        sum = _mm512_fmadd_ps(v_diff, v_diff, sum);
    }

    float tmp[16];
    _mm512_storeu_ps(tmp, sum);
    float total = 0;
    for (int k = 0; k < 16; k++) {
        total += tmp[k];
    }

    for (; j < subdim; j++) {
        float rec = (float)code[j] * s + o;
        float diff = query[j] - rec;
        total += diff * diff;
    }

    *out = total;
}

// buildDistanceTableInt8Avx512 fills out[0:256] with distances between query and all
// 256 int8 centroids in codebook.
// codebook is laid out as 256 consecutive centroids, each of length subdim.
void buildDistanceTableInt8Avx512(const float *__restrict__ query,
                                 const int8_t *__restrict__ codebook,
                                 int64_t subdim,
                                 const float *__restrict__ scale,
                                 const float *__restrict__ offset,
                                 float *__restrict__ out) {
    float s = *scale;
    float o = *offset;

    __m512 v_scale = _mm512_set1_ps(s);
    __m512 v_offset = _mm512_set1_ps(o);

    for (int c = 0; c < 256; c++) {
        const int8_t *code = codebook + (int64_t)c * subdim;
        __m512 sum = _mm512_setzero_ps();

        int64_t j = 0;
        for (; j <= subdim - 16; j += 16) {
            __m128i v_i8 = _mm_loadu_si128((__m128i const *)(code + j));
            __m512i v_i32 = _mm512_cvtepi8_epi32(v_i8);
            __m512 v_f32 = _mm512_cvtepi32_ps(v_i32);

            __m512 v_rec = _mm512_fmadd_ps(v_f32, v_scale, v_offset);
            __m512 v_q = _mm512_loadu_ps(query + j);
            __m512 v_diff = _mm512_sub_ps(v_q, v_rec);
            sum = _mm512_fmadd_ps(v_diff, v_diff, sum);
        }

        float tmp[16];
        _mm512_storeu_ps(tmp, sum);
        float total = 0;
        for (int k = 0; k < 16; k++) {
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

// findNearestCentroidInt8Avx512 returns the index (0..255) of the closest centroid.
void findNearestCentroidInt8Avx512(const float *__restrict__ query,
                                  const int8_t *__restrict__ codebook,
                                  int64_t subdim,
                                  const float *__restrict__ scale,
                                  const float *__restrict__ offset,
                                  int64_t *__restrict__ outIndex) {
    float s = *scale;
    float o = *offset;

    __m512 v_scale = _mm512_set1_ps(s);
    __m512 v_offset = _mm512_set1_ps(o);

    float best = 0;
    int bestInit = 0;
    int64_t bestIdx = 0;

    for (int c = 0; c < 256; c++) {
        const int8_t *code = codebook + (int64_t)c * subdim;
        __m512 sum = _mm512_setzero_ps();

        int64_t j = 0;
        for (; j <= subdim - 16; j += 16) {
            __m128i v_i8 = _mm_loadu_si128((__m128i const *)(code + j));
            __m512i v_i32 = _mm512_cvtepi8_epi32(v_i8);
            __m512 v_f32 = _mm512_cvtepi32_ps(v_i32);

            __m512 v_rec = _mm512_fmadd_ps(v_f32, v_scale, v_offset);
            __m512 v_q = _mm512_loadu_ps(query + j);
            __m512 v_diff = _mm512_sub_ps(v_q, v_rec);
            sum = _mm512_fmadd_ps(v_diff, v_diff, sum);
        }

        float tmp[16];
        _mm512_storeu_ps(tmp, sum);
        float total = 0;
        for (int k = 0; k < 16; k++) {
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
