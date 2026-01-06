#include <arm_neon.h>
#include <stdint.h>

// squaredL2Int8DequantizedNeon computes squared L2 distance between a float32 query
// and an int8 code vector after dequantization:
//   rec = code[i]*scale + offset
void squaredL2Int8DequantizedNeon(const float *__restrict__ query,
                                 const int8_t *__restrict__ code,
                                 int64_t subdim,
                                 const float *__restrict__ scale,
                                 const float *__restrict__ offset,
                                 float *__restrict__ out) {
    float s = *scale;
    float o = *offset;
    float32x4_t v_scale = vdupq_n_f32(s);
    float32x4_t v_offset = vdupq_n_f32(o);
    // Avoid literal-pool constants; keep generator relocation-free.
    uint32x4_t z = vdupq_n_u32(0);
    float32x4_t sum = vreinterpretq_f32_u32(z);

    int64_t j = 0;
    for (; j <= subdim - 8; j += 8) {
        int8x8_t v_i8 = vld1_s8(code + j);
        int16x8_t v_i16 = vmovl_s8(v_i8);

        int32x4_t v_i32_low = vmovl_s16(vget_low_s16(v_i16));
        int32x4_t v_i32_high = vmovl_s16(vget_high_s16(v_i16));

        float32x4_t v_f32_low = vcvtq_f32_s32(v_i32_low);
        float32x4_t v_f32_high = vcvtq_f32_s32(v_i32_high);

        float32x4_t v_rec_low = vfmaq_f32(v_offset, v_f32_low, v_scale);
        float32x4_t v_rec_high = vfmaq_f32(v_offset, v_f32_high, v_scale);

        float32x4_t v_q_low = vld1q_f32(query + j);
        float32x4_t v_q_high = vld1q_f32(query + j + 4);

        float32x4_t v_diff_low = vsubq_f32(v_q_low, v_rec_low);
        float32x4_t v_diff_high = vsubq_f32(v_q_high, v_rec_high);

        sum = vfmaq_f32(sum, v_diff_low, v_diff_low);
        sum = vfmaq_f32(sum, v_diff_high, v_diff_high);
    }

    float total = vaddvq_f32(sum);

    for (; j < subdim; j++) {
        float rec = (float)code[j] * s + o;
        float diff = query[j] - rec;
        total += diff * diff;
    }

    *out = total;
}

// buildDistanceTableInt8Neon fills out[0:256] with distances between query and all
// 256 int8 centroids in codebook.
void buildDistanceTableInt8Neon(const float *__restrict__ query,
                               const int8_t *__restrict__ codebook,
                               int64_t subdim,
                               const float *__restrict__ scale,
                               const float *__restrict__ offset,
                               float *__restrict__ out) {
    float s = *scale;
    float o = *offset;
    float32x4_t v_scale = vdupq_n_f32(s);
    float32x4_t v_offset = vdupq_n_f32(o);

    for (int c = 0; c < 256; c++) {
        const int8_t *code = codebook + (int64_t)c * subdim;
        uint32x4_t z = vdupq_n_u32(0);
        float32x4_t sum = vreinterpretq_f32_u32(z);

        int64_t j = 0;
        for (; j <= subdim - 8; j += 8) {
            int8x8_t v_i8 = vld1_s8(code + j);
            int16x8_t v_i16 = vmovl_s8(v_i8);

            int32x4_t v_i32_low = vmovl_s16(vget_low_s16(v_i16));
            int32x4_t v_i32_high = vmovl_s16(vget_high_s16(v_i16));

            float32x4_t v_f32_low = vcvtq_f32_s32(v_i32_low);
            float32x4_t v_f32_high = vcvtq_f32_s32(v_i32_high);

            float32x4_t v_rec_low = vfmaq_f32(v_offset, v_f32_low, v_scale);
            float32x4_t v_rec_high = vfmaq_f32(v_offset, v_f32_high, v_scale);

            float32x4_t v_q_low = vld1q_f32(query + j);
            float32x4_t v_q_high = vld1q_f32(query + j + 4);

            float32x4_t v_diff_low = vsubq_f32(v_q_low, v_rec_low);
            float32x4_t v_diff_high = vsubq_f32(v_q_high, v_rec_high);

            sum = vfmaq_f32(sum, v_diff_low, v_diff_low);
            sum = vfmaq_f32(sum, v_diff_high, v_diff_high);
        }

        float total = vaddvq_f32(sum);

        for (; j < subdim; j++) {
            float rec = (float)code[j] * s + o;
            float diff = query[j] - rec;
            total += diff * diff;
        }

        out[c] = total;
    }
}

// findNearestCentroidInt8Neon returns the index (0..255) of the closest centroid.
void findNearestCentroidInt8Neon(const float *__restrict__ query,
                                const int8_t *__restrict__ codebook,
                                int64_t subdim,
                                const float *__restrict__ scale,
                                const float *__restrict__ offset,
                                int64_t *__restrict__ outIndex) {
    float s = *scale;
    float o = *offset;
    float32x4_t v_scale = vdupq_n_f32(s);
    float32x4_t v_offset = vdupq_n_f32(o);

    float best = 0;
    int bestInit = 0;
    int64_t bestIdx = 0;

    for (int c = 0; c < 256; c++) {
        const int8_t *code = codebook + (int64_t)c * subdim;
        uint32x4_t z = vdupq_n_u32(0);
        float32x4_t sum = vreinterpretq_f32_u32(z);

        int64_t j = 0;
        for (; j <= subdim - 8; j += 8) {
            int8x8_t v_i8 = vld1_s8(code + j);
            int16x8_t v_i16 = vmovl_s8(v_i8);

            int32x4_t v_i32_low = vmovl_s16(vget_low_s16(v_i16));
            int32x4_t v_i32_high = vmovl_s16(vget_high_s16(v_i16));

            float32x4_t v_f32_low = vcvtq_f32_s32(v_i32_low);
            float32x4_t v_f32_high = vcvtq_f32_s32(v_i32_high);

            float32x4_t v_rec_low = vfmaq_f32(v_offset, v_f32_low, v_scale);
            float32x4_t v_rec_high = vfmaq_f32(v_offset, v_f32_high, v_scale);

            float32x4_t v_q_low = vld1q_f32(query + j);
            float32x4_t v_q_high = vld1q_f32(query + j + 4);

            float32x4_t v_diff_low = vsubq_f32(v_q_low, v_rec_low);
            float32x4_t v_diff_high = vsubq_f32(v_q_high, v_rec_high);

            sum = vfmaq_f32(sum, v_diff_low, v_diff_low);
            sum = vfmaq_f32(sum, v_diff_high, v_diff_high);
        }

        float total = vaddvq_f32(sum);

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
