#include <arm_neon.h>
#include <stdint.h>

// Sq8L2BatchNeon computes squared L2 distance between a float32 query and SQ8 encoded vectors.
void sq8L2BatchNeon(const float *__restrict__ query, const int8_t *__restrict__ codes, 
                        const float *__restrict__ scales, const float *__restrict__ biases,
                        int64_t dim, int64_t n, float *__restrict__ out) {
    
    for (int64_t i = 0; i < n; i++) {
        const int8_t *code = codes + i * dim;
        float scale = scales[i];
        float bias = biases[i];
        
        float32x4_t v_scale = vdupq_n_f32(scale);
        float32x4_t v_bias = vdupq_n_f32(bias);
        float32x4_t sum = vdupq_n_f32(0.0f);
        
        int64_t j = 0;
        for (; j <= dim - 8; j += 8) {
            // Load 8 int8 codes
            int8x8_t v_i8 = vld1_s8(code + j);
            
            // Expand to int16
            int16x8_t v_i16 = vmovl_s8(v_i8);
            
            // Split into two int32x4
            int32x4_t v_i32_low = vmovl_s16(vget_low_s16(v_i16));
            int32x4_t v_i32_high = vmovl_s16(vget_high_s16(v_i16));
            
            // Convert to float
            float32x4_t v_f32_low = vcvtq_f32_s32(v_i32_low);
            float32x4_t v_f32_high = vcvtq_f32_s32(v_i32_high);
            
            // Reconstruct low
            float32x4_t v_rec_low = vfmaq_f32(v_bias, v_f32_low, v_scale);
            // Reconstruct high
            float32x4_t v_rec_high = vfmaq_f32(v_bias, v_f32_high, v_scale);
            
            // Load query
            float32x4_t v_q_low = vld1q_f32(query + j);
            float32x4_t v_q_high = vld1q_f32(query + j + 4);
            
            // Diff
            float32x4_t v_diff_low = vsubq_f32(v_q_low, v_rec_low);
            float32x4_t v_diff_high = vsubq_f32(v_q_high, v_rec_high);
            
            // Accumulate squared diff
            sum = vfmaq_f32(sum, v_diff_low, v_diff_low);
            sum = vfmaq_f32(sum, v_diff_high, v_diff_high);
        }
        
        // Horizontal sum
        float total = vaddvq_f32(sum);
        
        // Remainder
        for (; j < dim; j++) {
            float rec = (float)code[j] * scale + bias;
            float diff = query[j] - rec;
            total += diff * diff;
        }
        
        out[i] = total;
    }
}

// Sq8uL2BatchPerDimensionNeon computes squared L2 distance between a float32 query and SQ8 (uint8) encoded vectors with per-dimension scaling.
void sq8uL2BatchPerDimensionNeon(const float *__restrict__ query, const uint8_t *__restrict__ codes, 
                        const float *__restrict__ mins, const float *__restrict__ invScales,
                        int64_t dim, int64_t n, float *__restrict__ out) {
    
    for (int64_t i = 0; i < n; i++) {
        const uint8_t *code = codes + i * dim;
        float32x4_t sum = vdupq_n_f32(0.0f);
        
        int64_t j = 0;
        for (; j <= dim - 8; j += 8) {
            // Load 8 uint8 codes
            uint8x8_t v_u8 = vld1_u8(code + j);
            
            // Expand to uint16
            uint16x8_t v_u16 = vmovl_u8(v_u8);
            
            // Split into two uint32x4 -> float32x4
            uint32x4_t v_u32_low = vmovl_u16(vget_low_u16(v_u16));
            uint32x4_t v_u32_high = vmovl_u16(vget_high_u16(v_u16));
            
            float32x4_t v_f32_low = vcvtq_f32_u32(v_u32_low);
            float32x4_t v_f32_high = vcvtq_f32_u32(v_u32_high);
            
            // Load mins and invScales
            float32x4_t v_min_low = vld1q_f32(mins + j);
            float32x4_t v_min_high = vld1q_f32(mins + j + 4);
            
            float32x4_t v_invScale_low = vld1q_f32(invScales + j);
            float32x4_t v_invScale_high = vld1q_f32(invScales + j + 4);
            
            // Reconstruct: min + val * invScale
            float32x4_t v_rec_low = vfmaq_f32(v_min_low, v_f32_low, v_invScale_low);
            float32x4_t v_rec_high = vfmaq_f32(v_min_high, v_f32_high, v_invScale_high);
            
            // Load query
            float32x4_t v_q_low = vld1q_f32(query + j);
            float32x4_t v_q_high = vld1q_f32(query + j + 4);
            
            // Diff
            float32x4_t v_diff_low = vsubq_f32(v_q_low, v_rec_low);
            float32x4_t v_diff_high = vsubq_f32(v_q_high, v_rec_high);
            
            // Accumulate squared diff
            sum = vfmaq_f32(sum, v_diff_low, v_diff_low);
            sum = vfmaq_f32(sum, v_diff_high, v_diff_high);
        }
        
        // Handle remaining elements
        float s = vaddvq_f32(sum);
        for (; j < dim; j++) {
            float val = mins[j] + (float)code[j] * invScales[j];
            float diff = query[j] - val;
            s += diff * diff;
        }
        out[i] = s;
    }
}
