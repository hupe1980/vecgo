#include <arm_neon.h>
#include <stdint.h>

// INT4 (4-bit) quantization SIMD kernels for ARM NEON
//
// INT4 is nibble-packed: high nibble = first value, low nibble = second value
// Dequantization: val[i] = (quant[i] / 15.0) * diff[i] + min[i]

// int4L2DistanceNeon computes squared L2 distance between query and INT4 code.
void int4L2DistanceNeon(const float *__restrict__ query,
                        const uint8_t *__restrict__ code,
                        int64_t dim,
                        const float *__restrict__ min,
                        const float *__restrict__ diff,
                        float *__restrict__ out) {
    // Avoid literal pool constants
    uint32x4_t z = vdupq_n_u32(0);
    float32x4_t sum = vreinterpretq_f32_u32(z);
    const float32x4_t scale = vdupq_n_f32(1.0f / 15.0f);
    const uint8x16_t nibble_mask = vdupq_n_u8(0x0F);

    int64_t i = 0;
    // Process 16 dimensions at a time (8 bytes of packed INT4)
    for (; i <= dim - 16; i += 16) {
        // Load 8 bytes of packed INT4 data
        uint8x8_t packed = vld1_u8(code + i / 2);
        
        // Unpack high nibbles (shift right by 4)
        uint8x8_t high_nibbles = vshr_n_u8(packed, 4);
        // Unpack low nibbles (mask)
        uint8x8_t low_nibbles = vand_u8(packed, vget_low_u8(nibble_mask));
        
        // Interleave: [h0,l0,h1,l1,h2,l2,h3,l3,h4,l4,h5,l5,h6,l6,h7,l7]
        uint8x8x2_t interleaved = vzip_u8(high_nibbles, low_nibbles);
        uint8x16_t all_vals = vcombine_u8(interleaved.val[0], interleaved.val[1]);
        
        // Process first 8 values
        uint16x8_t vals_16_lo = vmovl_u8(vget_low_u8(all_vals));
        uint32x4_t vals_32_0 = vmovl_u16(vget_low_u16(vals_16_lo));
        uint32x4_t vals_32_1 = vmovl_u16(vget_high_u16(vals_16_lo));
        
        float32x4_t f_0 = vcvtq_f32_u32(vals_32_0);
        float32x4_t f_1 = vcvtq_f32_u32(vals_32_1);
        
        // Load diff and min for first 8
        float32x4_t diff_0 = vld1q_f32(diff + i);
        float32x4_t diff_1 = vld1q_f32(diff + i + 4);
        float32x4_t min_0 = vld1q_f32(min + i);
        float32x4_t min_1 = vld1q_f32(min + i + 4);
        
        // Dequantize: val = (quant / 15.0) * diff + min
        float32x4_t dequant_0 = vfmaq_f32(min_0, vmulq_f32(f_0, scale), diff_0);
        float32x4_t dequant_1 = vfmaq_f32(min_1, vmulq_f32(f_1, scale), diff_1);
        
        // Load query
        float32x4_t q_0 = vld1q_f32(query + i);
        float32x4_t q_1 = vld1q_f32(query + i + 4);
        
        // Compute differences and accumulate
        float32x4_t d_0 = vsubq_f32(q_0, dequant_0);
        float32x4_t d_1 = vsubq_f32(q_1, dequant_1);
        sum = vfmaq_f32(sum, d_0, d_0);
        sum = vfmaq_f32(sum, d_1, d_1);
        
        // Process next 8 values
        uint16x8_t vals_16_hi = vmovl_u8(vget_high_u8(all_vals));
        uint32x4_t vals_32_2 = vmovl_u16(vget_low_u16(vals_16_hi));
        uint32x4_t vals_32_3 = vmovl_u16(vget_high_u16(vals_16_hi));
        
        float32x4_t f_2 = vcvtq_f32_u32(vals_32_2);
        float32x4_t f_3 = vcvtq_f32_u32(vals_32_3);
        
        float32x4_t diff_2 = vld1q_f32(diff + i + 8);
        float32x4_t diff_3 = vld1q_f32(diff + i + 12);
        float32x4_t min_2 = vld1q_f32(min + i + 8);
        float32x4_t min_3 = vld1q_f32(min + i + 12);
        
        float32x4_t dequant_2 = vfmaq_f32(min_2, vmulq_f32(f_2, scale), diff_2);
        float32x4_t dequant_3 = vfmaq_f32(min_3, vmulq_f32(f_3, scale), diff_3);
        
        float32x4_t q_2 = vld1q_f32(query + i + 8);
        float32x4_t q_3 = vld1q_f32(query + i + 12);
        
        float32x4_t d_2 = vsubq_f32(q_2, dequant_2);
        float32x4_t d_3 = vsubq_f32(q_3, dequant_3);
        sum = vfmaq_f32(sum, d_2, d_2);
        sum = vfmaq_f32(sum, d_3, d_3);
    }
    
    // Horizontal sum
    float total = vaddvq_f32(sum);
    
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

// int4L2DistancePrecomputedNeon uses precomputed lookup tables.
void int4L2DistancePrecomputedNeon(const float *__restrict__ query,
                                    const uint8_t *__restrict__ code,
                                    int64_t dim,
                                    const float *__restrict__ lookupTable,
                                    float *__restrict__ out) {
    uint32x4_t z = vdupq_n_u32(0);
    float32x4_t sum = vreinterpretq_f32_u32(z);
    
    int64_t i = 0;
    // Process 8 dimensions at a time
    for (; i <= dim - 8; i += 8) {
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
        
        float32x4_t dequant_0 = vld1q_f32(vals);
        float32x4_t dequant_1 = vld1q_f32(vals + 4);
        float32x4_t q_0 = vld1q_f32(query + i);
        float32x4_t q_1 = vld1q_f32(query + i + 4);
        
        float32x4_t d_0 = vsubq_f32(q_0, dequant_0);
        float32x4_t d_1 = vsubq_f32(q_1, dequant_1);
        sum = vfmaq_f32(sum, d_0, d_0);
        sum = vfmaq_f32(sum, d_1, d_1);
    }
    
    float total = vaddvq_f32(sum);
    
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

// int4L2DistanceBatchNeon computes L2 distances for multiple INT4 codes.
void int4L2DistanceBatchNeon(const float *__restrict__ query,
                             const uint8_t *__restrict__ codes,
                             int64_t dim,
                             int64_t n,
                             const float *__restrict__ min,
                             const float *__restrict__ diff,
                             float *__restrict__ out) {
    int64_t codeSize = (dim + 1) / 2;
    
    for (int64_t j = 0; j < n; j++) {
        int4L2DistanceNeon(query, codes + j * codeSize, dim, min, diff, out + j);
    }
}
