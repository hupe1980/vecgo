#ifdef __ARM_FEATURE_SVE2
#include <arm_sve.h>
#include <stdint.h>

// INT4 (4-bit) quantization SIMD kernels for ARM SVE2
//
// SVE2 provides scalable vector lengths from 128-2048 bits.
// INT4 is nibble-packed: high nibble = first value, low nibble = second value
// Dequantization: val[i] = (quant[i] / 15.0) * diff[i] + min[i]

// int4L2DistanceSve2 computes squared L2 distance between query and INT4 code.
void int4L2DistanceSve2(const float *__restrict__ query,
                        const uint8_t *__restrict__ code,
                        int64_t dim,
                        const float *__restrict__ min,
                        const float *__restrict__ diff,
                        float *__restrict__ out) {
    svfloat32_t sum = svdup_f32(0.0f);
    const float inv15 = 1.0f / 15.0f;
    
    int64_t i = 0;
    svbool_t all_true = svptrue_b8();
    
    // Get the vector length for float32 (number of float32 lanes)
    uint64_t vl = svcntw();
    
    // Process full SVE vectors worth of dimensions
    // Each byte contains 2 INT4 values, so we process 2*vl dimensions per iteration
    for (; i + 2 * vl <= dim; i += 2 * vl) {
        // Load packed INT4 bytes
        svuint8_t packed = svld1_u8(svptrue_b8(), code + i / 2);
        
        // Extract high nibbles (first values)
        svuint8_t high_nibbles = svlsr_n_u8_x(all_true, packed, 4);
        // Extract low nibbles (second values)
        svuint8_t low_nibbles = svand_n_u8_x(all_true, packed, 0x0F);
        
        // We need to interleave high and low nibbles
        // Process high nibbles first (even indices), then low nibbles (odd indices)
        
        // Convert high nibbles to float32
        svuint16_t h16 = svunpklo_u16(high_nibbles);
        svuint32_t h32 = svunpklo_u32(h16);
        svfloat32_t h_f32 = svcvt_f32_u32_x(svptrue_b32(), h32);
        
        // Load diff and min for high nibble positions (even indices)
        // We need strided load or gather - use interleaved approach
        svfloat32_t diff_h = svld1_gather_s32index_f32(svptrue_b32(), diff + i, 
                                                        svindex_s32(0, 2));
        svfloat32_t min_h = svld1_gather_s32index_f32(svptrue_b32(), min + i,
                                                       svindex_s32(0, 2));
        
        // Dequantize high nibbles
        svfloat32_t scaled_h = svmul_n_f32_x(svptrue_b32(), h_f32, inv15);
        svfloat32_t dequant_h = svmla_f32_x(svptrue_b32(), min_h, scaled_h, diff_h);
        
        // Load query for high nibble positions
        svfloat32_t q_h = svld1_gather_s32index_f32(svptrue_b32(), query + i,
                                                     svindex_s32(0, 2));
        
        // Compute differences
        svfloat32_t d_h = svsub_f32_x(svptrue_b32(), q_h, dequant_h);
        sum = svmla_f32_x(svptrue_b32(), sum, d_h, d_h);
        
        // Now process low nibbles (odd indices)
        svuint16_t l16 = svunpklo_u16(low_nibbles);
        svuint32_t l32 = svunpklo_u32(l16);
        svfloat32_t l_f32 = svcvt_f32_u32_x(svptrue_b32(), l32);
        
        svfloat32_t diff_l = svld1_gather_s32index_f32(svptrue_b32(), diff + i + 1,
                                                        svindex_s32(0, 2));
        svfloat32_t min_l = svld1_gather_s32index_f32(svptrue_b32(), min + i + 1,
                                                       svindex_s32(0, 2));
        
        svfloat32_t scaled_l = svmul_n_f32_x(svptrue_b32(), l_f32, inv15);
        svfloat32_t dequant_l = svmla_f32_x(svptrue_b32(), min_l, scaled_l, diff_l);
        
        svfloat32_t q_l = svld1_gather_s32index_f32(svptrue_b32(), query + i + 1,
                                                     svindex_s32(0, 2));
        
        svfloat32_t d_l = svsub_f32_x(svptrue_b32(), q_l, dequant_l);
        sum = svmla_f32_x(svptrue_b32(), sum, d_l, d_l);
    }
    
    // Horizontal sum
    float total = svaddv_f32(svptrue_b32(), sum);
    
    // Scalar tail
    for (; i < dim; i += 2) {
        uint8_t packed_byte = code[i / 2];
        uint8_t q1 = (packed_byte >> 4) & 0x0F;
        uint8_t q2 = packed_byte & 0x0F;
        
        float val1 = ((float)q1 * inv15) * diff[i] + min[i];
        float d1 = query[i] - val1;
        total += d1 * d1;
        
        if (i + 1 < dim) {
            float val2 = ((float)q2 * inv15) * diff[i + 1] + min[i + 1];
            float d2 = query[i + 1] - val2;
            total += d2 * d2;
        }
    }
    
    *out = total;
}

// int4L2DistancePrecomputedSve2 uses precomputed lookup tables.
// Uses SVE2 table lookup for fast dequantization.
void int4L2DistancePrecomputedSve2(const float *__restrict__ query,
                                    const uint8_t *__restrict__ code,
                                    int64_t dim,
                                    const float *__restrict__ lookupTable,
                                    float *__restrict__ out) {
    svfloat32_t sum = svdup_f32(0.0f);
    
    int64_t i = 0;
    uint64_t vl = svcntw();
    
    // Process vectors at a time
    for (; i + vl <= dim; i += vl) {
        // Load query vector
        svfloat32_t q = svld1_f32(svptrue_b32(), query + i);
        
        // Load and dequantize INT4 values using gather loads
        // Each dimension has 16 possible values in lookup table
        svfloat32_t dequant = svdup_f32(0.0f);
        
        // Use scalar lookups and then vectorize
        float vals[64];  // Max SVE vector length / 32 bits
        for (uint64_t k = 0; k < vl && i + k < dim; k++) {
            uint64_t byte_idx = (i + k) / 2;
            uint8_t packed_byte = code[byte_idx];
            uint8_t nibble = ((i + k) % 2 == 0) ? 
                             ((packed_byte >> 4) & 0x0F) : 
                             (packed_byte & 0x0F);
            vals[k] = lookupTable[(i + k) * 16 + nibble];
        }
        
        dequant = svld1_f32(svptrue_b32(), vals);
        
        // Compute difference and accumulate
        svfloat32_t d = svsub_f32_x(svptrue_b32(), q, dequant);
        sum = svmla_f32_x(svptrue_b32(), sum, d, d);
    }
    
    // Horizontal sum
    float total = svaddv_f32(svptrue_b32(), sum);
    
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

// int4L2DistanceBatchSve2 computes L2 distances for multiple INT4 codes.
void int4L2DistanceBatchSve2(const float *__restrict__ query,
                             const uint8_t *__restrict__ codes,
                             int64_t dim,
                             int64_t n,
                             const float *__restrict__ min,
                             const float *__restrict__ diff,
                             float *__restrict__ out) {
    int64_t codeSize = (dim + 1) / 2;
    
    for (int64_t j = 0; j < n; j++) {
        int4L2DistanceSve2(query, codes + j * codeSize, dim, min, diff, out + j);
    }
}

#endif // __ARM_FEATURE_SVE2
