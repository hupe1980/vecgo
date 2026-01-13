#ifdef __ARM_FEATURE_SVE2
#include <arm_sve.h>
#include <stdint.h>

// sq8_sve2.c - SVE2 optimized SQ8 (8-bit scalar quantization) operations
//
// SQ8 stores vectors as int8 with per-vector or per-dimension scale/bias.

// sq8L2BatchSve2 computes squared L2 distance between a float32 query and SQ8 encoded vectors.
// Uses per-vector scale and bias.
void sq8L2BatchSve2(const float *__restrict__ query, const int8_t *__restrict__ codes,
                    const float *__restrict__ scales, const float *__restrict__ biases,
                    int64_t dim, int64_t n, float *__restrict__ out) {
    
    svbool_t pg32 = svptrue_b32();
    int64_t vl32 = svcntw();
    
    for (int64_t i = 0; i < n; i++) {
        const int8_t *code = codes + i * dim;
        float scale = scales[i];
        float bias = biases[i];
        
        svfloat32_t v_scale = svdup_f32(scale);
        svfloat32_t v_bias = svdup_f32(bias);
        svfloat32_t sum = svdup_f32(0.0f);
        
        int64_t j = 0;
        
        // Process vl32 elements at a time
        for (; j + vl32 <= dim; j += vl32) {
            // Load int8 codes and expand to float32
            svbool_t pg8 = svwhilelt_b8_s64(j, dim);
            svint8_t v_i8 = svld1_s8(pg8, code + j);
            
            // Unpack int8 -> int16 -> int32 -> float32
            svint16_t v_i16 = svunpklo_s16(v_i8);
            svint32_t v_i32 = svunpklo_s32(v_i16);
            svfloat32_t v_f32 = svcvt_f32_s32_x(pg32, v_i32);
            
            // Reconstruct: rec = code * scale + bias
            svfloat32_t v_rec = svmla_f32_x(pg32, v_bias, v_f32, v_scale);
            
            // Load query
            svfloat32_t v_q = svld1_f32(pg32, query + j);
            
            // Compute diff and accumulate
            svfloat32_t v_diff = svsub_f32_x(pg32, v_q, v_rec);
            sum = svmla_f32_x(pg32, sum, v_diff, v_diff);
        }
        
        // Handle tail with predication
        if (j < dim) {
            svbool_t pg_tail = svwhilelt_b32_s64(j, dim);
            svbool_t pg8_tail = svwhilelt_b8_s64(j, dim);
            
            svint8_t v_i8 = svld1_s8(pg8_tail, code + j);
            svint16_t v_i16 = svunpklo_s16(v_i8);
            svint32_t v_i32 = svunpklo_s32(v_i16);
            svfloat32_t v_f32 = svcvt_f32_s32_x(pg_tail, v_i32);
            
            svfloat32_t v_rec = svmla_f32_m(pg_tail, v_bias, v_f32, v_scale);
            svfloat32_t v_q = svld1_f32(pg_tail, query + j);
            svfloat32_t v_diff = svsub_f32_x(pg_tail, v_q, v_rec);
            sum = svmla_f32_m(pg_tail, sum, v_diff, v_diff);
        }
        
        out[i] = svaddv_f32(pg32, sum);
    }
}

// sq8uL2BatchPerDimensionSve2 computes squared L2 distance between a float32 query
// and SQ8 (uint8) encoded vectors with per-dimension scaling.
void sq8uL2BatchPerDimensionSve2(const float *__restrict__ query, const uint8_t *__restrict__ codes,
                                  const float *__restrict__ mins, const float *__restrict__ invScales,
                                  int64_t dim, int64_t n, float *__restrict__ out) {
    
    svbool_t pg32 = svptrue_b32();
    int64_t vl32 = svcntw();
    
    for (int64_t i = 0; i < n; i++) {
        const uint8_t *code = codes + i * dim;
        svfloat32_t sum = svdup_f32(0.0f);
        
        int64_t j = 0;
        
        // Process vl32 elements at a time
        for (; j + vl32 <= dim; j += vl32) {
            // Load uint8 codes and expand to float32
            svbool_t pg8 = svwhilelt_b8_s64(j, dim);
            svuint8_t v_u8 = svld1_u8(pg8, code + j);
            
            // Unpack uint8 -> uint16 -> uint32 -> float32
            svuint16_t v_u16 = svunpklo_u16(v_u8);
            svuint32_t v_u32 = svunpklo_u32(v_u16);
            svfloat32_t v_f32 = svcvt_f32_u32_x(pg32, v_u32);
            
            // Load per-dimension mins and invScales
            svfloat32_t v_min = svld1_f32(pg32, mins + j);
            svfloat32_t v_invScale = svld1_f32(pg32, invScales + j);
            
            // Reconstruct: rec = min + code * invScale
            svfloat32_t v_rec = svmla_f32_x(pg32, v_min, v_f32, v_invScale);
            
            // Load query
            svfloat32_t v_q = svld1_f32(pg32, query + j);
            
            // Compute diff and accumulate
            svfloat32_t v_diff = svsub_f32_x(pg32, v_q, v_rec);
            sum = svmla_f32_x(pg32, sum, v_diff, v_diff);
        }
        
        // Handle tail with predication
        if (j < dim) {
            svbool_t pg_tail = svwhilelt_b32_s64(j, dim);
            svbool_t pg8_tail = svwhilelt_b8_s64(j, dim);
            
            svuint8_t v_u8 = svld1_u8(pg8_tail, code + j);
            svuint16_t v_u16 = svunpklo_u16(v_u8);
            svuint32_t v_u32 = svunpklo_u32(v_u16);
            svfloat32_t v_f32 = svcvt_f32_u32_x(pg_tail, v_u32);
            
            svfloat32_t v_min = svld1_f32(pg_tail, mins + j);
            svfloat32_t v_invScale = svld1_f32(pg_tail, invScales + j);
            svfloat32_t v_rec = svmla_f32_m(pg_tail, v_min, v_f32, v_invScale);
            
            svfloat32_t v_q = svld1_f32(pg_tail, query + j);
            svfloat32_t v_diff = svsub_f32_x(pg_tail, v_q, v_rec);
            sum = svmla_f32_m(pg_tail, sum, v_diff, v_diff);
        }
        
        out[i] = svaddv_f32(pg32, sum);
    }
}

#endif // __ARM_FEATURE_SVE2
