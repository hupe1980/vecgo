#ifdef __ARM_FEATURE_SVE2
#include <arm_sve.h>
#include <stdint.h>

// pq_int8_sve2.c - SVE2 optimized PQ (Product Quantization) int8 operations
//
// PQ encodes vectors as M subvector codes, each code indexes into a codebook
// of 256 centroids stored as int8 with scale/offset for dequantization.

// squaredL2Int8DequantizedSve2 computes squared L2 distance between a float32 query
// and an int8 code vector after dequantization: rec = code[i] * scale + offset
void squaredL2Int8DequantizedSve2(const float *__restrict__ query,
                                   const int8_t *__restrict__ code,
                                   int64_t subdim,
                                   const float *__restrict__ scale,
                                   const float *__restrict__ offset,
                                   float *__restrict__ out) {
    float s = *scale;
    float o = *offset;
    
    svfloat32_t v_scale = svdup_f32(s);
    svfloat32_t v_offset = svdup_f32(o);
    svfloat32_t sum = svdup_f32(0.0f);
    
    svbool_t pg32 = svptrue_b32();
    int64_t vl32 = svcntw();
    int64_t j = 0;
    
    // Process vl32 elements at a time
    for (; j + vl32 <= subdim; j += vl32) {
        // Load int8 codes
        svbool_t pg8 = svwhilelt_b8_s64(j, subdim);
        svint8_t v_i8 = svld1_s8(pg8, code + j);
        
        // Unpack int8 -> int16 -> int32 -> float32
        svint16_t v_i16 = svunpklo_s16(v_i8);
        svint32_t v_i32 = svunpklo_s32(v_i16);
        svfloat32_t v_f32 = svcvt_f32_s32_x(pg32, v_i32);
        
        // Reconstruct: rec = code * scale + offset
        svfloat32_t v_rec = svmla_f32_x(pg32, v_offset, v_f32, v_scale);
        
        // Load query
        svfloat32_t v_q = svld1_f32(pg32, query + j);
        
        // Compute diff and accumulate
        svfloat32_t v_diff = svsub_f32_x(pg32, v_q, v_rec);
        sum = svmla_f32_x(pg32, sum, v_diff, v_diff);
    }
    
    // Handle tail with predication
    if (j < subdim) {
        svbool_t pg_tail = svwhilelt_b32_s64(j, subdim);
        svbool_t pg8_tail = svwhilelt_b8_s64(j, subdim);
        
        svint8_t v_i8 = svld1_s8(pg8_tail, code + j);
        svint16_t v_i16 = svunpklo_s16(v_i8);
        svint32_t v_i32 = svunpklo_s32(v_i16);
        svfloat32_t v_f32 = svcvt_f32_s32_x(pg_tail, v_i32);
        
        svfloat32_t v_rec = svmla_f32_m(pg_tail, v_offset, v_f32, v_scale);
        svfloat32_t v_q = svld1_f32(pg_tail, query + j);
        svfloat32_t v_diff = svsub_f32_x(pg_tail, v_q, v_rec);
        sum = svmla_f32_m(pg_tail, sum, v_diff, v_diff);
    }
    
    *out = svaddv_f32(pg32, sum);
}

// buildDistanceTableInt8Sve2 fills out[0:256] with distances between query and all
// 256 int8 centroids in codebook.
void buildDistanceTableInt8Sve2(const float *__restrict__ query,
                                 const int8_t *__restrict__ codebook,
                                 int64_t subdim,
                                 const float *__restrict__ scale,
                                 const float *__restrict__ offset,
                                 float *__restrict__ out) {
    float s = *scale;
    float o = *offset;
    
    svfloat32_t v_scale = svdup_f32(s);
    svfloat32_t v_offset = svdup_f32(o);
    svbool_t pg32 = svptrue_b32();
    int64_t vl32 = svcntw();
    
    for (int c = 0; c < 256; c++) {
        const int8_t *code = codebook + (int64_t)c * subdim;
        svfloat32_t sum = svdup_f32(0.0f);
        
        int64_t j = 0;
        for (; j + vl32 <= subdim; j += vl32) {
            svbool_t pg8 = svwhilelt_b8_s64(j, subdim);
            svint8_t v_i8 = svld1_s8(pg8, code + j);
            
            svint16_t v_i16 = svunpklo_s16(v_i8);
            svint32_t v_i32 = svunpklo_s32(v_i16);
            svfloat32_t v_f32 = svcvt_f32_s32_x(pg32, v_i32);
            
            svfloat32_t v_rec = svmla_f32_x(pg32, v_offset, v_f32, v_scale);
            svfloat32_t v_q = svld1_f32(pg32, query + j);
            svfloat32_t v_diff = svsub_f32_x(pg32, v_q, v_rec);
            sum = svmla_f32_x(pg32, sum, v_diff, v_diff);
        }
        
        // Handle tail
        if (j < subdim) {
            svbool_t pg_tail = svwhilelt_b32_s64(j, subdim);
            svbool_t pg8_tail = svwhilelt_b8_s64(j, subdim);
            
            svint8_t v_i8 = svld1_s8(pg8_tail, code + j);
            svint16_t v_i16 = svunpklo_s16(v_i8);
            svint32_t v_i32 = svunpklo_s32(v_i16);
            svfloat32_t v_f32 = svcvt_f32_s32_x(pg_tail, v_i32);
            
            svfloat32_t v_rec = svmla_f32_m(pg_tail, v_offset, v_f32, v_scale);
            svfloat32_t v_q = svld1_f32(pg_tail, query + j);
            svfloat32_t v_diff = svsub_f32_x(pg_tail, v_q, v_rec);
            sum = svmla_f32_m(pg_tail, sum, v_diff, v_diff);
        }
        
        out[c] = svaddv_f32(pg32, sum);
    }
}

// findNearestCentroidInt8Sve2 returns the index (0..255) of the closest centroid.
void findNearestCentroidInt8Sve2(const float *__restrict__ query,
                                  const int8_t *__restrict__ codebook,
                                  int64_t subdim,
                                  const float *__restrict__ scale,
                                  const float *__restrict__ offset,
                                  int64_t *__restrict__ outIndex) {
    float s = *scale;
    float o = *offset;
    
    svfloat32_t v_scale = svdup_f32(s);
    svfloat32_t v_offset = svdup_f32(o);
    svbool_t pg32 = svptrue_b32();
    int64_t vl32 = svcntw();
    
    float best = 0;
    int bestInit = 0;
    int64_t bestIdx = 0;
    
    for (int c = 0; c < 256; c++) {
        const int8_t *code = codebook + (int64_t)c * subdim;
        svfloat32_t sum = svdup_f32(0.0f);
        
        int64_t j = 0;
        for (; j + vl32 <= subdim; j += vl32) {
            svbool_t pg8 = svwhilelt_b8_s64(j, subdim);
            svint8_t v_i8 = svld1_s8(pg8, code + j);
            
            svint16_t v_i16 = svunpklo_s16(v_i8);
            svint32_t v_i32 = svunpklo_s32(v_i16);
            svfloat32_t v_f32 = svcvt_f32_s32_x(pg32, v_i32);
            
            svfloat32_t v_rec = svmla_f32_x(pg32, v_offset, v_f32, v_scale);
            svfloat32_t v_q = svld1_f32(pg32, query + j);
            svfloat32_t v_diff = svsub_f32_x(pg32, v_q, v_rec);
            sum = svmla_f32_x(pg32, sum, v_diff, v_diff);
        }
        
        // Handle tail
        if (j < subdim) {
            svbool_t pg_tail = svwhilelt_b32_s64(j, subdim);
            svbool_t pg8_tail = svwhilelt_b8_s64(j, subdim);
            
            svint8_t v_i8 = svld1_s8(pg8_tail, code + j);
            svint16_t v_i16 = svunpklo_s16(v_i8);
            svint32_t v_i32 = svunpklo_s32(v_i16);
            svfloat32_t v_f32 = svcvt_f32_s32_x(pg_tail, v_i32);
            
            svfloat32_t v_rec = svmla_f32_m(pg_tail, v_offset, v_f32, v_scale);
            svfloat32_t v_q = svld1_f32(pg_tail, query + j);
            svfloat32_t v_diff = svsub_f32_x(pg_tail, v_q, v_rec);
            sum = svmla_f32_m(pg_tail, sum, v_diff, v_diff);
        }
        
        float total = svaddv_f32(pg32, sum);
        
        if (!bestInit || total < best) {
            best = total;
            bestIdx = (int64_t)c;
            bestInit = 1;
        }
    }
    
    *outIndex = bestIdx;
}

#endif // __ARM_FEATURE_SVE2
