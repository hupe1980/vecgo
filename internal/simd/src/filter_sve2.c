// SIMD filter operations using ARM SVE2 for numeric indexing
// Compile with: clang -O3 -march=armv9-a+sve2 -S -o filter_sve2.s filter_sve2.c
// SVE2 provides scalable vectors (128-2048 bits) for better performance
// on modern ARM servers (AWS Graviton 3+, Ampere Altra, etc.)

#include <arm_sve.h>
#include <stdint.h>

// FilterRangeF64SVE2 - Check if values are in range [minVal, maxVal]
// Uses SVE2's scalable vectors for optimal performance across different hardware
//
// Arguments:
//   values: pointer to float64 array
//   n: number of elements
//   minVal: minimum value (inclusive)
//   maxVal: maximum value (inclusive)
//   dst: output byte array (1 = in range, 0 = out of range)
void filterRangeF64Sve2(
    const double* __restrict__ values,
    int64_t n,
    double minVal,
    double maxVal,
    uint8_t* __restrict__ dst
) {
    int64_t i = 0;
    
    // SVE2 approach: process elements and use svcompact/scalar extraction
    // Since SVE2 predicate results need to be converted to bytes, we use
    // a hybrid approach for correctness
    svbool_t pg64 = svptrue_b64();
    int64_t vl = svcntd();  // Vector length for 64-bit elements
    
    // Process full vectors
    for (; i + vl <= n; i += vl) {
        svfloat64_t v = svld1_f64(pg64, &values[i]);
        
        // Compare: v >= minVal AND v <= maxVal (inclusive bounds)
        svbool_t cmp_ge = svcmpge_n_f64(pg64, v, minVal);
        svbool_t cmp_le = svcmple_n_f64(pg64, v, maxVal);
        svbool_t in_range = svand_b_z(pg64, cmp_ge, cmp_le);
        
        // Convert predicate to uint64 vector: 0 or 1 per lane
        svuint64_t ones = svdup_u64(1);
        svuint64_t zeros = svdup_u64(0);
        svuint64_t result = svsel_u64(in_range, ones, zeros);
        
        // Store to temp buffer and copy to dst as bytes
        // This is the cleanest way to handle 64-bit to 8-bit narrowing
        uint64_t tmp[32];  // Max SVE vector is 2048 bits = 32 x 64-bit
        svst1_u64(pg64, tmp, result);
        
        for (int64_t j = 0; j < vl; j++) {
            dst[i + j] = (uint8_t)tmp[j];
        }
    }
    
    // Handle remainder with predicated operations
    if (i < n) {
        svbool_t pg = svwhilelt_b64(i, n);
        svfloat64_t v = svld1_f64(pg, &values[i]);
        
        svbool_t cmp_ge = svcmpge_n_f64(pg, v, minVal);
        svbool_t cmp_le = svcmple_n_f64(pg, v, maxVal);
        svbool_t in_range = svand_b_z(pg, cmp_ge, cmp_le);
        
        svuint64_t ones = svdup_u64(1);
        svuint64_t zeros = svdup_u64(0);
        svuint64_t result = svsel_u64(in_range, ones, zeros);
        
        uint64_t tmp[32];
        svst1_u64(pg, tmp, result);
        
        for (int64_t j = 0; i + j < n; j++) {
            dst[i + j] = (uint8_t)tmp[j];
        }
    }
}

// FilterRangeF64IndicesSVE2 - Return indices of values in range
// Writes count of matching indices to *countOut
void filterRangeF64IndicesSve2(
    const double* __restrict__ values,
    int64_t n,
    double minVal,
    double maxVal,
    int32_t* __restrict__ dst,
    int64_t* __restrict__ countOut
) {
    int64_t count = 0;
    int64_t i = 0;
    svbool_t pg64 = svptrue_b64();
    int64_t vl = svcntd();
    
    // Process full vectors
    for (; i + vl <= n; i += vl) {
        svfloat64_t v = svld1_f64(pg64, &values[i]);
        
        // Compare: v >= minVal AND v <= maxVal (inclusive bounds)
        svbool_t cmp_ge = svcmpge_n_f64(pg64, v, minVal);
        svbool_t cmp_le = svcmple_n_f64(pg64, v, maxVal);
        svbool_t in_range = svand_b_z(pg64, cmp_ge, cmp_le);
        
        // Convert predicate to uint64 vector for extraction
        svuint64_t ones = svdup_u64(1);
        svuint64_t zeros = svdup_u64(0);
        svuint64_t result = svsel_u64(in_range, ones, zeros);
        
        uint64_t tmp[32];
        svst1_u64(pg64, tmp, result);
        
        for (int64_t j = 0; j < vl; j++) {
            if (tmp[j]) {
                dst[count++] = (int32_t)(i + j);
            }
        }
    }
    
    // Handle remainder
    if (i < n) {
        svbool_t pg = svwhilelt_b64(i, n);
        svfloat64_t v = svld1_f64(pg, &values[i]);
        
        svbool_t cmp_ge = svcmpge_n_f64(pg, v, minVal);
        svbool_t cmp_le = svcmple_n_f64(pg, v, maxVal);
        svbool_t in_range = svand_b_z(pg, cmp_ge, cmp_le);
        
        svuint64_t ones = svdup_u64(1);
        svuint64_t zeros = svdup_u64(0);
        svuint64_t result = svsel_u64(in_range, ones, zeros);
        
        uint64_t tmp[32];
        svst1_u64(pg, tmp, result);
        
        for (int64_t j = 0; i + j < n; j++) {
            if (tmp[j]) {
                dst[count++] = (int32_t)(i + j);
            }
        }
    }
    
    *countOut = count;
}

// CountRangeF64SVE2 - Count values in range [minVal, maxVal]
// Writes count to *countOut
void countRangeF64Sve2(
    const double* __restrict__ values,
    int64_t n,
    double minVal,
    double maxVal,
    int64_t* __restrict__ countOut
) {
    int64_t count = 0;
    int64_t i = 0;
    svbool_t pg64 = svptrue_b64();
    int64_t vl = svcntd();
    
    for (; i + vl <= n; i += vl) {
        svfloat64_t v = svld1_f64(pg64, &values[i]);
        
        svbool_t cmp_ge = svcmpge_n_f64(pg64, v, minVal);
        svbool_t cmp_le = svcmple_n_f64(pg64, v, maxVal);
        svbool_t in_range = svand_b_z(pg64, cmp_ge, cmp_le);
        
        // Count active predicate bits
        count += svcntp_b64(pg64, in_range);
    }
    
    // Handle remainder with predicated load
    if (i < n) {
        svbool_t pg = svwhilelt_b64(i, n);
        svfloat64_t v = svld1_f64(pg, &values[i]);
        
        svbool_t cmp_ge = svcmpge_n_f64(pg, v, minVal);
        svbool_t cmp_le = svcmple_n_f64(pg, v, maxVal);
        svbool_t in_range = svand_b_z(pg, cmp_ge, cmp_le);
        
        count += svcntp_b64(pg, in_range);
    }
    
    *countOut = count;
}

// GatherU32SVE2 - Gather uint32 values at specified indices
// SVE2 has native gather support
void gatherU32Sve2(
    const uint32_t* __restrict__ src,
    const int32_t* __restrict__ indices,
    int64_t n,
    uint32_t* __restrict__ dst
) {
    int64_t i = 0;
    svbool_t pg32 = svptrue_b32();
    int64_t vl = svcntw();  // Vector length for 32-bit elements
    
    for (; i + vl <= n; i += vl) {
        // Load indices
        svint32_t idx = svld1_s32(pg32, &indices[i]);
        
        // Gather from source using indices
        svuint32_t gathered = svld1_gather_s32index_u32(pg32, src, idx);
        
        // Store results
        svst1_u32(pg32, &dst[i], gathered);
    }
    
    // Handle remainder
    if (i < n) {
        svbool_t pg = svwhilelt_b32(i, n);
        svint32_t idx = svld1_s32(pg, &indices[i]);
        svuint32_t gathered = svld1_gather_s32index_u32(pg, src, idx);
        svst1_u32(pg, &dst[i], gathered);
    }
}
