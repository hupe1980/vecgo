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
    
    // Get the number of double elements that fit in a vector
    int64_t vl = svcntd();  // Vector length for 64-bit elements
    
    while (i < n) {
        // Create predicate for active lanes (handles tail)
        svbool_t pg = svwhilelt_b64(i, n);
        
        // Load values
        svfloat64_t v = svld1_f64(pg, &values[i]);
        
        // Compare: v >= minVal AND v <= maxVal
        svbool_t cmp_ge = svcmpge_n_f64(pg, v, minVal);  // v >= min
        svbool_t cmp_le = svcmple_n_f64(pg, v, maxVal);  // v <= max
        svbool_t in_range = svand_b_z(pg, cmp_ge, cmp_le);
        
        // Convert predicate to 0/1 bytes and store
        // SVE2 doesn't have direct predicate-to-byte, so we use select
        svuint8_t zeros = svdup_n_u8(0);
        svuint8_t ones = svdup_n_u8(1);
        
        // We need to handle this differently - extract predicate bits
        // For each active lane, store 1 if in_range, else 0
        for (int64_t j = 0; j < vl && (i + j) < n; j++) {
            dst[i + j] = svptest_first(svptrue_b64(), 
                svand_b_z(svptrue_b64(), 
                    in_range, 
                    svwhilelt_b64(j, j + 1))) ? 1 : 0;
        }
        
        i += vl;
    }
}

// FilterRangeF64SVE2Simple - Simpler version using lane-by-lane extraction
void filterRangeF64Sve2Simple(
    const double* __restrict__ values,
    int64_t n,
    double minVal,
    double maxVal,
    uint8_t* __restrict__ dst
) {
    int64_t i = 0;
    int64_t vl = svcntd();
    
    // Process vector-length elements at a time
    for (; i + vl <= n; i += vl) {
        svbool_t pg = svptrue_b64();
        svfloat64_t v = svld1_f64(pg, &values[i]);
        
        svbool_t cmp_ge = svcmpge_n_f64(pg, v, minVal);
        svbool_t cmp_le = svcmple_n_f64(pg, v, maxVal);
        svbool_t in_range = svand_b_z(pg, cmp_ge, cmp_le);
        
        // Use svsel to create 0/1 vector, then compact store
        svuint64_t result = svsel_u64(in_range, svdup_n_u64(1), svdup_n_u64(0));
        
        // Store results (narrowing from u64 to u8)
        for (int64_t j = 0; j < vl; j++) {
            dst[i + j] = (uint8_t)svlasta_u64(svwhilelt_b64(j, j + 1), result);
        }
    }
    
    // Handle remainder with scalar
    for (; i < n; i++) {
        dst[i] = (values[i] >= minVal && values[i] <= maxVal) ? 1 : 0;
    }
}

// FilterRangeF64IndicesSVE2 - Return indices of values in range
// Uses SVE2 compact operations for efficient index extraction
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
    int64_t vl = svcntd();
    
    for (; i + vl <= n; i += vl) {
        svbool_t pg = svptrue_b64();
        svfloat64_t v = svld1_f64(pg, &values[i]);
        
        svbool_t cmp_ge = svcmpge_n_f64(pg, v, minVal);
        svbool_t cmp_le = svcmple_n_f64(pg, v, maxVal);
        svbool_t in_range = svand_b_z(pg, cmp_ge, cmp_le);
        
        // Create index vector and compact store matching indices
        svint64_t indices = svindex_s64(i, 1);  // [i, i+1, i+2, ...]
        
        // Compact store only matching elements
        svst1_s64(in_range, (int64_t*)&dst[count], indices);
        
        // Count matching elements
        count += svcntp_b64(pg, in_range);
    }
    
    // Handle remainder
    for (; i < n; i++) {
        if (values[i] >= minVal && values[i] <= maxVal) {
            dst[count++] = (int32_t)i;
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
    int64_t vl = svcntd();
    
    for (; i + vl <= n; i += vl) {
        svbool_t pg = svptrue_b64();
        svfloat64_t v = svld1_f64(pg, &values[i]);
        
        svbool_t cmp_ge = svcmpge_n_f64(pg, v, minVal);
        svbool_t cmp_le = svcmple_n_f64(pg, v, maxVal);
        svbool_t in_range = svand_b_z(pg, cmp_ge, cmp_le);
        
        // Count active predicate bits
        count += svcntp_b64(pg, in_range);
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
    int64_t vl = svcntw();  // Vector length for 32-bit elements
    
    for (; i + vl <= n; i += vl) {
        svbool_t pg = svptrue_b32();
        
        // Load indices
        svint32_t idx = svld1_s32(pg, &indices[i]);
        
        // Gather from source using indices
        // SVE2 gather: svld1_gather_s32index_u32
        svuint32_t gathered = svld1_gather_s32index_u32(pg, src, idx);
        
        // Store results
        svst1_u32(pg, &dst[i], gathered);
    }
    
    // Handle remainder
    if (i < n) {
        svbool_t pg = svwhilelt_b32(i, n);
        svint32_t idx = svld1_s32(pg, &indices[i]);
        svuint32_t gathered = svld1_gather_s32index_u32(pg, src, idx);
        svst1_u32(pg, &dst[i], gathered);
    }
}
