// SIMD filter operations using ARM NEON for numeric indexing
// Compile with: clang -O3 -march=armv8-a+simd -S -o filter_neon.s filter_neon.c

#include <arm_neon.h>
#include <stdint.h>

// FilterRangeF64NEON - Check if values are in range [minVal, maxVal]
// Processes 2 float64 values at a time using NEON (128-bit)
//
// Arguments:
//   values: pointer to float64 array
//   n: number of elements
//   minVal: minimum value (inclusive)
//   maxVal: maximum value (inclusive)
//   dst: output byte array (1 = in range, 0 = out of range)
void filterRangeF64Neon(
    const double* __restrict__ values,
    int64_t n,
    double minVal,
    double maxVal,
    uint8_t* __restrict__ dst
) {
    float64x2_t vmin = vdupq_n_f64(minVal);
    float64x2_t vmax = vdupq_n_f64(maxVal);
    
    // Constant for masking to 0 or 1
    uint8x8_t one = vdup_n_u8(1);
    
    int64_t i = 0;
    
    // Process 8 doubles at a time for better throughput
    for (; i + 8 <= n; i += 8) {
        float64x2_t v0 = vld1q_f64(&values[i]);
        float64x2_t v1 = vld1q_f64(&values[i + 2]);
        float64x2_t v2 = vld1q_f64(&values[i + 4]);
        float64x2_t v3 = vld1q_f64(&values[i + 6]);
        
        // Compare all vectors: result is all 1s or all 0s per lane
        uint64x2_t r0 = vandq_u64(vcgeq_f64(v0, vmin), vcleq_f64(v0, vmax));
        uint64x2_t r1 = vandq_u64(vcgeq_f64(v1, vmin), vcleq_f64(v1, vmax));
        uint64x2_t r2 = vandq_u64(vcgeq_f64(v2, vmin), vcleq_f64(v2, vmax));
        uint64x2_t r3 = vandq_u64(vcgeq_f64(v3, vmin), vcleq_f64(v3, vmax));
        
        // Narrow 64-bit to 32-bit (takes lower 32 bits of each 64-bit lane)
        uint32x2_t n0 = vmovn_u64(r0);
        uint32x2_t n1 = vmovn_u64(r1);
        uint32x2_t n2 = vmovn_u64(r2);
        uint32x2_t n3 = vmovn_u64(r3);
        
        // Combine into 4x32-bit vectors
        uint32x4_t w01 = vcombine_u32(n0, n1);
        uint32x4_t w23 = vcombine_u32(n2, n3);
        
        // Narrow 32-bit to 16-bit
        uint16x4_t h01 = vmovn_u32(w01);
        uint16x4_t h23 = vmovn_u32(w23);
        
        // Combine and narrow to 8-bit
        uint16x8_t h = vcombine_u16(h01, h23);
        uint8x8_t bytes = vmovn_u16(h);
        
        // AND with 1 to get exactly 0 or 1 (bytes are 0xFF or 0x00)
        bytes = vand_u8(bytes, one);
        
        vst1_u8(&dst[i], bytes);
    }
    
    // Process 2 at a time
    for (; i + 2 <= n; i += 2) {
        float64x2_t v = vld1q_f64(&values[i]);
        uint64x2_t in_range = vandq_u64(vcgeq_f64(v, vmin), vcleq_f64(v, vmax));
        
        dst[i]     = (vgetq_lane_u64(in_range, 0) != 0) ? 1 : 0;
        dst[i + 1] = (vgetq_lane_u64(in_range, 1) != 0) ? 1 : 0;
    }
    
    // Handle remainder
    for (; i < n; i++) {
        dst[i] = (values[i] >= minVal && values[i] <= maxVal) ? 1 : 0;
    }
}

// FilterRangeF64IndicesNEON - Return indices of values in range
// Writes count of matching indices to *countOut
void filterRangeF64IndicesNeon(
    const double* __restrict__ values,
    int64_t n,
    double minVal,
    double maxVal,
    int32_t* __restrict__ dst,
    int64_t* __restrict__ countOut
) {
    float64x2_t vmin = vdupq_n_f64(minVal);
    float64x2_t vmax = vdupq_n_f64(maxVal);
    
    int64_t count = 0;
    int64_t i = 0;
    
    for (; i + 2 <= n; i += 2) {
        float64x2_t v = vld1q_f64(&values[i]);
        
        uint64x2_t cmp_ge = vcgeq_f64(v, vmin);
        uint64x2_t cmp_le = vcleq_f64(v, vmax);
        uint64x2_t in_range = vandq_u64(cmp_ge, cmp_le);
        
        // Extract and store matching indices
        if (vgetq_lane_u64(in_range, 0)) {
            dst[count++] = (int32_t)i;
        }
        if (vgetq_lane_u64(in_range, 1)) {
            dst[count++] = (int32_t)(i + 1);
        }
    }
    
    // Handle remainder
    for (; i < n; i++) {
        if (values[i] >= minVal && values[i] <= maxVal) {
            dst[count++] = (int32_t)i;
        }
    }
    
    *countOut = count;
}

// CountRangeF64NEON - Count values in range [minVal, maxVal]
// Writes count to *countOut
void countRangeF64Neon(
    const double* __restrict__ values,
    int64_t n,
    double minVal,
    double maxVal,
    int64_t* __restrict__ countOut
) {
    float64x2_t vmin = vdupq_n_f64(minVal);
    float64x2_t vmax = vdupq_n_f64(maxVal);
    
    int64_t i = 0;
    
    // Accumulate counts in vector register
    uint64x2_t vcount = vdupq_n_u64(0);
    
    // Process 4 doubles at a time for better ILP
    for (; i + 4 <= n; i += 4) {
        float64x2_t v0 = vld1q_f64(&values[i]);
        float64x2_t v1 = vld1q_f64(&values[i + 2]);
        
        uint64x2_t r0 = vandq_u64(vcgeq_f64(v0, vmin), vcleq_f64(v0, vmax));
        uint64x2_t r1 = vandq_u64(vcgeq_f64(v1, vmin), vcleq_f64(v1, vmax));
        
        // Right shift by 63 to get 0 or 1
        vcount = vaddq_u64(vcount, vshrq_n_u64(r0, 63));
        vcount = vaddq_u64(vcount, vshrq_n_u64(r1, 63));
    }
    
    // Sum the two lanes
    int64_t count = vgetq_lane_u64(vcount, 0) + vgetq_lane_u64(vcount, 1);
    
    // Handle remainder
    for (; i < n; i++) {
        if (values[i] >= minVal && values[i] <= maxVal) {
            count++;
        }
    }
    
    *countOut = count;
}

// GatherU32NEON - Gather uint32 values at specified indices
// NEON doesn't have native gather, so we use optimized scalar with prefetch
void gatherU32Neon(
    const uint32_t* __restrict__ src,
    const int32_t* __restrict__ indices,
    int64_t n,
    uint32_t* __restrict__ dst
) {
    int64_t i = 0;
    
    // Process 4 at a time with prefetch hints
    for (; i + 4 <= n; i += 4) {
        // Prefetch next batch
        __builtin_prefetch(&indices[i + 8], 0, 0);
        
        int32_t i0 = indices[i];
        int32_t i1 = indices[i + 1];
        int32_t i2 = indices[i + 2];
        int32_t i3 = indices[i + 3];
        
        // Prefetch source data
        __builtin_prefetch(&src[indices[i + 4]], 0, 0);
        
        dst[i]     = src[i0];
        dst[i + 1] = src[i1];
        dst[i + 2] = src[i2];
        dst[i + 3] = src[i3];
    }
    
    // Handle remainder
    for (; i < n; i++) {
        dst[i] = src[indices[i]];
    }
}
