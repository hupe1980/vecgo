#ifdef __ARM_FEATURE_SVE2
#include <arm_sve.h>
#include <stdint.h>

// f16_sve2.c - SVE2 optimized float16 operations
//
// SVE2 provides native float16 support with hardware conversion to float32.

// f16ToF32Sve2 converts a batch of float16 values to float32 using SVE2.
// in: pointer to the input array (n uint16/float16 values)
// out: pointer to the output array (n float32 values)
// n: number of elements
void f16ToF32Sve2(const uint16_t *__restrict__ in, float *__restrict__ out, int64_t n) {
    int64_t i = 0;
    svbool_t pg16 = svptrue_b16();
    svbool_t pg32 = svptrue_b32();
    int64_t vl16 = svcnth();  // Number of 16-bit elements per vector
    int64_t vl32 = svcntw();  // Number of 32-bit elements per vector
    
    // Process vl32 elements at a time (since output is float32)
    for (; i + vl32 <= n; i += vl32) {
        // Load float16 values as uint16
        svfloat16_t v_f16 = svreinterpret_f16_u16(svld1_u16(pg16, in + i));
        
        // Convert low half to float32
        svfloat32_t v_f32_lo = svcvt_f32_f16_x(pg32, v_f16);
        
        // Store result
        svst1_f32(pg32, out + i, v_f32_lo);
    }
    
    // Handle tail with predication
    if (i < n) {
        svbool_t pg_tail16 = svwhilelt_b16_s64(i, n);
        svbool_t pg_tail32 = svwhilelt_b32_s64(i, n);
        
        svfloat16_t v_f16 = svreinterpret_f16_u16(svld1_u16(pg_tail16, in + i));
        svfloat32_t v_f32 = svcvt_f32_f16_x(pg_tail32, v_f16);
        svst1_f32(pg_tail32, out + i, v_f32);
    }
}

// f32ToF16Sve2 converts a batch of float32 values to float16 using SVE2.
// in: pointer to the input array (n float32 values)
// out: pointer to the output array (n uint16/float16 values)
// n: number of elements
void f32ToF16Sve2(const float *__restrict__ in, uint16_t *__restrict__ out, int64_t n) {
    int64_t i = 0;
    svbool_t pg32 = svptrue_b32();
    int64_t vl32 = svcntw();
    
    // Process vl32 elements at a time
    for (; i + vl32 <= n; i += vl32) {
        // Load float32 values
        svfloat32_t v_f32 = svld1_f32(pg32, in + i);
        
        // Convert to float16
        svfloat16_t v_f16 = svcvt_f16_f32_x(pg32, v_f32);
        
        // Store as uint16
        svst1_u16(pg32, out + i, svreinterpret_u16_f16(v_f16));
    }
    
    // Handle tail with predication
    if (i < n) {
        svbool_t pg_tail = svwhilelt_b32_s64(i, n);
        
        svfloat32_t v_f32 = svld1_f32(pg_tail, in + i);
        svfloat16_t v_f16 = svcvt_f16_f32_x(pg_tail, v_f32);
        svst1_u16(pg_tail, out + i, svreinterpret_u16_f16(v_f16));
    }
}

// f16DotProductSve2 computes dot product of two float16 vectors.
// Values are converted to float32 for accumulation, then result is float32.
void f16DotProductSve2(const uint16_t *__restrict__ a, const uint16_t *__restrict__ b, 
                        int64_t n, float *__restrict__ result) {
    svfloat32_t sum = svdup_f32(0.0f);
    int64_t i = 0;
    
    svbool_t pg16 = svptrue_b16();
    svbool_t pg32 = svptrue_b32();
    int64_t vl32 = svcntw();
    
    // Process vl32 elements at a time
    for (; i + vl32 <= n; i += vl32) {
        // Load float16 values
        svfloat16_t va_f16 = svreinterpret_f16_u16(svld1_u16(pg16, a + i));
        svfloat16_t vb_f16 = svreinterpret_f16_u16(svld1_u16(pg16, b + i));
        
        // Convert to float32
        svfloat32_t va_f32 = svcvt_f32_f16_x(pg32, va_f16);
        svfloat32_t vb_f32 = svcvt_f32_f16_x(pg32, vb_f16);
        
        // Multiply and accumulate
        sum = svmla_f32_x(pg32, sum, va_f32, vb_f32);
    }
    
    // Handle tail with predication
    if (i < n) {
        svbool_t pg_tail16 = svwhilelt_b16_s64(i, n);
        svbool_t pg_tail32 = svwhilelt_b32_s64(i, n);
        
        svfloat16_t va_f16 = svreinterpret_f16_u16(svld1_u16(pg_tail16, a + i));
        svfloat16_t vb_f16 = svreinterpret_f16_u16(svld1_u16(pg_tail16, b + i));
        
        svfloat32_t va_f32 = svcvt_f32_f16_x(pg_tail32, va_f16);
        svfloat32_t vb_f32 = svcvt_f32_f16_x(pg_tail32, vb_f16);
        
        sum = svmla_f32_m(pg_tail32, sum, va_f32, vb_f32);
    }
    
    *result = svaddv_f32(pg32, sum);
}

// f16SquaredL2Sve2 computes squared L2 distance between two float16 vectors.
void f16SquaredL2Sve2(const uint16_t *__restrict__ a, const uint16_t *__restrict__ b,
                       int64_t n, float *__restrict__ result) {
    svfloat32_t sum = svdup_f32(0.0f);
    int64_t i = 0;
    
    svbool_t pg16 = svptrue_b16();
    svbool_t pg32 = svptrue_b32();
    int64_t vl32 = svcntw();
    
    // Process vl32 elements at a time
    for (; i + vl32 <= n; i += vl32) {
        // Load float16 values
        svfloat16_t va_f16 = svreinterpret_f16_u16(svld1_u16(pg16, a + i));
        svfloat16_t vb_f16 = svreinterpret_f16_u16(svld1_u16(pg16, b + i));
        
        // Convert to float32
        svfloat32_t va_f32 = svcvt_f32_f16_x(pg32, va_f16);
        svfloat32_t vb_f32 = svcvt_f32_f16_x(pg32, vb_f16);
        
        // Compute diff and accumulate squared
        svfloat32_t diff = svsub_f32_x(pg32, va_f32, vb_f32);
        sum = svmla_f32_x(pg32, sum, diff, diff);
    }
    
    // Handle tail with predication
    if (i < n) {
        svbool_t pg_tail16 = svwhilelt_b16_s64(i, n);
        svbool_t pg_tail32 = svwhilelt_b32_s64(i, n);
        
        svfloat16_t va_f16 = svreinterpret_f16_u16(svld1_u16(pg_tail16, a + i));
        svfloat16_t vb_f16 = svreinterpret_f16_u16(svld1_u16(pg_tail16, b + i));
        
        svfloat32_t va_f32 = svcvt_f32_f16_x(pg_tail32, va_f16);
        svfloat32_t vb_f32 = svcvt_f32_f16_x(pg_tail32, vb_f16);
        
        svfloat32_t diff = svsub_f32_x(pg_tail32, va_f32, vb_f32);
        sum = svmla_f32_m(pg_tail32, sum, diff, diff);
    }
    
    *result = svaddv_f32(pg32, sum);
}

#endif // __ARM_FEATURE_SVE2
