#include <arm_neon.h>
#include <stdint.h>

// F16ToF32Neon converts a batch of float16 values to float32 using NEON.
// in: pointer to the input array (n uint16/float16 values)
// out: pointer to the output array (n float32 values)
// n: number of elements
void f16ToF32Neon(const uint16_t *__restrict__ in, float *__restrict__ out, int64_t n) {
    int64_t i = 0;
    // Process 8 elements per iteration
    for (; i <= n - 8; i += 8) {
        // Load 8 float16 values (128 bits)
        // Note: vld1q_f16 requires __fp16 support or casting
        float16x8_t v_fp16 = vld1q_f16((const float16_t*)(in + i));
        
        // Convert low 4 elements
        float32x4_t v_low = vcvt_f32_f16(vget_low_f16(v_fp16));
        // Convert high 4 elements
        float32x4_t v_high = vcvt_f32_f16(vget_high_f16(v_fp16));
        
        vst1q_f32(out + i, v_low);
        vst1q_f32(out + i + 4, v_high);
    }

    // Handle remaining elements
    for (; i < n; i++) {
        // Scalar conversion
        // Assuming compiler supports __fp16 or we do bit manipulation
        // For C reference, we can use the intrinsic for single element
        // or just cast if supported.
        // vcvt_f32_f16 takes float16x4_t.
        uint16_t temp = in[i];
        float16x4_t v_fp16 = vset_lane_u16(temp, vdup_n_u16(0), 0); // Bit cast hack if needed
        // Actually, simpler:
        __fp16 h = *(__fp16*)&temp;
        out[i] = (float)h;
    }
}
