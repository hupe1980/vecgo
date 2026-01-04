#include <immintrin.h>
#include <stdint.h>

// F16ToF32Avx converts a batch of float16 values to float32 using AVX2/F16C.
// in: pointer to the input array (n uint16/float16 values)
// out: pointer to the output array (n float32 values)
// n: number of elements
void f16ToF32Avx(const uint16_t *__restrict__ in, float *__restrict__ out, int64_t n) {
    int64_t i = 0;
    // Process 8 elements per iteration (128-bit load -> 256-bit store)
    for (; i <= n - 8; i += 8) {
        __m128i v_fp16 = _mm_loadu_si128((__m128i const*)(in + i));
        __m256 v_fp32 = _mm256_cvtph_ps(v_fp16);
        _mm256_storeu_ps(out + i, v_fp32);
    }

    // Handle remaining elements
    for (; i < n; i++) {
        // Scalar fallback (using intrinsic for single conversion if needed, or just loop)
        // Since we are in C for ASM generation, we can use the intrinsic for scalar too
        // or just let the compiler handle it. 
        // _cvtsh_ss is for scalar half->float but requires SSE2/F16C.
        // For simplicity in C reference, we can use the vector instruction on a partial load
        // or just cast if the compiler supports _Float16.
        // But standard C doesn't always have _Float16.
        // Let's use the intrinsic on a zero-padded vector.
        uint16_t temp = in[i];
        __m128i v_fp16 = _mm_set1_epi16(temp); // Broadcast
        __m256 v_fp32 = _mm256_cvtph_ps(v_fp16);
        float f;
        _mm_store_ss(&f, _mm256_castps256_ps128(v_fp32));
        out[i] = f;
    }
}
