#include <immintrin.h>
#include <stdint.h>

// F16ToF32Avx512 converts a batch of float16 values to float32 using AVX-512.
// in: pointer to the input array (n uint16/float16 values)
// out: pointer to the output array (n float32 values)
// n: number of elements
void f16ToF32Avx512(const uint16_t *__restrict__ in, float *__restrict__ out, int64_t n) {
    int64_t i = 0;
    // Process 16 elements per iteration (256-bit load -> 512-bit store)
    for (; i <= n - 16; i += 16) {
        __m256i v_fp16 = _mm256_loadu_si256((__m256i const*)(in + i));
        __m512 v_fp32 = _mm512_cvtph_ps(v_fp16);
        _mm512_storeu_ps(out + i, v_fp32);
    }

    // Handle remaining elements using masking
    if (i < n) {
        int64_t remain = n - i;
        __mmask16 mask = (1 << remain) - 1;
        __m256i v_fp16 = _mm256_maskz_loadu_epi16(mask, in + i);
        __m512 v_fp32 = _mm512_cvtph_ps(v_fp16);
        _mm512_mask_storeu_ps(out + i, mask, v_fp32);
    }
}
