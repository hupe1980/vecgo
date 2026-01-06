#include <immintrin.h>
#include <stdint.h>

long long popcountAvx512(const unsigned char *a, int64_t n) {
    long long result = 0;
    int64_t i = 0;
    __m512i acc = _mm512_setzero_si512();

    for (; i <= n - 64; i += 64) {
        __m512i va = _mm512_loadu_si512((const void*)(a + i));
        __m512i p = _mm512_popcnt_epi64(va);
        acc = _mm512_add_epi64(acc, p);
    }

    result = _mm512_reduce_add_epi64(acc);

#ifndef SIMD_NO_TAIL
    for (; i < n; i++) {
        result += __builtin_popcount(a[i]);
    }
#endif
    return result;
}

long long hammingAvx512(const unsigned char *a, const unsigned char *b, int64_t n) {
    long long result = 0;
    int64_t i = 0;
    __m512i acc = _mm512_setzero_si512();

    for (; i <= n - 64; i += 64) {
        __m512i va = _mm512_loadu_si512((const void*)(a + i));
        __m512i vb = _mm512_loadu_si512((const void*)(b + i));
        __m512i x = _mm512_xor_si512(va, vb);
        __m512i p = _mm512_popcnt_epi64(x);
        acc = _mm512_add_epi64(acc, p);
    }

    result = _mm512_reduce_add_epi64(acc);

#ifndef SIMD_NO_TAIL
    for (; i < n; i++) {
        result += __builtin_popcount(a[i] ^ b[i]);
    }
#endif
    return result;
}
