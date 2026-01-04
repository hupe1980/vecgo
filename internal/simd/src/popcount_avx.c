#include <immintrin.h>
#include <stdint.h>

long long popcountAvx(const unsigned char *a, int64_t n) {
    long long result = 0;
    int64_t i = 0;
    const __m256i lookup = _mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4
    );
    const __m256i low_mask = _mm256_set1_epi8(0x0F);
    __m256i acc = _mm256_setzero_si256();

    for (; i <= n - 32; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i lo = _mm256_and_si256(va, low_mask);
        __m256i hi = _mm256_and_si256(_mm256_srli_epi16(va, 4), low_mask);
        __m256i pop = _mm256_add_epi8(_mm256_shuffle_epi8(lookup, lo), _mm256_shuffle_epi8(lookup, hi));
        acc = _mm256_add_epi64(acc, _mm256_sad_epu8(pop, _mm256_setzero_si256()));
    }

    result += _mm256_extract_epi64(acc, 0);
    result += _mm256_extract_epi64(acc, 1);
    result += _mm256_extract_epi64(acc, 2);
    result += _mm256_extract_epi64(acc, 3);

    for (; i < n; i++) {
        result += __builtin_popcount(a[i]);
    }
    return result;
}

long long hammingAvx(const unsigned char *a, const unsigned char *b, int64_t n) {
    long long result = 0;
    int64_t i = 0;
    const __m256i lookup = _mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4
    );
    const __m256i low_mask = _mm256_set1_epi8(0x0F);
    __m256i acc = _mm256_setzero_si256();

    for (; i <= n - 32; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));
        __m256i x = _mm256_xor_si256(va, vb);

        __m256i lo = _mm256_and_si256(x, low_mask);
        __m256i hi = _mm256_and_si256(_mm256_srli_epi16(x, 4), low_mask);
        __m256i pop = _mm256_add_epi8(_mm256_shuffle_epi8(lookup, lo), _mm256_shuffle_epi8(lookup, hi));
        acc = _mm256_add_epi64(acc, _mm256_sad_epu8(pop, _mm256_setzero_si256()));
    }

    result += _mm256_extract_epi64(acc, 0);
    result += _mm256_extract_epi64(acc, 1);
    result += _mm256_extract_epi64(acc, 2);
    result += _mm256_extract_epi64(acc, 3);

    for (; i < n; i++) {
        result += __builtin_popcount(a[i] ^ b[i]);
    }
    return result;
}
