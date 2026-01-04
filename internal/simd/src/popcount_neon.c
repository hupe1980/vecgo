#include <arm_neon.h>
#include <stdint.h>

long long popcountNeon(const unsigned char *a, int64_t n) {
    long long result = 0;
    int64_t i = 0;
    uint16x8_t acc_16 = vdupq_n_u16(0);

    for (; i <= n - 16; i += 16) {
        uint8x16_t va = vld1q_u8(a + i);
        uint8x16_t cnt = vcntq_u8(va);
        acc_16 = vpadalq_u8(acc_16, cnt);
    }

    uint32x4_t acc_32 = vpaddlq_u16(acc_16);
    uint64x2_t acc_64 = vpaddlq_u32(acc_32);
    result = vgetq_lane_u64(acc_64, 0) + vgetq_lane_u64(acc_64, 1);

    for (; i < n; i++) {
        result += __builtin_popcount(a[i]);
    }
    return result;
}

long long hammingNeon(const unsigned char *a, const unsigned char *b, int64_t n) {
    long long result = 0;
    int64_t i = 0;
    uint16x8_t acc_16 = vdupq_n_u16(0);

    for (; i <= n - 16; i += 16) {
        uint8x16_t va = vld1q_u8(a + i);
        uint8x16_t vb = vld1q_u8(b + i);
        uint8x16_t x = veorq_u8(va, vb);
        uint8x16_t cnt = vcntq_u8(x);
        acc_16 = vpadalq_u8(acc_16, cnt);
    }

    uint32x4_t acc_32 = vpaddlq_u16(acc_16);
    uint64x2_t acc_64 = vpaddlq_u32(acc_32);
    result = vgetq_lane_u64(acc_64, 0) + vgetq_lane_u64(acc_64, 1);

    for (; i < n; i++) {
        result += __builtin_popcount(a[i] ^ b[i]);
    }
    return result;
}
