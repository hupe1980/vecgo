#ifdef __ARM_FEATURE_SVE2
#include <arm_sve.h>
#include <stdint.h>

// popcount_sve2.c - SVE2 optimized popcount and hamming distance operations
//
// SVE2 provides efficient bit manipulation operations including
// population count (number of set bits) and bitwise operations.

// popcountSve2 counts the total number of set bits in a byte array.
long long popcountSve2(const unsigned char *a, int64_t n) {
    svuint64_t acc = svdup_u64(0);
    int64_t i = 0;
    
    svbool_t pg8 = svptrue_b8();
    svbool_t pg64 = svptrue_b64();
    int64_t vl8 = svcntb();  // Number of bytes per vector
    
    // Process full SVE vectors
    for (; i + vl8 <= n; i += vl8) {
        // Load bytes
        svuint8_t va = svld1_u8(pg8, a + i);
        
        // Count bits per byte using SVE2 popcount
        svuint8_t cnt = svcnt_u8_x(pg8, va);
        
        // Use horizontal add to sum the popcount results
        // This avoids the complex widening ladder
        uint64_t chunk_sum = svaddv_u8(pg8, cnt);
        acc = svadd_u64_x(pg64, acc, svdup_u64(chunk_sum));
    }
    
    // Horizontal sum of accumulator
    long long result = svaddv_u64(pg64, acc);
    
    // Handle tail
    for (; i < n; i++) {
        result += __builtin_popcount(a[i]);
    }
    
    return result;
}

// hammingSve2 computes Hamming distance (number of differing bits) between two byte arrays.
long long hammingSve2(const unsigned char *a, const unsigned char *b, int64_t n) {
    svuint64_t acc = svdup_u64(0);
    int64_t i = 0;
    
    svbool_t pg8 = svptrue_b8();
    svbool_t pg64 = svptrue_b64();
    int64_t vl8 = svcntb();
    
    // Process full SVE vectors
    for (; i + vl8 <= n; i += vl8) {
        // Load bytes from both arrays
        svuint8_t va = svld1_u8(pg8, a + i);
        svuint8_t vb = svld1_u8(pg8, b + i);
        
        // XOR to find differing bits
        svuint8_t xored = sveor_u8_x(pg8, va, vb);
        
        // Count bits per byte
        svuint8_t cnt = svcnt_u8_x(pg8, xored);
        
        // Use horizontal add to sum the popcount results
        uint64_t chunk_sum = svaddv_u8(pg8, cnt);
        acc = svadd_u64_x(pg64, acc, svdup_u64(chunk_sum));
    }
    
    // Horizontal sum
    long long result = svaddv_u64(pg64, acc);
    
    // Handle tail
    for (; i < n; i++) {
        result += __builtin_popcount(a[i] ^ b[i]);
    }
    
    return result;
}

// jaccardSve2 computes Jaccard distance based on binary vectors.
// Jaccard distance = 1 - (intersection / union) = 1 - (popcount(a & b) / popcount(a | b))
void jaccardSve2(const unsigned char *a, const unsigned char *b, int64_t n, float *result) {
    svuint64_t acc_and = svdup_u64(0);
    svuint64_t acc_or = svdup_u64(0);
    int64_t i = 0;
    
    svbool_t pg8 = svptrue_b8();
    svbool_t pg64 = svptrue_b64();
    int64_t vl8 = svcntb();
    
    // Process full SVE vectors
    for (; i + vl8 <= n; i += vl8) {
        svuint8_t va = svld1_u8(pg8, a + i);
        svuint8_t vb = svld1_u8(pg8, b + i);
        
        // AND and OR
        svuint8_t v_and = svand_u8_x(pg8, va, vb);
        svuint8_t v_or = svorr_u8_x(pg8, va, vb);
        
        // Count bits
        svuint8_t cnt_and = svcnt_u8_x(pg8, v_and);
        svuint8_t cnt_or = svcnt_u8_x(pg8, v_or);
        
        // Sum the popcount results using horizontal adds
        uint64_t and_sum = svaddv_u8(pg8, cnt_and);
        uint64_t or_sum = svaddv_u8(pg8, cnt_or);
        acc_and = svadd_u64_x(pg64, acc_and, svdup_u64(and_sum));
        acc_or = svadd_u64_x(pg64, acc_or, svdup_u64(or_sum));
    }
    
    // Horizontal sums
    long long intersection = svaddv_u64(pg64, acc_and);
    long long union_val = svaddv_u64(pg64, acc_or);
    
    // Handle tail
    for (; i < n; i++) {
        intersection += __builtin_popcount(a[i] & b[i]);
        union_val += __builtin_popcount(a[i] | b[i]);
    }
    
    // Compute Jaccard distance
    if (union_val > 0) {
        *result = 1.0f - (float)intersection / (float)union_val;
    } else {
        *result = 0.0f;  // Both vectors are zero
    }
}

#endif // __ARM_FEATURE_SVE2
