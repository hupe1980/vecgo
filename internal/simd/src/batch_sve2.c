#ifdef __ARM_FEATURE_SVE2
#include <arm_sve.h>
#include <stdint.h>

// batch_sve2.c - SVE2 optimized batch distance operations
//
// Computes distances between a query vector and multiple target vectors
// in a single pass for better memory locality and vectorization.

// squaredL2BatchSve2 computes squared L2 distance for a batch of vectors using SVE2.
// query: pointer to the query vector (dim floats)
// targets: pointer to the flattened target vectors (n * dim floats)
// dim: dimension of the vectors
// n: number of target vectors
// out: pointer to the output array (n floats)
void squaredL2BatchSve2(float *__restrict__ query, float *__restrict__ targets,
                         int64_t dim, int64_t n, float *__restrict__ out) {
    svbool_t pg = svptrue_b32();
    int64_t vl = svcntw();
    
    for (int64_t i = 0; i < n; i++) {
        float *target = targets + i * dim;
        svfloat32_t sum = svdup_f32(0.0f);
        
        int64_t j = 0;
        // Main loop - process full SVE vectors
        for (; j + vl <= dim; j += vl) {
            svfloat32_t vq = svld1_f32(pg, query + j);
            svfloat32_t vt = svld1_f32(pg, target + j);
            svfloat32_t diff = svsub_f32_x(pg, vq, vt);
            sum = svmla_f32_x(pg, sum, diff, diff);
        }
        
        // Handle tail with predication
        if (j < dim) {
            svbool_t pg_tail = svwhilelt_b32_s64(j, dim);
            svfloat32_t vq = svld1_f32(pg_tail, query + j);
            svfloat32_t vt = svld1_f32(pg_tail, target + j);
            svfloat32_t diff = svsub_f32_x(pg_tail, vq, vt);
            sum = svmla_f32_m(pg_tail, sum, diff, diff);
        }
        
        out[i] = svaddv_f32(pg, sum);
    }
}

// dotBatchSve2 computes dot product for a batch of vectors using SVE2.
void dotBatchSve2(float *__restrict__ query, float *__restrict__ targets,
                   int64_t dim, int64_t n, float *__restrict__ out) {
    svbool_t pg = svptrue_b32();
    int64_t vl = svcntw();
    
    for (int64_t i = 0; i < n; i++) {
        float *target = targets + i * dim;
        svfloat32_t sum = svdup_f32(0.0f);
        
        int64_t j = 0;
        // Main loop - process full SVE vectors
        for (; j + vl <= dim; j += vl) {
            svfloat32_t vq = svld1_f32(pg, query + j);
            svfloat32_t vt = svld1_f32(pg, target + j);
            sum = svmla_f32_x(pg, sum, vq, vt);
        }
        
        // Handle tail with predication
        if (j < dim) {
            svbool_t pg_tail = svwhilelt_b32_s64(j, dim);
            svfloat32_t vq = svld1_f32(pg_tail, query + j);
            svfloat32_t vt = svld1_f32(pg_tail, target + j);
            sum = svmla_f32_m(pg_tail, sum, vq, vt);
        }
        
        out[i] = svaddv_f32(pg, sum);
    }
}

// cosineBatchSve2 computes cosine similarity for a batch of vectors using SVE2.
// Assumes query is pre-normalized. Returns 1 - dot(q, t) / ||t|| for distance.
void cosineBatchSve2(float *__restrict__ query, float *__restrict__ targets,
                      int64_t dim, int64_t n, float *__restrict__ out) {
    svbool_t pg = svptrue_b32();
    int64_t vl = svcntw();
    
    for (int64_t i = 0; i < n; i++) {
        float *target = targets + i * dim;
        svfloat32_t dot_sum = svdup_f32(0.0f);
        svfloat32_t norm_sum = svdup_f32(0.0f);
        
        int64_t j = 0;
        // Main loop
        for (; j + vl <= dim; j += vl) {
            svfloat32_t vq = svld1_f32(pg, query + j);
            svfloat32_t vt = svld1_f32(pg, target + j);
            
            // Dot product
            dot_sum = svmla_f32_x(pg, dot_sum, vq, vt);
            // Target norm squared
            norm_sum = svmla_f32_x(pg, norm_sum, vt, vt);
        }
        
        // Handle tail with predication
        if (j < dim) {
            svbool_t pg_tail = svwhilelt_b32_s64(j, dim);
            svfloat32_t vq = svld1_f32(pg_tail, query + j);
            svfloat32_t vt = svld1_f32(pg_tail, target + j);
            
            dot_sum = svmla_f32_m(pg_tail, dot_sum, vq, vt);
            norm_sum = svmla_f32_m(pg_tail, norm_sum, vt, vt);
        }
        
        float dot = svaddv_f32(pg, dot_sum);
        float norm = svaddv_f32(pg, norm_sum);
        
        // Cosine distance = 1 - cosine_similarity
        // Handle zero norm case
        if (norm > 0.0f) {
            // Use SVE rsqrt estimate + Newton-Raphson refinement to avoid sqrtf relocation
            svfloat32_t v_norm = svdup_f32(norm);
            svfloat32_t rsqrt0 = svrsqrte_f32(v_norm);
            // One NR iteration: rsqrt1 = rsqrt0 * (3 - norm * rsqrt0 * rsqrt0) / 2
            svfloat32_t step = svrsqrts_f32(svmul_f32_x(pg, v_norm, rsqrt0), rsqrt0);
            svfloat32_t rsqrt1 = svmul_f32_x(pg, rsqrt0, step);
            float inv_norm = svlastb_f32(pg, rsqrt1);
            out[i] = 1.0f - dot * inv_norm;
        } else {
            out[i] = 1.0f;  // Maximum distance for zero vectors
        }
    }
}

#endif // __ARM_FEATURE_SVE2
