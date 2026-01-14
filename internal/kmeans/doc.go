// Package kmeans implements k-means clustering for quantization training.
//
// Used internally by Product Quantization (PQ) and Optimized PQ (OPQ)
// to learn codebooks from training data, and by the flat segment writer
// for partitioned IVF-style indexing.
//
// # Features
//
//   - Lloyd's algorithm with early termination on convergence
//   - SIMD-accelerated distance computation (AVX-512/AVX2/NEON/SVE2) for L2 metric
//   - Context support for cancellation of long-running training
//   - Empty cluster re-initialization to avoid degenerate solutions
//   - Adaptive sorting: partial selection for small n, full sort for larger n
//
// # Performance
//
// For the common case of L2 distance, assignment uses SIMD batch distance
// computation (simd.SquaredL2Batch) which processes all centroids in a
// single vectorized operation, yielding ~4-8x speedup over scalar loops.
//
// FindClosestCentroids (used in search hot path) uses partial selection
// for small nprobes values (n <= k/4 && n < 16), avoiding full sort overhead.
package kmeans
