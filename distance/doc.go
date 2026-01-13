// Package distance provides vector distance calculations with SIMD acceleration.
//
// All distance functions use SIMD-optimized implementations when available:
//   - AVX-512/AVX2 on x86-64
//   - NEON/SVE2 on ARM64
//
// # Supported Metrics
//
//   - MetricL2: Squared Euclidean distance (default)
//   - MetricCosine: Cosine similarity (normalized dot product)
//   - MetricDot: Dot product (inner product)
//
// # Usage
//
//	dist := distance.SquaredL2(a, b)
//	sim := distance.Dot(a, b)
//	normalized := distance.Normalize(vec)
package distance
