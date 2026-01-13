// Package simd provides SIMD-optimized vector operations.
//
// # Supported Platforms
//
//   - x86-64: AVX-512, AVX2, SSE4.2
//   - ARM64: NEON, SVE2
//
// Runtime CPU feature detection selects the optimal implementation.
// Build with -tags noasm to force the generic Go fallback.
//
// # Operations
//
//   - Distance: Dot, SquaredL2, Cosine
//   - Batch: DotBatch, SquaredL2Batch
//   - Quantized: PQ ADC lookup, SQ8 L2, INT4 L2
//   - Utility: Normalize, F16→F32 conversion
//
// # Performance
//
// AVX-512 provides ~8x speedup over scalar code for float32 operations.
// INT4 SIMD with precomputed lookup tables achieves ~95ns per distance.
package simd
