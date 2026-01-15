// Package simd provides SIMD-optimized vector operations.
//
// # Supported Platforms
//
//   - x86-64: AVX-512, AVX2 (falls back to generic Go for AVX1-only CPUs)
//   - ARM64: NEON, SVE2
//
// Runtime CPU feature detection selects the optimal implementation.
// Build with -tags noasm to force the generic Go fallback.
//
// # Operations
//
//   - Distance: Dot, SquaredL2, Hamming
//   - Batch: DotBatch, SquaredL2Batch
//   - Quantized: PQ ADC lookup, SQ8 L2, INT4 L2
//   - Utility: ScaleInPlace
//
// # Performance
//
// Run benchmarks with `go test ./internal/simd -bench . -benchmem` to measure speedup.
// Typical results show 4-8x improvement over generic Go for float32 operations.
package simd
