// Package simd provides SIMD-optimized vector operations.
//
// # Supported Platforms
//
//   - x86-64: AVX-512, AVX2 (requires FMA)
//   - ARM64: NEON, SVE2
//
// Runtime CPU feature detection selects the optimal implementation.
// Build with -tags noasm to force the generic Go fallback.
//
// # SIMD Implementation Override
//
// Set VECGO_SIMD environment variable to force a specific implementation:
//
//   - VECGO_SIMD=generic  - Pure Go fallback (no SIMD)
//   - VECGO_SIMD=neon     - ARM64 NEON (128-bit)
//   - VECGO_SIMD=sve2     - ARM64 SVE2 (scalable)
//   - VECGO_SIMD=avx2     - x86-64 AVX2 (256-bit)
//   - VECGO_SIMD=avx512   - x86-64 AVX-512 (512-bit)
//
// This is useful for:
//   - Testing specific code paths on multi-ISA machines
//   - Benchmarking different implementations
//   - CI verification of all SIMD paths
//
// # Operations
//
//   - Distance: Dot, SquaredL2, Hamming
//   - Batch: DotBatch, SquaredL2Batch
//   - Quantized: PQ ADC lookup, SQ8 L2, INT4 L2
//   - Filter: FilterRangeF64, GatherU32
//
// # Architecture
//
// The package uses a simple function-pointer dispatch pattern:
//
//  1. capability.go detects CPU features and active ISA at init time
//  2. kernels.go defines function types and generic implementations
//  3. kernels_arm64.go / kernels_amd64.go register platform-specific kernels
//
// All dispatch happens once at package init - zero runtime overhead.
//
// # Performance
//
// Typical speedup over generic Go:
//
//   - float32 dot product: 4-8x
//   - Hamming distance: 8-16x
//   - Batch operations: 10-20x
//
// Run benchmarks: go test ./internal/simd -bench . -benchmem
package simd

import "math"

// Sqrt returns the square root of x.
// Inlined wrapper for math.Sqrt - no SIMD benefit for scalar ops.
func Sqrt(x float32) float32 {
	return float32(math.Sqrt(float64(x)))
}
