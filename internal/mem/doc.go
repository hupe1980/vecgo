// Package mem provides memory allocation utilities for SIMD operations.
//
// # Aligned Allocation
//
// SIMD instructions (AVX-512, AVX2, NEON) achieve optimal performance when
// operating on memory addresses aligned to cache line boundaries. Misaligned
// accesses may cause:
//   - Performance penalties (cross-cache-line loads)
//   - Crashes on strict architectures (some ARM processors)
//   - Suboptimal vectorization in compilers
//
// This package provides 64-byte aligned allocation, which satisfies:
//   - AVX-512 (requires 64-byte alignment for optimal performance)
//   - AVX2 (requires 32-byte, satisfied by 64)
//   - SSE (requires 16-byte, satisfied by 64)
//   - ARM NEON (requires 16-byte, satisfied by 64)
//
// # Memory Overhead
//
// Each allocation may use up to 63 extra bytes to ensure alignment.
// For small allocations (< 64 bytes), consider batching to amortize overhead.
//
// # Thread Safety
//
// All functions are safe for concurrent use (they only call make()).
//
// # Example
//
//	// Allocate aligned float32 slice for SIMD distance computation
//	vec := mem.AllocAlignedFloat32(128) // 128 floats, 512 bytes, 64-byte aligned
package mem
