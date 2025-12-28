// Package distance provides public API for vector distance calculations.
// All distance functions use SIMD-optimized implementations from internal/math32
// when available (AVX/AVX512 on x86-64, NEON on ARM64).
package distance

import (
	"slices"

	"github.com/hupe1980/vecgo/internal/math32"
)

// Dot calculates the dot product of two vectors.
// Assumes vectors are the same length (caller's responsibility).
// Uses SIMD acceleration when available.
func Dot(a, b []float32) float32 {
	return math32.Dot(a, b)
}

// SquaredL2 calculates the squared L2 (Euclidean) distance between two vectors.
// Assumes vectors are the same length (caller's responsibility).
// Uses SIMD acceleration when available.
func SquaredL2(a, b []float32) float32 {
	return math32.SquaredL2(a, b)
}

// NormalizeL2InPlace L2-normalizes v in place.
// Returns false if v has zero L2 norm.
func NormalizeL2InPlace(v []float32) bool {
	if len(v) == 0 {
		return false
	}
	norm2 := math32.Dot(v, v)
	if norm2 == 0 {
		return false
	}
	inv := 1 / math32.Sqrt(norm2)
	math32.ScaleInPlace(v, inv)
	return true
}

// NormalizeL2Copy returns a normalized copy of src.
// Returns false if src has zero L2 norm.
func NormalizeL2Copy(src []float32) ([]float32, bool) {
	dst := slices.Clone(src)
	if !NormalizeL2InPlace(dst) {
		return nil, false
	}
	return dst, true
}
