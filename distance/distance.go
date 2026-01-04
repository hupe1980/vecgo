// Package distance provides public API for vector distance calculations.
// All distance functions use SIMD-optimized implementations from internal/simd
// when available (AVX/AVX512 on x86-64, NEON on ARM64).
package distance

import (
	"fmt"
	"slices"

	"github.com/hupe1980/vecgo/internal/simd"
)

// Dot calculates the dot product of two vectors.
// Assumes vectors are the same length (caller's responsibility).
// Uses SIMD acceleration when available.
func Dot(a, b []float32) float32 {
	return simd.Dot(a, b)
}

// SquaredL2 calculates the squared L2 (Euclidean) distance between two vectors.
// Assumes vectors are the same length (caller's responsibility).
// Uses SIMD acceleration when available.
func SquaredL2(a, b []float32) float32 {
	return simd.SquaredL2(a, b)
}

// Hamming calculates the Hamming distance between two byte slices.
// Assumes slices are the same length.
// Returns the count of differing bits as a float32.
func Hamming(a, b []byte) float32 {
	return float32(simd.Hamming(a, b))
}

// NormalizeL2InPlace L2-normalizes v in place.
// Returns false if v has zero L2 norm.
func NormalizeL2InPlace(v []float32) bool {
	if len(v) == 0 {
		return false
	}
	norm2 := simd.Dot(v, v)
	if norm2 == 0 {
		return false
	}
	inv := 1 / simd.Sqrt(norm2)
	simd.ScaleInPlace(v, inv)
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

// Metric represents the distance metric used for vector comparison.
type Metric int

const (
	MetricL2 Metric = iota
	MetricCosine
	MetricDot
	MetricHamming
)

func (m Metric) String() string {
	switch m {
	case MetricL2:
		return "L2"
	case MetricCosine:
		return "Cosine"
	case MetricDot:
		return "Dot"
	case MetricHamming:
		return "Hamming"
	default:
		return fmt.Sprintf("Unknown(%d)", m)
	}
}

// Func is a function type for distance calculation.
type Func func(a, b []float32) float32

// FuncBytes is a function type for distance calculation on byte slices.
type FuncBytes func(a, b []byte) float32

// Provider returns the distance function for the given metric.
func Provider(m Metric) (Func, error) {
	switch m {
	case MetricL2:
		return SquaredL2, nil
	case MetricCosine, MetricDot:
		return Dot, nil
	default:
		return nil, fmt.Errorf("unsupported metric for float32: %v", m)
	}
}

// ProviderBytes returns the distance function for the given metric on byte slices.
func ProviderBytes(m Metric) (FuncBytes, error) {
	switch m {
	case MetricHamming:
		return Hamming, nil
	default:
		return nil, fmt.Errorf("unsupported metric for bytes: %v", m)
	}
}
