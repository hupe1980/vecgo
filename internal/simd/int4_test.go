package simd

import (
	"math"
	"testing"
)

func TestInt4L2Distance(t *testing.T) {
	// Create a simple test case
	// 4-dimensional vector
	query := []float32{0.5, 0.5, 0.5, 0.5}

	// min/diff for normalization
	min := []float32{0.0, 0.0, 0.0, 0.0}
	diff := []float32{1.0, 1.0, 1.0, 1.0}

	// Encode a vector [0.5, 0.5, 0.5, 0.5] with INT4
	// quant = round(0.5 * 15) = 8 (binary: 1000)
	// code[0] = (8 << 4) | 8 = 0x88
	// code[1] = (8 << 4) | 8 = 0x88
	code := []byte{0x88, 0x88}

	dist := Int4L2Distance(query, code, min, diff)

	// Expected distance: 4 * ((8/15 - 0.5)^2) ≈ 4 * 0.0011 ≈ 0.0044
	expectedVal := float32(8) / 15.0
	expectedDiff := expectedVal - 0.5
	expectedDist := float32(4 * expectedDiff * expectedDiff)

	if math.Abs(float64(dist-expectedDist)) > 0.001 {
		t.Errorf("Int4L2Distance() = %v, want %v", dist, expectedDist)
	}
}

func TestInt4L2DistanceBatch(t *testing.T) {
	dim := 4
	n := 3
	query := []float32{0.5, 0.5, 0.5, 0.5}

	min := []float32{0.0, 0.0, 0.0, 0.0}
	diff := []float32{1.0, 1.0, 1.0, 1.0}

	// 3 codes, each encoding different values
	// Code 1: [8/15, 8/15, 8/15, 8/15] ≈ [0.53, 0.53, 0.53, 0.53]
	// Code 2: [0/15, 0/15, 0/15, 0/15] = [0, 0, 0, 0]
	// Code 3: [15/15, 15/15, 15/15, 15/15] = [1, 1, 1, 1]
	codeSize := (dim + 1) / 2
	codes := make([]byte, n*codeSize)
	codes[0] = 0x88
	codes[1] = 0x88 // Code 1
	codes[2] = 0x00
	codes[3] = 0x00 // Code 2
	codes[4] = 0xFF
	codes[5] = 0xFF // Code 3

	out := make([]float32, n)
	Int4L2DistanceBatch(query, codes, dim, n, min, diff, out)

	// Verify distances are computed
	for i, d := range out {
		if d < 0 {
			t.Errorf("Distance %d is negative: %v", i, d)
		}
	}

	// Code 2 (all zeros) should have highest distance to [0.5, 0.5, 0.5, 0.5]
	// Distance = 4 * 0.5^2 = 1.0
	if math.Abs(float64(out[1]-1.0)) > 0.001 {
		t.Errorf("Code 2 distance = %v, want ~1.0", out[1])
	}
}

func TestInt4L2DistancePrecomputed(t *testing.T) {
	dim := 4
	query := []float32{0.5, 0.5, 0.5, 0.5}

	min := []float32{0.0, 0.0, 0.0, 0.0}
	diff := []float32{1.0, 1.0, 1.0, 1.0}

	// Build lookup table
	table := BuildInt4LookupTable(min, diff)

	// Verify table size
	if len(table) != 16*dim {
		t.Errorf("Lookup table size = %d, want %d", len(table), 16*dim)
	}

	// Same code as basic test
	code := []byte{0x88, 0x88}

	dist := Int4L2DistancePrecomputed(query, code, table)

	// Should match non-precomputed version
	expectedDist := Int4L2Distance(query, code, min, diff)

	if math.Abs(float64(dist-expectedDist)) > 0.0001 {
		t.Errorf("Precomputed distance = %v, want %v", dist, expectedDist)
	}
}

func TestBuildInt4LookupTable(t *testing.T) {
	min := []float32{-1.0, 0.0}
	diff := []float32{2.0, 1.0}

	table := BuildInt4LookupTable(min, diff)

	// Check dimension 0: val = q/15 * 2.0 + (-1.0)
	// q=0: -1.0, q=15: 1.0
	if math.Abs(float64(table[0*16+0]-(-1.0))) > 0.001 {
		t.Errorf("table[0][0] = %v, want -1.0", table[0])
	}
	if math.Abs(float64(table[0*16+15]-1.0)) > 0.001 {
		t.Errorf("table[0][15] = %v, want 1.0", table[15])
	}

	// Check dimension 1: val = q/15 * 1.0 + 0.0
	// q=0: 0.0, q=15: 1.0
	if math.Abs(float64(table[1*16+0]-0.0)) > 0.001 {
		t.Errorf("table[1][0] = %v, want 0.0", table[16])
	}
	if math.Abs(float64(table[1*16+15]-1.0)) > 0.001 {
		t.Errorf("table[1][15] = %v, want 1.0", table[31])
	}
}

func BenchmarkInt4L2Distance(b *testing.B) {
	dim := 128
	query := make([]float32, dim)
	min := make([]float32, dim)
	diff := make([]float32, dim)

	for i := 0; i < dim; i++ {
		query[i] = float32(i) / float32(dim)
		min[i] = 0.0
		diff[i] = 1.0
	}

	codeSize := (dim + 1) / 2
	code := make([]byte, codeSize)
	for i := range code {
		code[i] = 0x77 // Both nibbles = 7
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Int4L2Distance(query, code, min, diff)
	}
}

func BenchmarkInt4L2DistancePrecomputed(b *testing.B) {
	dim := 128
	query := make([]float32, dim)
	min := make([]float32, dim)
	diff := make([]float32, dim)

	for i := 0; i < dim; i++ {
		query[i] = float32(i) / float32(dim)
		min[i] = 0.0
		diff[i] = 1.0
	}

	codeSize := (dim + 1) / 2
	code := make([]byte, codeSize)
	for i := range code {
		code[i] = 0x77
	}

	table := BuildInt4LookupTable(min, diff)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Int4L2DistancePrecomputed(query, code, table)
	}
}

func BenchmarkInt4L2DistanceBatch(b *testing.B) {
	dim := 128
	n := 1000
	query := make([]float32, dim)
	min := make([]float32, dim)
	diff := make([]float32, dim)

	for i := 0; i < dim; i++ {
		query[i] = float32(i) / float32(dim)
		min[i] = 0.0
		diff[i] = 1.0
	}

	codeSize := (dim + 1) / 2
	codes := make([]byte, n*codeSize)
	for i := range codes {
		codes[i] = 0x77
	}

	out := make([]float32, n)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Int4L2DistanceBatch(query, codes, dim, n, min, diff, out)
	}
}
