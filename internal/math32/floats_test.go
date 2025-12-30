package math32

import (
	"testing"

	"github.com/hupe1980/vecgo/testutil"
	"github.com/stretchr/testify/assert"
)

func TestDot(t *testing.T) {
	tests := []struct {
		name     string
		a, b     []float32
		expected float32
	}{
		{"Positive values (size 3)", []float32{1, 2, 3}, []float32{4, 5, 6}, 32.0},
		{"Negative values (size 3)", []float32{-1, -2, -3}, []float32{-4, -5, -6}, 32.0},
		{"More than 4 (size 6)", []float32{1, 2, 3, 1, 2, 3}, []float32{4, 5, 6, 4, 5, 6}, 64.0},
		{"Mixed values (size 3)", []float32{1, -2, 3}, []float32{-4, 5, -6}, -32.0},
		{"Zero values (size 3)", []float32{0, 0, 0}, []float32{0, 0, 0}, 0.0},
		{"Positive values (size 9)", []float32{1, 2, 3, 4, 5, 6, 7, 8, 9}, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9}, 285.0},
		{"Positive values (size 10)", []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, 385.0},
		{"Positive values (size 15)", []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, 1240.0},
		{"Positive values (size 16)", []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}, 1496.0},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result := Dot(tc.a, tc.b)
			assert.Equal(t, tc.expected, result)
		})
	}
}

// BenchmarkDot-10    	    7623	    157954 ns/op	       0 B/op	       0 allocs/op
func BenchmarkDot(b *testing.B) {
	// Generate random float32 slices for benchmarking.
	const size = 1000000 // Size of slices
	rng := testutil.NewRNG(0)
	va := rng.UniformVectors(1, size)[0]
	vb := rng.UniformVectors(1, size)[0]

	// Run the Dot function b.N times and measure the time taken.
	b.ResetTimer()
	for b.Loop() {
		_ = Dot(va, vb)
	}
}

func TestSquaredL2(t *testing.T) {
	tests := []struct {
		name     string
		a, b     []float32
		expected float32
	}{
		{"Positive values", []float32{1, 2, 3}, []float32{4, 5, 6}, 27.0},
		{"Negative values", []float32{-1, -2, -3}, []float32{-4, -5, -6}, 27.0},
		{"1 Remainder", []float32{1, 2, 3, 1, 2, 3}, []float32{4, 5, 6, 4, 5, 6}, 54.0},
		{"Mixed values", []float32{1, -2, 3}, []float32{-4, 5, -6}, 155.0},
		{"Zero values", []float32{0, 0, 0}, []float32{0, 0, 0}, 0.0},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result := SquaredL2(tc.a, tc.b)
			assert.Equal(t, tc.expected, result)
		})
	}
}

// BenchmarkSquaredL2-10    	    5128	    235120 ns/op	       0 B/op	       0 allocs/op
func BenchmarkSquaredL2(b *testing.B) {
	// Generate random float32 slices for benchmarking.
	const size = 1000000 // Size of slices
	rng := testutil.NewRNG(0)
	va := rng.UniformVectors(1, size)[0]
	vb := rng.UniformVectors(1, size)[0]

	// Run the Dot function b.N times and measure the time taken.
	b.ResetTimer()
	for b.Loop() {
		_ = SquaredL2(va, vb)
	}
}

func TestPqAdcLookup(t *testing.T) {
	// M=2 sub-vectors, 256 centroids each.
	// Distance table size = 2 * 256 = 512
	m := 2
	k := 256
	table := make([]float32, m*k)

	// Fill table with known values
	// Sub-vector 0: distances 0..255
	// Sub-vector 1: distances 0..255
	for i := 0; i < m*k; i++ {
		table[i] = float32(i % k)
	}

	tests := []struct {
		name     string
		codes    []byte
		expected float32
	}{
		{
			name:     "First centroids",
			codes:    []byte{0, 0},
			expected: 0.0 + 0.0, // table[0] + table[256] -> 0 + 0
		},
		{
			name:     "Last centroids",
			codes:    []byte{255, 255},
			expected: 255.0 + 255.0, // table[255] + table[511] -> 255 + 255
		},
		{
			name:     "Mixed centroids",
			codes:    []byte{10, 20},
			expected: 10.0 + 20.0, // table[10] + table[256+20] -> 10 + 20
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result := PqAdcLookup(table, tc.codes, m)
			assert.Equal(t, tc.expected, result)
		})
	}
}

func TestScaleInPlace(t *testing.T) {
	tests := []struct {
		name     string
		input    []float32
		scalar   float32
		expected []float32
	}{
		{"Scale by 2", []float32{1, 2, 3}, 2.0, []float32{2, 4, 6}},
		{"Scale by 0", []float32{1, 2, 3}, 0.0, []float32{0, 0, 0}},
		{"Scale by -1", []float32{1, -2, 3}, -1.0, []float32{-1, 2, -3}},
		{"Empty", []float32{}, 2.0, []float32{}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			// Copy input because it's modified in place
			arr := make([]float32, len(tc.input))
			copy(arr, tc.input)

			ScaleInPlace(arr, tc.scalar)
			assert.Equal(t, tc.expected, arr)
		})
	}
}
