package simd

import (
	"math/rand"
	"testing"

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
	va := randomFloats(size)
	vb := randomFloats(size)

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

func TestDotBatch(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	dims := []int{1, 3, 7, 16, 33}
	batchSizes := []int{1, 5, 17}

	for _, dim := range dims {
		for _, n := range batchSizes {
			query := make([]float32, dim)
			for i := range query {
				query[i] = rng.Float32()*2 - 1
			}

			targets := make([]float32, n*dim)
			for i := range targets {
				targets[i] = rng.Float32()*2 - 1
			}

			out := make([]float32, n)
			DotBatch(query, targets, dim, out)

			for i := 0; i < n; i++ {
				offset := i * dim
				vec := targets[offset : offset+dim]
				expected := dotGeneric(query, vec)
				assert.InDelta(t, expected, out[i], 1e-4)
			}
		}
	}
}

func TestSquaredL2Batch(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	dims := []int{1, 3, 7, 16, 33}
	batchSizes := []int{1, 5, 17}

	for _, dim := range dims {
		for _, n := range batchSizes {
			query := make([]float32, dim)
			for i := range query {
				query[i] = rng.Float32()*2 - 1
			}

			targets := make([]float32, n*dim)
			for i := range targets {
				targets[i] = rng.Float32()*2 - 1
			}

			out := make([]float32, n)
			SquaredL2Batch(query, targets, dim, out)

			for i := 0; i < n; i++ {
				offset := i * dim
				vec := targets[offset : offset+dim]
				expected := squaredL2Generic(query, vec)
				assert.InDelta(t, expected, out[i], 1e-4)
			}
		}
	}
}

// BenchmarkSquaredL2-10    	    5128	    235120 ns/op	       0 B/op	       0 allocs/op
func BenchmarkSquaredL2(b *testing.B) {
	// Generate random float32 slices for benchmarking.
	const size = 1000000 // Size of slices
	va := randomFloats(size)
	vb := randomFloats(size)

	// Run the Dot function b.N times and measure the time taken.
	b.ResetTimer()
	for b.Loop() {
		_ = SquaredL2(va, vb)
	}
}

func randomFloats(n int) []float32 {
	res := make([]float32, n)
	for i := range res {
		res[i] = rand.Float32()
	}
	return res
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

func TestF16ToF32(t *testing.T) {
	// Known values
	// 0x3c00 = 1.0
	// 0x4000 = 2.0
	// 0xc000 = -2.0
	// 0x0000 = 0.0
	in := []uint16{0x3c00, 0x4000, 0xc000, 0x0000}
	expected := []float32{1.0, 2.0, -2.0, 0.0}
	out := make([]float32, len(in))

	F16ToF32(in, out)
	assert.Equal(t, expected, out)
}

func TestSq8L2Batch(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	dim := 16
	n := 10

	query := make([]float32, dim)
	for i := range query {
		query[i] = rng.Float32()
	}

	codes := make([]int8, n*dim)
	scales := make([]float32, n)
	biases := make([]float32, n)

	for i := 0; i < n; i++ {
		scales[i] = rng.Float32()
		biases[i] = rng.Float32()
		for j := 0; j < dim; j++ {
			codes[i*dim+j] = int8(rng.Intn(256) - 128)
		}
	}

	out := make([]float32, n)
	Sq8L2Batch(query, codes, scales, biases, dim, out)

	// Verify with generic implementation logic
	for i := 0; i < n; i++ {
		var sum float32
		for j := 0; j < dim; j++ {
			val := float32(codes[i*dim+j])*scales[i] + biases[i]
			diff := query[j] - val
			sum += diff * diff
		}
		assert.InDelta(t, sum, out[i], 1e-2)
	}
}

func TestPopcount(t *testing.T) {
	tests := []struct {
		name string
		in   []byte
		want int64
	}{
		{"Empty", []byte{}, 0},
		{"Zero", []byte{0, 0, 0}, 0},
		{"All ones", []byte{0xFF, 0xFF}, 16},
		{"Mixed", []byte{0x0F, 0xF0, 0xAA}, 4 + 4 + 4}, // 12
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := Popcount(tc.in)
			assert.Equal(t, tc.want, got)
		})
	}
}

func TestHamming(t *testing.T) {
	tests := []struct {
		name string
		a, b []byte
		want int64
	}{
		{"Empty", []byte{}, []byte{}, 0},
		{"Identical", []byte{0xFF, 0xAA}, []byte{0xFF, 0xAA}, 0},
		{"Complement", []byte{0x00, 0xFF}, []byte{0xFF, 0x00}, 16},
		{"Mixed", []byte{0x0F}, []byte{0xF0}, 8},
		{"8 bytes", []byte{1, 0, 0, 0, 0, 0, 0, 0}, []byte{0, 0, 0, 0, 0, 0, 0, 0}, 1},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := Hamming(tc.a, tc.b)
			assert.Equal(t, tc.want, got)
		})
	}
}
