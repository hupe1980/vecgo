package simd

import (
	"math/rand"
	"strconv"
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

func TestDot_Nil(t *testing.T) {
	var a []float32
	var b []float32
	assert.Equal(t, float32(0), Dot(a, b))
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

func TestSquaredL2_Nil(t *testing.T) {
	var a []float32
	var b []float32
	assert.Equal(t, float32(0), SquaredL2(a, b))
}

func TestDotSquaredL2_EquivalenceBoundaries(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	lengths := []int{0, 1, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65}

	for _, n := range lengths {
		a := make([]float32, n)
		b := make([]float32, n)
		for i := 0; i < n; i++ {
			// Keep values in a modest range to reduce float error amplification.
			a[i] = rng.Float32()*2 - 1
			b[i] = rng.Float32()*2 - 1
		}

		wantDot := dotGeneric(a, b)
		gotDot := Dot(a, b)
		assert.InDelta(t, wantDot, gotDot, 1e-4, "dot mismatch for n=%d", n)

		wantL2 := squaredL2Generic(a, b)
		gotL2 := SquaredL2(a, b)
		assert.InDelta(t, wantL2, gotL2, 1e-4, "squaredL2 mismatch for n=%d", n)
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

func BenchmarkPopcount_128Bytes(b *testing.B) {
	in := make([]byte, 128)
	for i := range in {
		in[i] = byte(i * 131)
	}
	var sink int64
	b.ResetTimer()
	for b.Loop() {
		sink = Popcount(in)
	}
	_ = sink
}

func BenchmarkPopcount_768Bytes(b *testing.B) {
	in := make([]byte, 768)
	for i := range in {
		in[i] = byte(i * 131)
	}
	var sink int64
	b.ResetTimer()
	for b.Loop() {
		sink = Popcount(in)
	}
	_ = sink
}

func BenchmarkHamming_128Bytes(b *testing.B) {
	a := make([]byte, 128)
	bv := make([]byte, 128)
	for i := range a {
		a[i] = byte(i * 131)
		bv[i] = byte((i * 131) ^ 0x5A)
	}
	var sink int64
	b.ResetTimer()
	for b.Loop() {
		sink = Hamming(a, bv)
	}
	_ = sink
}

func BenchmarkHamming_768Bytes(b *testing.B) {
	a := make([]byte, 768)
	bv := make([]byte, 768)
	for i := range a {
		a[i] = byte(i * 131)
		bv[i] = byte((i * 131) ^ 0x5A)
	}
	var sink int64
	b.ResetTimer()
	for b.Loop() {
		sink = Hamming(a, bv)
	}
	_ = sink
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

func TestPqAdcLookup_ExactM8(t *testing.T) {
	const m = 8
	table := make([]float32, m*256)
	for i := 0; i < m; i++ {
		for j := 0; j < 256; j++ {
			// Keep values as small integers so the sum is exactly representable in float32.
			table[i*256+j] = float32(i*1000 + j)
		}
	}

	codes := []byte{0, 1, 2, 3, 4, 5, 6, 7}
	var want float32
	for i := 0; i < m; i++ {
		want += table[i*256+int(codes[i])]
	}

	got := PqAdcLookup(table, codes, m)
	assert.Equal(t, want, got)
}

func TestPqAdcLookup_ExactM16(t *testing.T) {
	const m = 16
	table := make([]float32, m*256)
	for i := 0; i < m; i++ {
		for j := 0; j < 256; j++ {
			// Keep values as small integers so the sum is exactly representable in float32.
			table[i*256+j] = float32(i*1000 + j)
		}
	}

	codes := []byte{0, 17, 34, 51, 68, 85, 102, 119, 136, 153, 170, 187, 204, 221, 238, 255}
	var want float32
	for i := 0; i < m; i++ {
		want += table[i*256+int(codes[i])]
	}

	got := PqAdcLookup(table, codes, m)
	assert.Equal(t, want, got)
}

func TestPqAdcLookup_EquivalenceBoundaries(t *testing.T) {
	rng := rand.New(rand.NewSource(123))
	ms := []int{0, 1, 2, 7, 8, 9, 15, 16, 17}

	for _, m := range ms {
		t.Run("m="+strconv.Itoa(m), func(t *testing.T) {
			if m == 0 {
				assert.Equal(t, float32(0), PqAdcLookup(nil, nil, 0))
				return
			}

			table := make([]float32, m*256)
			codes := make([]byte, m)
			for i := range table {
				table[i] = rng.Float32()*2 - 1
			}
			for i := 0; i < m; i++ {
				codes[i] = byte(rng.Intn(256))
			}

			want := pqAdcLookupGeneric(table, codes, m)
			got := PqAdcLookup(table, codes, m)
			assert.InDelta(t, want, got, 1e-4, "pqAdc mismatch for m=%d", m)
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

func TestScaleInPlace_Nil(t *testing.T) {
	var a []float32
	ScaleInPlace(a, 2)
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

func TestSq8L2Batch_EquivalenceBoundaries(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	dims := []int{1, 7, 8, 15, 16, 17, 31, 32, 33}
	batchSizes := []int{1, 2, 5}

	for _, dim := range dims {
		for _, n := range batchSizes {
			query := make([]float32, dim)
			for i := range query {
				query[i] = rng.Float32()*2 - 1
			}

			codes := make([]int8, n*dim)
			scales := make([]float32, n)
			biases := make([]float32, n)
			for i := 0; i < n; i++ {
				scales[i] = rng.Float32()*2 - 1
				biases[i] = rng.Float32()*2 - 1
				for j := 0; j < dim; j++ {
					codes[i*dim+j] = int8(rng.Intn(256) - 128)
				}
			}

			got := make([]float32, n)
			want := make([]float32, n)
			Sq8L2Batch(query, codes, scales, biases, dim, got)
			sq8L2BatchGeneric(query, codes, scales, biases, dim, want)

			for i := 0; i < n; i++ {
				assert.InDelta(t, want[i], got[i], 1e-2, "sq8L2 mismatch for dim=%d n=%d i=%d", dim, n, i)
			}
		}
	}
}

func TestSq8uL2BatchPerDimension_EquivalenceBoundaries(t *testing.T) {
	rng := rand.New(rand.NewSource(9))
	dims := []int{1, 7, 8, 15, 16, 17, 31, 32, 33}
	batchSizes := []int{1, 2, 5}

	for _, dim := range dims {
		mins := make([]float32, dim)
		invScales := make([]float32, dim)
		query := make([]float32, dim)
		for i := 0; i < dim; i++ {
			query[i] = rng.Float32()*2 - 1
			mins[i] = rng.Float32()*2 - 1
			invScales[i] = rng.Float32()*2 - 1
		}

		for _, n := range batchSizes {
			codes := make([]byte, n*dim)
			for i := range codes {
				codes[i] = byte(rng.Intn(256))
			}

			got := make([]float32, n)
			want := make([]float32, n)
			Sq8uL2BatchPerDimension(query, codes, mins, invScales, dim, got)
			sq8uL2BatchPerDimensionGeneric(query, codes, mins, invScales, dim, want)

			for i := 0; i < n; i++ {
				assert.InDelta(t, want[i], got[i], 5e-2, "sq8uL2 mismatch for dim=%d n=%d i=%d", dim, n, i)
			}
		}
	}
}

func TestPopcount(t *testing.T) {
	tests := []struct {
		name string
		in   []byte
		want int64
	}{
		{"Empty", []byte{}, 0},
		{"Nil", nil, 0},
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

func TestPopcount_EquivalenceBoundaries(t *testing.T) {
	lengths := []int{0, 1, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65}
	for _, n := range lengths {
		in := make([]byte, n)
		for i := range in {
			// Deterministic, non-trivial bit patterns.
			in[i] = byte(i*131) ^ byte(i>>1)
		}
		want := popcountGeneric(in)
		got := Popcount(in)
		assert.Equal(t, want, got, "popcount mismatch for n=%d", n)
	}
}

func TestHamming(t *testing.T) {
	tests := []struct {
		name string
		a, b []byte
		want int64
	}{
		{"Empty", []byte{}, []byte{}, 0},
		{"Nil", nil, nil, 0},
		{"Identical", []byte{0xFF, 0xAA}, []byte{0xFF, 0xAA}, 0},
		{"Complement", []byte{0x00, 0xFF}, []byte{0xFF, 0x00}, 16},
		{"Mixed", []byte{0x0F}, []byte{0xF0}, 8},
		{"8 bytes", []byte{1, 0, 0, 0, 0, 0, 0, 0}, []byte{0, 0, 0, 0, 0, 0, 0, 0}, 1},
		{
			"17 bytes (tail path)",
			[]byte{0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10},
			[]byte{0xFF, 0x01, 0xFD, 0x03, 0xFB, 0x05, 0xF9, 0x07, 0xF7, 0x09, 0xF5, 0x0B, 0xF3, 0x0D, 0xF1, 0x0F, 0xEF},
			// Every even index differs by XOR with 0xFF, odds are identical.
			// 9 bytes differ (indices 0,2,4,...,16) => 9 * 8 bits.
			72,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := Hamming(tc.a, tc.b)
			assert.Equal(t, tc.want, got)
		})
	}
}

func TestHamming_EquivalenceBoundaries(t *testing.T) {
	lengths := []int{0, 1, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65}
	for _, n := range lengths {
		a := make([]byte, n)
		b := make([]byte, n)
		for i := 0; i < n; i++ {
			a[i] = byte(i*131) ^ byte(i>>1)
			b[i] = a[i] ^ 0x5A
		}
		want := hammingGeneric(a, b)
		got := Hamming(a, b)
		assert.Equal(t, want, got, "hamming mismatch for n=%d", n)
	}
}
