package simd

import (
	"math"
	"testing"
)

func TestFilterRangeF64(t *testing.T) {
	tests := []struct {
		name     string
		values   []float64
		minVal   float64
		maxVal   float64
		expected []byte
	}{
		{
			name:     "Empty",
			values:   []float64{},
			minVal:   0,
			maxVal:   10,
			expected: []byte{},
		},
		{
			name:     "All in range",
			values:   []float64{1.0, 2.0, 3.0, 4.0, 5.0},
			minVal:   0.0,
			maxVal:   10.0,
			expected: []byte{1, 1, 1, 1, 1},
		},
		{
			name:     "None in range",
			values:   []float64{1.0, 2.0, 3.0, 4.0, 5.0},
			minVal:   10.0,
			maxVal:   20.0,
			expected: []byte{0, 0, 0, 0, 0},
		},
		{
			name:     "Some in range",
			values:   []float64{1.0, 5.0, 10.0, 15.0, 20.0},
			minVal:   5.0,
			maxVal:   15.0,
			expected: []byte{0, 1, 1, 1, 0},
		},
		{
			name:     "Boundary inclusive",
			values:   []float64{5.0, 10.0},
			minVal:   5.0,
			maxVal:   10.0,
			expected: []byte{1, 1},
		},
		{
			name:     "Single element in range",
			values:   []float64{5.0},
			minVal:   0.0,
			maxVal:   10.0,
			expected: []byte{1},
		},
		{
			name:     "Single element out of range",
			values:   []float64{15.0},
			minVal:   0.0,
			maxVal:   10.0,
			expected: []byte{0},
		},
		{
			name:     "Negative values",
			values:   []float64{-10.0, -5.0, 0.0, 5.0, 10.0},
			minVal:   -5.0,
			maxVal:   5.0,
			expected: []byte{0, 1, 1, 1, 0},
		},
		{
			name:     "Large array (16 elements - tests unrolling)",
			values:   []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			minVal:   5.0,
			maxVal:   12.0,
			expected: []byte{0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0},
		},
		{
			name:     "17 elements (tests remainder handling)",
			values:   []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17},
			minVal:   5.0,
			maxVal:   12.0,
			expected: []byte{0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			dst := make([]byte, len(tc.values))
			result := FilterRangeF64(tc.values, tc.minVal, tc.maxVal, dst)

			if len(result) != len(tc.expected) {
				t.Fatalf("length mismatch: got %d, want %d", len(result), len(tc.expected))
			}

			for i, v := range result {
				if v != tc.expected[i] {
					t.Errorf("index %d: got %d, want %d", i, v, tc.expected[i])
				}
			}
		})
	}
}

func TestFilterRangeF64_SpecialValues(t *testing.T) {
	tests := []struct {
		name     string
		values   []float64
		minVal   float64
		maxVal   float64
		expected []byte
	}{
		{
			name:     "Infinity",
			values:   []float64{math.Inf(-1), -1.0, 0.0, 1.0, math.Inf(1)},
			minVal:   -1.0,
			maxVal:   1.0,
			expected: []byte{0, 1, 1, 1, 0},
		},
		{
			name:     "Include positive infinity",
			values:   []float64{0.0, 1.0, math.Inf(1)},
			minVal:   0.0,
			maxVal:   math.Inf(1),
			expected: []byte{1, 1, 1},
		},
		{
			name:     "Very small range",
			values:   []float64{1.0, 1.0000001, 1.0000002},
			minVal:   1.0,
			maxVal:   1.0000001,
			expected: []byte{1, 1, 0},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			dst := make([]byte, len(tc.values))
			result := FilterRangeF64(tc.values, tc.minVal, tc.maxVal, dst)

			for i, v := range result {
				if v != tc.expected[i] {
					t.Errorf("index %d: got %d, want %d", i, v, tc.expected[i])
				}
			}
		})
	}
}

func TestFilterRangeF64Indices(t *testing.T) {
	tests := []struct {
		name     string
		values   []float64
		minVal   float64
		maxVal   float64
		expected []int32
	}{
		{
			name:     "Empty",
			values:   []float64{},
			minVal:   0,
			maxVal:   10,
			expected: []int32{},
		},
		{
			name:     "All in range",
			values:   []float64{1.0, 2.0, 3.0, 4.0, 5.0},
			minVal:   0.0,
			maxVal:   10.0,
			expected: []int32{0, 1, 2, 3, 4},
		},
		{
			name:     "None in range",
			values:   []float64{1.0, 2.0, 3.0, 4.0, 5.0},
			minVal:   10.0,
			maxVal:   20.0,
			expected: []int32{},
		},
		{
			name:     "Some in range",
			values:   []float64{1.0, 5.0, 10.0, 15.0, 20.0},
			minVal:   5.0,
			maxVal:   15.0,
			expected: []int32{1, 2, 3},
		},
		{
			name:     "Alternating",
			values:   []float64{0.0, 5.0, 0.0, 5.0, 0.0, 5.0},
			minVal:   4.0,
			maxVal:   6.0,
			expected: []int32{1, 3, 5},
		},
		{
			name:     "Large array",
			values:   []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			minVal:   5.0,
			maxVal:   8.0,
			expected: []int32{4, 5, 6, 7},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			dst := make([]int32, len(tc.values))
			result := FilterRangeF64Indices(tc.values, tc.minVal, tc.maxVal, dst)

			if len(result) != len(tc.expected) {
				t.Fatalf("length mismatch: got %d, want %d", len(result), len(tc.expected))
			}

			for i, v := range result {
				if v != tc.expected[i] {
					t.Errorf("index %d: got %d, want %d", i, v, tc.expected[i])
				}
			}
		})
	}
}

func TestCountRangeF64(t *testing.T) {
	tests := []struct {
		name     string
		values   []float64
		minVal   float64
		maxVal   float64
		expected int
	}{
		{
			name:     "Empty",
			values:   []float64{},
			minVal:   0,
			maxVal:   10,
			expected: 0,
		},
		{
			name:     "All in range",
			values:   []float64{1.0, 2.0, 3.0, 4.0, 5.0},
			minVal:   0.0,
			maxVal:   10.0,
			expected: 5,
		},
		{
			name:     "None in range",
			values:   []float64{1.0, 2.0, 3.0, 4.0, 5.0},
			minVal:   10.0,
			maxVal:   20.0,
			expected: 0,
		},
		{
			name:     "Some in range",
			values:   []float64{1.0, 5.0, 10.0, 15.0, 20.0},
			minVal:   5.0,
			maxVal:   15.0,
			expected: 3,
		},
		{
			name:     "Large array",
			values:   []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			minVal:   5.0,
			maxVal:   12.0,
			expected: 8,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result := CountRangeF64(tc.values, tc.minVal, tc.maxVal)
			if result != tc.expected {
				t.Errorf("got %d, want %d", result, tc.expected)
			}
		})
	}
}

func TestGatherU32(t *testing.T) {
	tests := []struct {
		name     string
		src      []uint32
		indices  []int32
		expected []uint32
	}{
		{
			name:     "Empty indices",
			src:      []uint32{10, 20, 30, 40, 50},
			indices:  []int32{},
			expected: []uint32{},
		},
		{
			name:     "Single index",
			src:      []uint32{10, 20, 30, 40, 50},
			indices:  []int32{2},
			expected: []uint32{30},
		},
		{
			name:     "Multiple indices",
			src:      []uint32{10, 20, 30, 40, 50},
			indices:  []int32{0, 2, 4},
			expected: []uint32{10, 30, 50},
		},
		{
			name:     "All indices",
			src:      []uint32{10, 20, 30, 40, 50},
			indices:  []int32{0, 1, 2, 3, 4},
			expected: []uint32{10, 20, 30, 40, 50},
		},
		{
			name:     "Reverse order",
			src:      []uint32{10, 20, 30, 40, 50},
			indices:  []int32{4, 3, 2, 1, 0},
			expected: []uint32{50, 40, 30, 20, 10},
		},
		{
			name:     "Duplicate indices",
			src:      []uint32{10, 20, 30, 40, 50},
			indices:  []int32{0, 0, 2, 2, 4},
			expected: []uint32{10, 10, 30, 30, 50},
		},
		{
			name:     "Large gather",
			src:      []uint32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
			indices:  []int32{15, 14, 13, 12, 11, 10, 9, 8},
			expected: []uint32{15, 14, 13, 12, 11, 10, 9, 8},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			dst := make([]uint32, len(tc.indices))
			result := GatherU32(tc.src, tc.indices, dst)

			if len(result) != len(tc.expected) {
				t.Fatalf("length mismatch: got %d, want %d", len(result), len(tc.expected))
			}

			for i, v := range result {
				if v != tc.expected[i] {
					t.Errorf("index %d: got %d, want %d", i, v, tc.expected[i])
				}
			}
		})
	}
}

// TestFilterRangeF64_EquivalenceBoundaries tests various sizes to ensure
// SIMD implementations match the generic implementation.
func TestFilterRangeF64_EquivalenceBoundaries(t *testing.T) {
	sizes := []int{0, 1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129}

	for _, size := range sizes {
		t.Run("size="+string(rune('0'+size/100))+string(rune('0'+(size/10)%10))+string(rune('0'+size%10)), func(t *testing.T) {
			values := make([]float64, size)
			for i := range values {
				values[i] = float64(i)
			}

			minVal := float64(size / 4)
			maxVal := float64(3 * size / 4)

			// Get result from current implementation
			dst := make([]byte, size)
			result := FilterRangeF64(values, minVal, maxVal, dst)

			// Verify against expected
			for i, v := range values {
				expected := byte(0)
				if v >= minVal && v <= maxVal {
					expected = 1
				}
				if result[i] != expected {
					t.Errorf("size=%d, index %d: got %d, want %d (value=%.1f, range=[%.1f,%.1f])",
						size, i, result[i], expected, v, minVal, maxVal)
				}
			}
		})
	}
}

// TestFilterRangeF64Indices_EquivalenceBoundaries tests various sizes.
func TestFilterRangeF64Indices_EquivalenceBoundaries(t *testing.T) {
	sizes := []int{0, 1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33}

	for _, size := range sizes {
		t.Run("", func(t *testing.T) {
			values := make([]float64, size)
			for i := range values {
				values[i] = float64(i)
			}

			minVal := float64(size / 4)
			maxVal := float64(3 * size / 4)

			dst := make([]int32, size)
			result := FilterRangeF64Indices(values, minVal, maxVal, dst)

			// Build expected indices
			var expected []int32
			for i, v := range values {
				if v >= minVal && v <= maxVal {
					expected = append(expected, int32(i))
				}
			}

			if len(result) != len(expected) {
				t.Errorf("size=%d: length mismatch: got %d, want %d", size, len(result), len(expected))
				return
			}

			for i := range result {
				if result[i] != expected[i] {
					t.Errorf("size=%d, index %d: got %d, want %d", size, i, result[i], expected[i])
				}
			}
		})
	}
}

// Benchmarks

func BenchmarkFilterRangeF64(b *testing.B) {
	sizes := []int{64, 256, 1024, 4096}
	
	for _, size := range sizes {
		values := make([]float64, size)
		for i := range values {
			values[i] = float64(i)
		}
		dst := make([]byte, size)
		minVal := float64(size / 4)
		maxVal := float64(3 * size / 4)

		b.Run("", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				FilterRangeF64(values, minVal, maxVal, dst)
			}
		})
	}
}

func BenchmarkFilterRangeF64Indices(b *testing.B) {
	sizes := []int{64, 256, 1024, 4096}
	
	for _, size := range sizes {
		values := make([]float64, size)
		for i := range values {
			values[i] = float64(i)
		}
		dst := make([]int32, size)
		minVal := float64(size / 4)
		maxVal := float64(3 * size / 4)

		b.Run("", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				FilterRangeF64Indices(values, minVal, maxVal, dst)
			}
		})
	}
}

func BenchmarkCountRangeF64(b *testing.B) {
	sizes := []int{64, 256, 1024, 4096}
	
	for _, size := range sizes {
		values := make([]float64, size)
		for i := range values {
			values[i] = float64(i)
		}
		minVal := float64(size / 4)
		maxVal := float64(3 * size / 4)

		b.Run("", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				CountRangeF64(values, minVal, maxVal)
			}
		})
	}
}

func BenchmarkGatherU32(b *testing.B) {
	sizes := []int{64, 256, 1024, 4096}
	
	for _, size := range sizes {
		src := make([]uint32, size*2)
		for i := range src {
			src[i] = uint32(i)
		}
		indices := make([]int32, size)
		for i := range indices {
			indices[i] = int32(i * 2) // Every other element
		}
		dst := make([]uint32, size)

		b.Run("", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				GatherU32(src, indices, dst)
			}
		})
	}
}
