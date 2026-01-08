package distance

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDot(t *testing.T) {
	tests := []struct {
		name     string
		a, b     []float32
		expected float32
	}{
		{"Simple", []float32{1, 2, 3}, []float32{4, 5, 6}, 32},
		{"Zero", []float32{0, 0, 0}, []float32{0, 0, 0}, 0},
		{"Mixed", []float32{1, -1, 2}, []float32{1, 1, -2}, -4},
		{"Empty", []float32{}, []float32{}, 0},
		{"Single", []float32{2}, []float32{3}, 6},
		// Large vector to trigger potential loop unrolling/SIMD
		{"Large", make([]float32, 1024), make([]float32, 1024), 0}, // Zeros
	}

	// Setup large vector
	for i := range tests[5].a {
		tests[5].a[i] = 1
		tests[5].b[i] = 1
	}
	tests[5].expected = 1024

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Dot(tt.a, tt.b)
			assert.InDelta(t, tt.expected, got, 1e-5)
		})
	}
}

func TestSquaredL2(t *testing.T) {
	tests := []struct {
		name     string
		a, b     []float32
		expected float32
	}{
		{"Simple", []float32{1, 2, 3}, []float32{4, 5, 6}, 27},
		{"Zero", []float32{0, 0, 0}, []float32{0, 0, 0}, 0},
		{"Identical", []float32{1, 2, 3}, []float32{1, 2, 3}, 0},
		{"Mixed", []float32{1, -1}, []float32{-1, 1}, 8}, // (1 - -1)^2 + (-1 - 1)^2 = 4 + 4 = 8
		{"Empty", []float32{}, []float32{}, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := SquaredL2(tt.a, tt.b)
			assert.InDelta(t, tt.expected, got, 1e-5)
		})
	}
}

func TestHamming(t *testing.T) {
	tests := []struct {
		name     string
		a, b     []byte
		expected float32
	}{
		{"Simple", []byte{0xFF, 0x00}, []byte{0x00, 0xFF}, 16},
		{"Identical", []byte{0xAA, 0x55}, []byte{0xAA, 0x55}, 0},
		{"Partial", []byte{0b11110000}, []byte{0b11111111}, 4},
		{"Empty", []byte{}, []byte{}, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Hamming(tt.a, tt.b)
			assert.Equal(t, tt.expected, got)
		})
	}
}

func TestNormalizeL2(t *testing.T) {
	t.Run("InPlace", func(t *testing.T) {
		// Normal case
		v := []float32{3, 4}
		ok := NormalizeL2InPlace(v)
		assert.True(t, ok)
		assert.InDelta(t, float32(0.6), v[0], 1e-5)
		assert.InDelta(t, float32(0.8), v[1], 1e-5)

		// Length check of norm
		assert.InDelta(t, float32(1.0), float32(math.Sqrt(float64(v[0]*v[0]+v[1]*v[1]))), 1e-5)

		// Zero vector
		vZero := []float32{0, 0}
		ok = NormalizeL2InPlace(vZero)
		assert.False(t, ok)

		// Empty vector
		vEmpty := []float32{}
		ok = NormalizeL2InPlace(vEmpty)
		assert.False(t, ok)
	})

	t.Run("Copy", func(t *testing.T) {
		v := []float32{1, 0}
		dst, ok := NormalizeL2Copy(v)
		assert.True(t, ok)
		assert.Equal(t, float32(1), dst[0])
		assert.NotSame(t, &v[0], &dst[0])

		vZero := []float32{0, 0}
		dst, ok = NormalizeL2Copy(vZero)
		assert.False(t, ok)
		assert.Nil(t, dst)
	})
}

func TestMetric(t *testing.T) {
	t.Run("String", func(t *testing.T) {
		assert.Equal(t, "L2", MetricL2.String())
		assert.Equal(t, "Cosine", MetricCosine.String())
		assert.Equal(t, "Dot", MetricDot.String())
		assert.Equal(t, "Hamming", MetricHamming.String())
		assert.Equal(t, "Unknown(99)", Metric(99).String())
	})

	t.Run("Provider", func(t *testing.T) {
		f, err := Provider(MetricL2)
		require.NoError(t, err)
		assert.NotNil(t, f)
		assert.InDelta(t, float32(27), f([]float32{1, 2, 3}, []float32{4, 5, 6}), 1e-5)

		f, err = Provider(MetricDot)
		require.NoError(t, err)
		assert.NotNil(t, f)

		f, err = Provider(MetricCosine)
		require.NoError(t, err)
		assert.NotNil(t, f)
		// Cosine usually implies providing normalized vectors to Dot,
		// or implementing full Cosine.
		// Current implementation maps Cosine -> Dot.
		// This implies the user must normalize vectors.
		// Let's verify that assumption in comments or behavior.

		_, err = Provider(MetricHamming)
		assert.Error(t, err, "Should error for float32 provider with Hamming")

		_, err = Provider(Metric(99))
		assert.Error(t, err)
	})

	t.Run("ProviderBytes", func(t *testing.T) {
		f, err := ProviderBytes(MetricHamming)
		require.NoError(t, err)
		assert.NotNil(t, f)
		assert.Equal(t, float32(0), f([]byte{1}, []byte{1}))

		_, err = ProviderBytes(MetricL2)
		assert.Error(t, err, "Should error for byte provider with L2")
	})
}
