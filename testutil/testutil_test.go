package testutil

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestUniformVectors(t *testing.T) {
	rng := NewRNG(4711)

	v := rng.UniformVectors(8, 32)

	assert.Equal(t, 8, len(v))
	assert.Equal(t, 32, len(v[0]))
	assert.LessOrEqual(t, v[0][0], float32(1.0))
	assert.GreaterOrEqual(t, v[1][0], float32(0.0))
}

func TestUniformRangeVectors(t *testing.T) {
	rng := NewRNG(4711)

	v := rng.UniformRangeVectors(8, 32)

	assert.Equal(t, 8, len(v))
	assert.Equal(t, 32, len(v[0]))
	assert.LessOrEqual(t, v[0][0], float32(1.0))
	assert.GreaterOrEqual(t, v[1][0], float32(-1.0))
}

func TestUnitVectors(t *testing.T) {
	rng := NewRNG(4711)

	v := rng.UnitVectors(8, 32)

	assert.Equal(t, 8, len(v))
	assert.Equal(t, 32, len(v[0]))

	// Check normalization
	for _, vec := range v {
		var sum float32
		for _, val := range vec {
			sum += val * val
		}
		assert.InDelta(t, float32(1.0), sum, 1e-5)
	}
}

func TestClusteredVectors(t *testing.T) {
	rng := NewRNG(4711)

	v := rng.ClusteredVectors(100, 32, 5, 0.1)

	assert.Equal(t, 100, len(v))
	assert.Equal(t, 32, len(v[0]))
}

func TestReset(t *testing.T) {
	rng := NewRNG(4711)
	v1 := rng.UniformVectors(1, 10)

	rng.Reset()
	v2 := rng.UniformVectors(1, 10)

	assert.Equal(t, v1, v2)
}
