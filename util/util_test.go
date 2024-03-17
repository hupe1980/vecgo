package util

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestGenerateRandomVectors(t *testing.T) {
	rng := NewRNG(4711)

	v := rng.GenerateRandomVectors(8, 32)

	assert.Equal(t, 8, len(v))
	assert.Equal(t, 32, len(v[0]))
	assert.LessOrEqual(t, v[0][0], float32(1.0))
	assert.GreaterOrEqual(t, v[1][0], float32(0.0))
}
