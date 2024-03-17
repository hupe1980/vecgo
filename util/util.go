package util

import "math/rand"

// RNG struct encapsulates the random number generator and seed.
type RNG struct {
	rand *rand.Rand
	seed int64
}

// NewRNG creates a new RNG instance with the specified seed.
func NewRNG(seed int64) *RNG {
	return &RNG{
		rand: rand.New(rand.NewSource(seed)), // nolint gosec
		seed: seed,
	}
}

// GenerateRandomVectors generates random vectors using the given RNG.
func (r *RNG) GenerateRandomVectors(num int, dimensions int) [][]float32 {
	vectors := make([][]float32, num)
	for i := range vectors {
		vectors[i] = make([]float32, dimensions)
		for j := range vectors[i] {
			vectors[i][j] = r.rand.Float32()
		}
	}

	return vectors
}
