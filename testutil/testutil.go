package testutil

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
// Values are in range [0, 1).
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

// GenerateNormalizedVectors generates random vectors with values in range [-1, 1).
func (r *RNG) GenerateNormalizedVectors(num int, dimensions int) [][]float32 {
	vectors := make([][]float32, num)
	for i := range vectors {
		vectors[i] = r.NormalizedVector(dimensions)
	}

	return vectors
}

// NormalizedVector generates a single random vector with values in range [-1, 1).
func (r *RNG) NormalizedVector(dimensions int) []float32 {
	vec := make([]float32, dimensions)
	for i := range vec {
		vec[i] = r.rand.Float32()*2 - 1 // Range [-1, 1)
	}

	return vec
}
