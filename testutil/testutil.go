package testutil

import (
	"math"
	"math/rand"
	"sort"
	"sync"
)

// SearchResult represents a search result.
type SearchResult struct {
	ID       uint64
	Distance float32
}

// RNG struct encapsulates the random number generator and seed.
// It is thread-safe.
type RNG struct {
	rand *rand.Rand
	seed int64
	mu   sync.Mutex
}

// NewRNG creates a new RNG instance with the specified seed.
func NewRNG(seed int64) *RNG {
	return &RNG{
		rand: rand.New(rand.NewSource(seed)), // nolint gosec
		seed: seed,
	}
}

// Reset resets the RNG to its initial seed.
func (r *RNG) Reset() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.rand.Seed(r.seed)
}

// Seed returns the initial seed.
func (r *RNG) Seed() int64 {
	return r.seed
}

// Intn returns a non-negative pseudo-random number in [0,n).
func (r *RNG) Intn(n int) int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.rand.Intn(n)
}

// Float32 returns, as a float32, a pseudo-random number in [0.0,1.0).
func (r *RNG) Float32() float32 {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.rand.Float32()
}

// UniformVectors generates random vectors with values in range [0, 1).
// Uses a single backing array for efficiency.
func (r *RNG) UniformVectors(num int, dimensions int) [][]float32 {
	r.mu.Lock()
	defer r.mu.Unlock()

	data := make([]float32, num*dimensions)
	vectors := make([][]float32, num)

	for i := range num {
		vec := data[i*dimensions : (i+1)*dimensions]
		for j := range vec {
			vec[j] = r.rand.Float32()
		}
		vectors[i] = vec
	}

	return vectors
}

// UniformRangeVectors generates random vectors with values in range [-1, 1).
// This replaces the old "GenerateNormalizedVectors" which was misnamed.
func (r *RNG) UniformRangeVectors(num int, dimensions int) [][]float32 {
	r.mu.Lock()
	defer r.mu.Unlock()

	data := make([]float32, num*dimensions)
	vectors := make([][]float32, num)

	for i := range num {
		vec := data[i*dimensions : (i+1)*dimensions]
		for j := range vec {
			vec[j] = r.rand.Float32()*2 - 1
		}
		vectors[i] = vec
	}

	return vectors
}

// GaussianVectors generates random vectors with values from a standard normal distribution.
func (r *RNG) GaussianVectors(num int, dimensions int) [][]float32 {
	r.mu.Lock()
	defer r.mu.Unlock()

	data := make([]float32, num*dimensions)
	vectors := make([][]float32, num)

	for i := range num {
		vec := data[i*dimensions : (i+1)*dimensions]
		for j := range vec {
			vec[j] = float32(r.rand.NormFloat64())
		}
		vectors[i] = vec
	}

	return vectors
}

// UnitVectors generates L2-normalized random vectors (on the hypersphere).
// Uses Gaussian distribution for uniform distribution on the sphere.
// Essential for Cosine/DotProduct benchmarks and HNSW/DiskANN graph quality.
func (r *RNG) UnitVectors(num int, dimensions int) [][]float32 {
	r.mu.Lock()
	defer r.mu.Unlock()

	data := make([]float32, num*dimensions)
	vectors := make([][]float32, num)

	for i := range num {
		vec := data[i*dimensions : (i+1)*dimensions]
		var norm float64
		for j := range vec {
			v := r.rand.NormFloat64()
			vec[j] = float32(v)
			norm += v * v
		}

		if norm == 0 {
			norm = 1 // Avoid division by zero, though unlikely with floats
		}

		invNorm := float32(1.0 / math.Sqrt(norm))
		for j := range vec {
			vec[j] *= invNorm
		}
		vectors[i] = vec
	}

	return vectors
}

// UnitVector generates a single L2-normalized random vector.
func (r *RNG) UnitVector(dimensions int) []float32 {
	r.mu.Lock()
	defer r.mu.Unlock()

	vec := make([]float32, dimensions)
	var norm float64
	for j := range vec {
		v := r.rand.NormFloat64()
		vec[j] = float32(v)
		norm += v * v
	}

	if norm == 0 {
		norm = 1
	}

	invNorm := float32(1.0 / math.Sqrt(norm))
	for j := range vec {
		vec[j] *= invNorm
	}
	return vec
}

// ClusteredVectors generates vectors clustered around random centroids.
// Useful for testing ANN index performance on non-uniform data.
func (r *RNG) ClusteredVectors(num, dim, clusters int, spread float32) [][]float32 {
	// Generate centroids (unit vectors) - calling internal method which locks, so we need to be careful if we were holding lock.
	// But UnitVectors acquires lock, so we are fine as long as we don't hold lock here yet.
	centroids := r.UnitVectors(clusters, dim)

	r.mu.Lock()
	defer r.mu.Unlock()

	data := make([]float32, num*dim)
	vectors := make([][]float32, num)

	for i := range num {
		centroid := centroids[i%clusters]
		vec := data[i*dim : (i+1)*dim]

		for j := range dim {
			// Add Gaussian noise to centroid
			vec[j] = centroid[j] + float32(r.rand.NormFloat64())*spread
		}
		vectors[i] = vec
	}

	return vectors
}

// ComputeRecall computes recall@k by comparing approximate results against ground truth.
func ComputeRecall(groundTruth, approximate []SearchResult) float64 {
	if len(groundTruth) == 0 || len(approximate) == 0 {
		if len(groundTruth) == 0 && len(approximate) == 0 {
			return 1.0
		}
		return 0.0
	}

	k := min(len(approximate), len(groundTruth))

	truthSet := make(map[uint64]struct{}, k)
	for i := range k {
		truthSet[groundTruth[i].ID] = struct{}{}
	}

	hits := 0
	for _, r := range approximate {
		if _, ok := truthSet[r.ID]; ok {
			hits++
		}
	}

	return float64(hits) / float64(k)
}

// BruteForceSearch performs exact search for ground truth.
func BruteForceSearch(vectors [][]float32, query []float32, k int) []SearchResult {
	type result struct {
		id   uint64
		dist float32
	}

	results := make([]result, len(vectors))

	for i, v := range vectors {
		d := squaredL2(query, v)
		results[i] = result{id: uint64(i), dist: d} //nolint:gosec
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].dist < results[j].dist
	})

	if len(results) > k {
		results = results[:k]
	}

	out := make([]SearchResult, len(results))
	for i, r := range results {
		out[i] = SearchResult{ID: r.id, Distance: r.dist}
	}
	return out
}

func squaredL2(a, b []float32) float32 {
	var sum float32
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return sum
}
