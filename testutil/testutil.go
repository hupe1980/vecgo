package testutil

import (
	"math"
	"math/rand"
	"sort"
	"sync"

	"github.com/hupe1980/vecgo/internal/simd"
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
		rand: rand.New(rand.NewSource(seed)),
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

// Uint64 returns a pseudo-random uint64.
func (r *RNG) Uint64() uint64 {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.rand.Uint64()
}

// Float32 returns, as a float32, a pseudo-random number in [0.0,1.0).
func (r *RNG) Float32() float32 {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.rand.Float32()
}

// FillUniform fills dst with random values in range [0, 1).
// Locks only once per call (preferred over calling Float32 in a loop).
func (r *RNG) FillUniform(dst []float32) {
	r.mu.Lock()
	defer r.mu.Unlock()
	for i := range dst {
		dst[i] = r.rand.Float32()
	}
}

// FillUniformRange fills dst with random values in range [minVal, maxVal).
func (r *RNG) FillUniformRange(dst []float32, minVal, maxVal float32) {
	r.mu.Lock()
	defer r.mu.Unlock()
	span := maxVal - minVal
	for i := range dst {
		dst[i] = minVal + r.rand.Float32()*span
	}
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
		simd.ScaleInPlace(vec, invNorm)
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
	simd.ScaleInPlace(vec, invNorm)
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

// Zipf returns a Zipfian-distributed value in [0, n).
// Uses Zipf's law: P(k) ∝ 1/k^s where s is the skew parameter.
// s=1.0 gives standard Zipf, s=1.5 gives heavy-tail (80/20 rule).
// This is how real-world data is distributed (power law).
func (r *RNG) Zipf(n int, s float64) int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.zipfLocked(n, s)
}

// zipfLocked is the internal implementation (caller must hold lock).
func (r *RNG) zipfLocked(n int, s float64) int {
	// Use rejection sampling from uniform distribution
	// This is mathematically correct for Zipf distribution
	if n <= 1 {
		return 0
	}

	// Compute normalization constant (harmonic number with exponent s)
	var hns float64
	for i := 1; i <= n; i++ {
		hns += 1.0 / math.Pow(float64(i), s)
	}

	// Sample from uniform and use inverse transform
	u := r.rand.Float64() * hns
	var cumulative float64
	for k := 1; k <= n; k++ {
		cumulative += 1.0 / math.Pow(float64(k), s)
		if u <= cumulative {
			return k - 1 // 0-indexed
		}
	}

	return n - 1
}

// ZipfBuckets generates n bucket assignments with Zipfian distribution.
// Returns slice where ~20% of buckets contain ~80% of values (when s=1.5).
func (r *RNG) ZipfBuckets(n, bucketCount int, s float64) []int64 {
	r.mu.Lock()
	defer r.mu.Unlock()

	buckets := make([]int64, n)
	for i := range n {
		buckets[i] = int64(r.zipfLocked(bucketCount, s))
	}

	return buckets
}

// ClusteredVectorsWithBuckets generates vectors where each bucket has its own centroid.
// This creates correlation between metadata (bucket) and vector space.
// Vectors in the same bucket are close together in vector space.
// noise controls how much deviation from centroid (0.1 = tight clusters, 0.3 = loose).
func (r *RNG) ClusteredVectorsWithBuckets(num, dim, bucketCount int, buckets []int64, noise float32) [][]float32 {
	// Generate one centroid per bucket
	centroids := r.UnitVectors(bucketCount, dim)

	r.mu.Lock()
	defer r.mu.Unlock()

	data := make([]float32, num*dim)
	vectors := make([][]float32, num)

	for i := range num {
		bucket := buckets[i]
		centroid := centroids[bucket]
		vec := data[i*dim : (i+1)*dim]

		// Vector = centroid + Gaussian noise
		for j := range dim {
			vec[j] = centroid[j] + float32(r.rand.NormFloat64())*noise
		}
		vectors[i] = vec
	}

	return vectors
}

// SparseMetadata generates metadata with missing fields.
// missingRate is the probability that a field is missing (0.3 = 30% missing).
func (r *RNG) SparseMetadata(n int, missingRate float64) []bool {
	r.mu.Lock()
	defer r.mu.Unlock()

	present := make([]bool, n)
	for i := range n {
		present[i] = r.rand.Float64() >= missingRate
	}

	return present
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

// ============================================================================
// Adversarial Distribution Generators
// ============================================================================

// SegmentLocalSkewBuckets generates bucket assignments where:
// - Globally uniform distribution (each bucket has ~equal total count)
// - But within each "segment" (chunk of numVecs/numSegments), one bucket dominates
//
// This creates the "Snowflake killer" scenario where:
// - Global stats say selectivity = 1%
// - Per-segment stats show selectivity = 90%
//
// Use this to test whether the planner trusts global stats blindly.
func (r *RNG) SegmentLocalSkewBuckets(numVecs, bucketCount, numSegments int, localDominance float64) []int64 {
	r.mu.Lock()
	defer r.mu.Unlock()

	buckets := make([]int64, numVecs)
	segmentSize := numVecs / numSegments
	if segmentSize < 1 {
		segmentSize = 1
	}

	for i := range numVecs {
		segmentIdx := i / segmentSize
		if segmentIdx >= numSegments {
			segmentIdx = numSegments - 1
		}

		// Each segment has a "dominant" bucket
		dominantBucket := int64(segmentIdx % bucketCount)

		// With probability localDominance, assign to dominant bucket
		// Otherwise, assign uniformly to other buckets
		if r.rand.Float64() < localDominance {
			buckets[i] = dominantBucket
		} else {
			// Assign to a random non-dominant bucket
			other := int64(r.rand.Intn(bucketCount - 1))
			if other >= dominantBucket {
				other++
			}
			buckets[i] = other
		}
	}

	return buckets
}

// CorrelatedVectorsWithBuckets generates vectors where metadata predicts vector similarity.
// Unlike random assignment, vectors with the same bucket value are CLOSE in vector space.
// This is realistic: "category=shoes" → vectors cluster in embedding space.
//
// What this tests:
// - Whether filtering helps graph pruning (vectors in same bucket are neighbors)
// - Whether predicate-aware HNSW traversal is beneficial
// - Realistic embedding behavior (semantic similarity ↔ metadata correlation)
func (r *RNG) CorrelatedVectorsWithBuckets(numVecs, dim, bucketCount int, buckets []int64, clusterTightness float32) [][]float32 {
	// This is just an alias to ClusteredVectorsWithBuckets with clearer semantics
	// The correlation strength is controlled by clusterTightness (lower = tighter clusters = more correlation)
	return r.ClusteredVectorsWithBuckets(numVecs, dim, bucketCount, buckets, clusterTightness)
}

// BooleanAdversarialBuckets generates bucket assignments designed to stress-test
// bitmap operations and filter evaluation.
//
// Creates a bimodal distribution where buckets are either:
// - Very selective (only a few vectors)
// - Very dense (many vectors)
//
// This tests:
// - Roaring bitmap container promotion (array → bitmap → runs)
// - Allocation behavior under alternating selectivity
// - Whether the planner correctly handles mixed selectivity per-field
func (r *RNG) BooleanAdversarialBuckets(numVecs, bucketCount int) ([]int64, []int64) {
	r.mu.Lock()
	defer r.mu.Unlock()

	bucketA := make([]int64, numVecs) // Primary bucket (bimodal)
	bucketB := make([]int64, numVecs) // Secondary bucket (interleaved for compound filters)

	// Create bimodal distribution for bucket A:
	// - Buckets 0-9: ~1% each (selective)
	// - Buckets 10-19: ~9% each (dense)
	selectiveBuckets := bucketCount / 10
	if selectiveBuckets < 1 {
		selectiveBuckets = 1
	}

	for i := range numVecs {
		// 10% chance of selective bucket, 90% dense
		if r.rand.Float64() < 0.10 {
			bucketA[i] = int64(r.rand.Intn(selectiveBuckets))
		} else {
			bucketA[i] = int64(selectiveBuckets + r.rand.Intn(bucketCount-selectiveBuckets))
		}

		// Bucket B: interleaved pattern to create complex (A OR B) scenarios
		// Half correlated with A, half anti-correlated
		if r.rand.Float64() < 0.5 {
			bucketB[i] = bucketA[i] // Correlated: (A=x OR B=x) has high overlap
		} else {
			// Anti-correlated: random bucket different from A
			other := int64(r.rand.Intn(bucketCount - 1))
			if other >= bucketA[i] {
				other++
			}
			bucketB[i] = other
		}
	}

	return bucketA, bucketB
}

// BruteForceSearch performs exact search for ground truth.
func BruteForceSearch(vectors [][]float32, query []float32, k int) []SearchResult {
	type result struct {
		id   uint64
		dist float32
	}

	results := make([]result, len(vectors))

	for i, v := range vectors {
		d := simd.SquaredL2(query, v)
		results[i] = result{id: uint64(i), dist: d}
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
