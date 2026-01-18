package kmeans

import (
	"context"
	"math"
	"math/rand"
	"slices"

	"github.com/hupe1980/vecgo/distance"
	"github.com/hupe1980/vecgo/internal/simd"
)

// TrainKMeans trains k centroids from the given vectors using Lloyd's algorithm.
// It returns the flattened centroids (k * dim).
// The context can be used for cancellation of long-running training.
func TrainKMeans(ctx context.Context, vectors []float32, dim int, k int, metric distance.Metric, maxIter int) ([]float32, error) {
	n := len(vectors) / dim
	if n < k {
		return nil, nil // Not enough vectors to cluster
	}

	centroids := make([]float32, k*dim)

	// Initialize centroids randomly from data points (k-means++)
	perm := rand.Perm(n)
	for i := range k {
		copy(centroids[i*dim:(i+1)*dim], vectors[perm[i]*dim:(perm[i]+1)*dim])
	}

	assignments := make([]int, n)
	counts := make([]int, k)
	sums := make([]float32, k*dim)

	// Pre-allocate distance buffer for SIMD batch computation
	dists := make([]float32, k)

	// Get distance function once
	distFunc, err := distance.Provider(metric)
	if err != nil {
		return nil, err
	}

	// Determine if we can use SIMD batch
	useSIMDBatchL2 := metric == distance.MetricL2
	useSIMDBatchDot := metric == distance.MetricDot || metric == distance.MetricCosine

	for range maxIter {
		// Check for cancellation
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}

		changed := false

		// Assignment step
		for i := 0; i < n; i++ {
			vec := vectors[i*dim : (i+1)*dim]
			var bestCluster int

			if useSIMDBatchL2 {
				// Use SIMD batch distance for L2 (lower is better)
				simd.SquaredL2Batch(vec, centroids, dim, dists)
				bestCluster = 0
				minDist := dists[0]
				for j := 1; j < k; j++ {
					if dists[j] < minDist {
						minDist = dists[j]
						bestCluster = j
					}
				}
			} else if useSIMDBatchDot {
				// Use SIMD batch dot product for Cosine/Dot (higher is better)
				simd.DotBatch(vec, centroids, dim, dists)
				bestCluster = 0
				maxDot := dists[0]
				for j := 1; j < k; j++ {
					if dists[j] > maxDot {
						maxDot = dists[j]
						bestCluster = j
					}
				}
			} else {
				// Fallback for other metrics
				bestCluster = 0
				minDist := float32(math.MaxFloat32)
				for j := range k {
					center := centroids[j*dim : (j+1)*dim]
					d := distFunc(vec, center)
					if d < minDist {
						minDist = d
						bestCluster = j
					}
				}
			}

			if assignments[i] != bestCluster {
				assignments[i] = bestCluster
				changed = true
			}
		}

		if !changed {
			break
		}

		// Update step - clear accumulators using Go 1.21+ builtin
		clear(sums)
		clear(counts)

		// Accumulate sums and counts
		for i := range n {
			cluster := assignments[i]
			vec := vectors[i*dim : (i+1)*dim]
			offset := cluster * dim
			for d := range dim {
				sums[offset+d] += vec[d]
			}
			counts[cluster]++
		}

		// Compute new centroids
		for j := range k {
			if counts[j] > 0 {
				scale := 1.0 / float32(counts[j])
				offset := j * dim
				for d := range dim {
					centroids[offset+d] = sums[offset+d] * scale
				}
			} else {
				// Re-initialize empty cluster with a random point
				idx := rand.Intn(n)
				copy(centroids[j*dim:(j+1)*dim], vectors[idx*dim:(idx+1)*dim])
			}
		}
	}

	return centroids, nil
}

// AssignPartition finds the closest centroid for a vector.
// For high-throughput batch assignment, use AssignPartitionBatch instead.
func AssignPartition(vec []float32, centroids []float32, dim int, metric distance.Metric) (int, error) {
	k := len(centroids) / dim

	// Use SIMD batch for L2 metric (lower is better)
	if metric == distance.MetricL2 {
		dists := make([]float32, k)
		simd.SquaredL2Batch(vec, centroids, dim, dists)

		bestCluster := 0
		minDist := dists[0]
		for j := 1; j < k; j++ {
			if dists[j] < minDist {
				minDist = dists[j]
				bestCluster = j
			}
		}
		return bestCluster, nil
	}

	// Use SIMD batch for Dot/Cosine metrics (higher is better)
	if metric == distance.MetricDot || metric == distance.MetricCosine {
		dists := make([]float32, k)
		simd.DotBatch(vec, centroids, dim, dists)

		bestCluster := 0
		maxDot := dists[0]
		for j := 1; j < k; j++ {
			if dists[j] > maxDot {
				maxDot = dists[j]
				bestCluster = j
			}
		}
		return bestCluster, nil
	}

	// Fallback for other metrics
	distFunc, err := distance.Provider(metric)
	if err != nil {
		return -1, err
	}

	bestCluster := 0
	minDist := float32(math.MaxFloat32)

	for j := range k {
		center := centroids[j*dim : (j+1)*dim]
		d := distFunc(vec, center)
		if d < minDist {
			minDist = d
			bestCluster = j
		}
	}

	return bestCluster, nil
}

type centroidDist struct {
	id   int
	dist float32
}

// cmpCentroidDistByDist compares centroidDist by distance ascending.
// Package-level function to avoid closure allocation.
func cmpCentroidDistByDist(a, b centroidDist) int {
	if a.dist < b.dist {
		return -1
	}
	if a.dist > b.dist {
		return 1
	}
	return 0
}

// FindClosestCentroids returns the indices of the n closest centroids to the query vector.
// This function is on the search hot path when partitioning is enabled.
func FindClosestCentroids(query []float32, centroids []float32, dim int, n int, metric distance.Metric) ([]int, error) {
	k := len(centroids) / dim
	if n > k {
		n = k
	}

	// Single allocation for both distances and result tracking
	dists := make([]centroidDist, k)

	// Use SIMD batch for L2 metric (lower is better)
	if metric == distance.MetricL2 {
		distValues := make([]float32, k)
		simd.SquaredL2Batch(query, centroids, dim, distValues)
		for i := range k {
			dists[i] = centroidDist{id: i, dist: distValues[i]}
		}
	} else if metric == distance.MetricDot || metric == distance.MetricCosine {
		// Use SIMD batch for Dot/Cosine metrics (higher is better, negate for sorting)
		distValues := make([]float32, k)
		simd.DotBatch(query, centroids, dim, distValues)
		for i := range k {
			// Negate so that higher dot products sort first (lower negative value)
			dists[i] = centroidDist{id: i, dist: -distValues[i]}
		}
	} else {
		// Fallback for other metrics
		distFunc, err := distance.Provider(metric)
		if err != nil {
			return nil, err
		}
		for i := range k {
			center := centroids[i*dim : (i+1)*dim]
			d := distFunc(query, center)
			dists[i] = centroidDist{id: i, dist: d}
		}
	}

	// For small n, use partial sort (selection) instead of full sort
	if n <= k/4 && n < 16 {
		// Selection algorithm - O(n*k) but avoids full sort overhead
		result := make([]int, n)
		for i := 0; i < n; i++ {
			minIdx := i
			for j := i + 1; j < k; j++ {
				if dists[j].dist < dists[minIdx].dist {
					minIdx = j
				}
			}
			dists[i], dists[minIdx] = dists[minIdx], dists[i]
			result[i] = dists[i].id
		}
		return result, nil
	}

	// Full sort for larger n values
	slices.SortFunc(dists, cmpCentroidDistByDist)

	result := make([]int, n)
	for i := 0; i < n; i++ {
		result[i] = dists[i].id
	}

	return result, nil
}
