package kmeans

import (
	"math"
	"math/rand"
	"sort"

	"github.com/hupe1980/vecgo/distance"
)

// TrainKMeans trains k centroids from the given vectors using Lloyd's algorithm.
// It returns the flattened centroids (k * dim).
func TrainKMeans(vectors []float32, dim int, k int, metric distance.Metric, maxIter int) ([]float32, error) {
	n := len(vectors) / dim
	if n < k {
		return nil, nil // Not enough vectors to cluster
	}

	centroids := make([]float32, k*dim)

	// Initialize centroids randomly from data points
	perm := rand.Perm(n)
	for i := 0; i < k; i++ {
		copy(centroids[i*dim:(i+1)*dim], vectors[perm[i]*dim:(perm[i]+1)*dim])
	}

	assignments := make([]int, n)
	counts := make([]int, k)
	sums := make([]float32, k*dim)

	distFunc, err := distance.Provider(metric)
	if err != nil {
		return nil, err
	}

	for iter := 0; iter < maxIter; iter++ {
		changed := false

		// Assignment step
		for i := 0; i < n; i++ {
			vec := vectors[i*dim : (i+1)*dim]
			bestCluster := -1
			minDist := float32(math.MaxFloat32)

			for j := 0; j < k; j++ {
				center := centroids[j*dim : (j+1)*dim]
				d := distFunc(vec, center)
				if d < minDist {
					minDist = d
					bestCluster = j
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

		// Update step
		for i := range sums {
			sums[i] = 0
		}
		for i := range counts {
			counts[i] = 0
		}

		for i := 0; i < n; i++ {
			cluster := assignments[i]
			vec := vectors[i*dim : (i+1)*dim]
			for d := 0; d < dim; d++ {
				sums[cluster*dim+d] += vec[d]
			}
			counts[cluster]++
		}

		for j := 0; j < k; j++ {
			if counts[j] > 0 {
				scale := 1.0 / float32(counts[j])
				for d := 0; d < dim; d++ {
					centroids[j*dim+d] = sums[j*dim+d] * scale
				}
			} else {
				// Re-initialize empty cluster with a random point
				// (Simple heuristic to avoid empty clusters)
				idx := rand.Intn(n)
				copy(centroids[j*dim:(j+1)*dim], vectors[idx*dim:(idx+1)*dim])
			}
		}
	}

	return centroids, nil
}

// AssignPartition finds the closest centroid for a vector.
func AssignPartition(vec []float32, centroids []float32, dim int, metric distance.Metric) (int, error) {
	k := len(centroids) / dim
	distFunc, err := distance.Provider(metric)
	if err != nil {
		return -1, err
	}

	bestCluster := -1
	minDist := float32(math.MaxFloat32)

	for j := 0; j < k; j++ {
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

// FindClosestCentroids returns the indices of the n closest centroids to the query vector.
func FindClosestCentroids(query []float32, centroids []float32, dim int, n int, metric distance.Metric) ([]int, error) {
	k := len(centroids) / dim
	if n > k {
		n = k
	}

	distFunc, err := distance.Provider(metric)
	if err != nil {
		return nil, err
	}

	dists := make([]centroidDist, k)
	for i := 0; i < k; i++ {
		center := centroids[i*dim : (i+1)*dim]
		d := distFunc(query, center)
		dists[i] = centroidDist{id: i, dist: d}
	}

	sort.Slice(dists, func(i, j int) bool {
		return dists[i].dist < dists[j].dist
	})

	result := make([]int, n)
	for i := 0; i < n; i++ {
		result[i] = dists[i].id
	}

	return result, nil
}
