package vecgo_bench_test

import (
	"context"
	"fmt"

	"github.com/hupe1980/vecgo"
	"github.com/hupe1980/vecgo/index"
	"github.com/hupe1980/vecgo/testutil"
)

var (
	// rng is a seeded random number generator for reproducible benchmarks
	rng = testutil.NewRNG(42)

	// categories for metadata testing
	categories = []string{
		"technology",
		"science",
		"business",
		"sports",
		"entertainment",
	}
)

// randomVector generates a random float32 vector of given dimension using seeded RNG.
// Values are in range [-1, 1).
func randomVector(dim int) []float32 {
	return rng.NormalizedVector(dim)
}

// computeRecall computes recall@k by comparing approximate results against ground truth.
// Ground truth is obtained from exact (Flat) search.
// Returns recall as a float64 between 0.0 and 1.0.
// Note: For post-filtered searches, fewer results may be returned than requested.
// Recall is computed as: (hits in approximate) / min(k, len(approximate), len(groundTruth))
func computeRecall[T any](groundTruth, approximate []vecgo.SearchResult[T]) float64 {
	if len(groundTruth) == 0 || len(approximate) == 0 {
		if len(groundTruth) == 0 && len(approximate) == 0 {
			return 1.0 // Both empty = perfect match
		}
		return 0.0
	}

	// Build set of ground truth IDs (only consider as many as we got back)
	k := len(approximate) // Use actual returned count as the target
	if k > len(groundTruth) {
		k = len(groundTruth)
	}

	truthSet := make(map[uint32]struct{}, k)
	for i := 0; i < k; i++ {
		truthSet[groundTruth[i].ID] = struct{}{}
	}

	// Count hits
	hits := 0
	for _, r := range approximate {
		if _, ok := truthSet[r.ID]; ok {
			hits++
		}
	}

	return float64(hits) / float64(k)
}

// groundTruthSearch performs exact search using Flat index for recall computation.
func groundTruthSearch(ctx context.Context, vectors [][]float32, query []float32, k int) []vecgo.SearchResult[int] {
	return groundTruthSearchWithDistance(ctx, vectors, query, k, index.DistanceTypeSquaredL2)
}

// groundTruthSearchWithDistance performs exact search using Flat index with specified distance type.
func groundTruthSearchWithDistance(ctx context.Context, vectors [][]float32, query []float32, k int, distType index.DistanceType) []vecgo.SearchResult[int] {
	dim := len(query)
	var builder *vecgo.FlatBuilder[int]
	switch distType {
	case index.DistanceTypeSquaredL2:
		builder = vecgo.Flat[int](dim).SquaredL2()
	case index.DistanceTypeCosine:
		builder = vecgo.Flat[int](dim).Cosine()
	case index.DistanceTypeDotProduct:
		builder = vecgo.Flat[int](dim).DotProduct()
	default:
		builder = vecgo.Flat[int](dim).SquaredL2()
	}

	db, err := builder.Build()
	if err != nil {
		return nil
	}
	defer db.Close()

	batch := make([]vecgo.VectorWithData[int], len(vectors))
	for i, v := range vectors {
		batch[i] = vecgo.VectorWithData[int]{Vector: v, Data: i}
	}
	db.BatchInsert(ctx, batch)

	results, _ := db.Search(query).KNN(k).Execute(ctx)
	return results
}

// groundTruthSearchFiltered performs exact search with filtering using original IDs.
// vectorsWithIDs maps original ID -> vector for filtered vectors only.
func groundTruthSearchFiltered(ctx context.Context, vectorsWithIDs map[uint32][]float32, query []float32, k int) []vecgo.SearchResult[int] {
	if len(vectorsWithIDs) == 0 {
		return nil
	}

	// Compute distances directly and sort to get ground truth
	type idDist struct {
		id   uint32
		dist float32
	}
	distances := make([]idDist, 0, len(vectorsWithIDs))
	for id, vec := range vectorsWithIDs {
		dist := squaredL2Distance(query, vec)
		distances = append(distances, idDist{id: id, dist: dist})
	}

	// Sort by distance (ascending)
	for i := 0; i < len(distances)-1; i++ {
		for j := i + 1; j < len(distances); j++ {
			if distances[j].dist < distances[i].dist {
				distances[i], distances[j] = distances[j], distances[i]
			}
		}
	}

	// Take top k
	if k > len(distances) {
		k = len(distances)
	}
	results := make([]vecgo.SearchResult[int], k)
	for i := 0; i < k; i++ {
		results[i] = vecgo.SearchResult[int]{
			SearchResult: index.SearchResult{
				ID:       distances[i].id,
				Distance: distances[i].dist,
			},
			Data: int(distances[i].id),
		}
	}
	return results
}

// squaredL2Distance computes squared L2 distance between two vectors
func squaredL2Distance(a, b []float32) float32 {
	var sum float32
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return sum
}

// formatDim formats dimension for benchmark names
func formatDim(dim int) string {
	return fmt.Sprintf("dim%d", dim)
}

// formatCount formats count for benchmark names
func formatCount(count int) string {
	if count >= 1000000 {
		return fmt.Sprintf("%dM", count/1000000)
	}
	if count >= 1000 {
		return fmt.Sprintf("%dK", count/1000)
	}
	return fmt.Sprintf("%d", count)
}

// formatPercent formats selectivity percentage
func formatPercent(selectivity float64) string {
	return fmt.Sprintf("%.0f%%", selectivity*100)
}
