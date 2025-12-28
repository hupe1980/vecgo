package vecgo_bench_test

import (
	"context"
	"fmt"

	"github.com/hupe1980/vecgo"
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
func computeRecall[T any](groundTruth, approximate []vecgo.SearchResult[T]) float64 {
	if len(groundTruth) == 0 {
		return 1.0
	}

	// Build set of ground truth IDs
	truthSet := make(map[uint32]struct{}, len(groundTruth))
	for _, r := range groundTruth {
		truthSet[r.ID] = struct{}{}
	}

	// Count hits
	hits := 0
	for _, r := range approximate {
		if _, ok := truthSet[r.ID]; ok {
			hits++
		}
	}

	return float64(hits) / float64(len(groundTruth))
}

// groundTruthSearch performs exact search using Flat index for recall computation.
func groundTruthSearch(ctx context.Context, vectors [][]float32, query []float32, k int) []vecgo.SearchResult[int] {
	dim := len(query)
	db, err := vecgo.Flat[int](dim).SquaredL2().Build()
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
