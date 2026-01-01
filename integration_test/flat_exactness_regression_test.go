package vecgo_test

import (
	"context"
	"testing"

	"github.com/hupe1980/vecgo"
	"github.com/hupe1980/vecgo/testutil"
)

// Regression test for a previously observed issue where Flat (exact) search
// would return <100% recall in some runs after optimizations.
//
// The most sensitive case is: insert enough vectors to trigger background
// flushing, then immediately search while the system may still be publishing
// newly-written vectors.
func TestFlatExactSearch_Is100PercentRecall(t *testing.T) {
	t.Parallel()

	const (
		dim  = 128
		size = 10000
		k    = 10
	)

	db, err := vecgo.Flat[int](dim).SquaredL2().Build()
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })

	rng := testutil.NewRNG(42)
	ctx := context.Background()

	vectors := make([][]float32, size)
	for i := 0; i < size; i++ {
		vec := rng.UnitVector(dim)
		vectors[i] = vec
		if _, err := db.Insert(ctx, vecgo.VectorWithData[int]{Vector: vec, Data: i}); err != nil {
			t.Fatalf("Insert(%d): %v", i, err)
		}
	}

	query := rng.UnitVector(dim)
	groundTruth := testutil.BruteForceSearch(vectors, query, k)

	// Repeat to catch any nondeterminism; this should be stable for an exact index.
	for iter := 0; iter < 50; iter++ {
		res, err := db.Search(query).KNN(k).Execute(ctx)
		if err != nil {
			t.Fatalf("Search(iter=%d): %v", iter, err)
		}
		approx := make([]testutil.SearchResult, len(res))
		for j, r := range res {
			approx[j] = testutil.SearchResult{ID: r.SearchResult.ID, Distance: r.SearchResult.Distance}
		}
		recall := testutil.ComputeRecall(groundTruth, approx)
		if recall != 1.0 {
			t.Fatalf("recall=%0.4f (iter=%d), want 1.0", recall, iter)
		}
	}
}
