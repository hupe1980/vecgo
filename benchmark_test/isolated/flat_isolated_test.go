package isolated

import (
	"context"
	"testing"

	"github.com/hupe1980/vecgo/index/flat"
	"github.com/hupe1980/vecgo/testutil"
)

func BenchmarkFlatIsolatedInsert(b *testing.B) {
	dim := 128

	b.Run("Direct", func(b *testing.B) {
		idx, err := flat.New(func(o *flat.Options) {
			o.Dimension = dim
		})
		if err != nil {
			b.Fatal(err)
		}

		rng := testutil.NewRNG(42)
		ctx := context.Background()

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			vec := rng.UnitVector(dim)
			if _, err := idx.Insert(ctx, vec); err != nil {
				b.Fatal(err)
			}
		}
	})
}

func BenchmarkFlatIsolatedSearch(b *testing.B) {
	dim := 128
	size := 10000
	k := 10

	idx, err := flat.New(func(o *flat.Options) {
		o.Dimension = dim
	})
	if err != nil {
		b.Fatal(err)
	}

	rng := testutil.NewRNG(42)
	ctx := context.Background()
	vectors := make([][]float32, size)

	// Pre-fill
	for i := 0; i < size; i++ {
		vec := rng.UnitVector(dim)
		vectors[i] = vec
		if _, err := idx.Insert(ctx, vec); err != nil {
			b.Fatal(err)
		}
	}

	query := rng.UnitVector(dim)
	groundTruth := testutil.BruteForceSearch(vectors, query, k)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		results, err := idx.KNNSearch(ctx, query, k, nil)
		if err != nil {
			b.Fatal(err)
		}

		if i == 0 {
			recall := testutil.ComputeRecall(groundTruth, toTestUtilResults(results))
			b.ReportMetric(recall, "recall")
		}
	}
}
