package isolated

import (
	"context"
	"testing"

	"github.com/hupe1980/vecgo/index/hnsw"
	"github.com/hupe1980/vecgo/testutil"
)

// BenchmarkHNSWIsolatedInsert benchmarks HNSW insert directly, bypassing the Engine/Tx/MemTable.
func BenchmarkHNSWIsolatedInsert(b *testing.B) {
	dim := 128

	b.Run("Direct", func(b *testing.B) {
		idx, err := hnsw.New(func(o *hnsw.Options) {
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
			// HNSW requires sequential IDs usually, or we manage them.
			// HNSW.Insert takes (ctx, id, vector)
			if err := idx.ApplyInsert(ctx, uint64(i), vec); err != nil {
				b.Fatal(err)
			}
		}
	})
}

// BenchmarkHNSWIsolatedSearch benchmarks HNSW search directly.
func BenchmarkHNSWIsolatedSearch(b *testing.B) {
	dim := 128
	size := 10000
	k := 10

	idx, err := hnsw.New(func(o *hnsw.Options) {
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
		if err := idx.ApplyInsert(ctx, uint64(i), vec); err != nil {
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
			recall := testutil.ComputeRecall(groundTruth, results)
			b.ReportMetric(recall, "recall")
		}
	}
}
