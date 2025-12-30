package vecgo_bench_test

import (
	"context"
	"testing"

	"github.com/hupe1980/vecgo"
	"github.com/hupe1980/vecgo/index"
	"github.com/hupe1980/vecgo/testutil"
)

func BenchmarkHNSW(b *testing.B) {
	dim := 128
	size := 10000
	k := 10

	b.Run("Insert", func(b *testing.B) {
		db, err := vecgo.HNSW[int](dim).
			SquaredL2().
			Build()
		if err != nil {
			b.Fatal(err)
		}
		defer db.Close()

		rng := testutil.NewRNG(42)
		ctx := context.Background()

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			vec := rng.UnitVector(dim)
			_, err := db.Insert(ctx, vecgo.VectorWithData[int]{
				Vector: vec,
				Data:   i,
			})
			if err != nil {
				b.Fatal(err)
			}
		}
	})

	b.Run("Search", func(b *testing.B) {
		db, err := vecgo.HNSW[int](dim).
			SquaredL2().
			Build()
		if err != nil {
			b.Fatal(err)
		}
		defer db.Close()

		rng := testutil.NewRNG(42)
		ctx := context.Background()

		// Pre-fill
		vectors := make([][]float32, size)
		for i := 0; i < size; i++ {
			vec := rng.UnitVector(dim)
			vectors[i] = vec
			_, err := db.Insert(ctx, vecgo.VectorWithData[int]{
				Vector: vec,
				Data:   i,
			})
			if err != nil {
				b.Fatal(err)
			}
		}

		query := rng.UnitVector(dim)
		groundTruth := testutil.BruteForceSearch(vectors, query, k)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			res, err := db.Search(query).KNN(k).Execute(ctx)
			if err != nil {
				b.Fatal(err)
			}

			if i == 0 {
				approx := make([]index.SearchResult, len(res))
				for j, r := range res {
					approx[j] = r.SearchResult
				}
				recall := testutil.ComputeRecall(groundTruth, approx)
				b.ReportMetric(recall, "recall")
			}
		}
	})
}
