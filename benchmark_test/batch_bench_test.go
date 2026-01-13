package benchmark_test

import (
	"context"
	"testing"

	"github.com/hupe1980/vecgo"
	"github.com/hupe1980/vecgo/model"

	"github.com/hupe1980/vecgo/testutil"
)

func BenchmarkBatchInsert(b *testing.B) {
	dim := 128
	batchSize := 100

	b.Run("Sequential", func(b *testing.B) {
		dir := b.TempDir()
		e, _ := vecgo.Open(vecgo.Local(dir), vecgo.Create(dim, vecgo.MetricL2))
		defer e.Close()

		vec := make([]float32, dim)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			e.Insert(vec, nil, nil)
		}
	})

	b.Run("Batch", func(b *testing.B) {
		dir := b.TempDir()
		e, _ := vecgo.Open(vecgo.Local(dir), vecgo.Create(dim, vecgo.MetricL2))
		defer e.Close()

		vec := make([]float32, dim)
		vectors := make([][]float32, batchSize)
		for i := 0; i < batchSize; i++ {
			vectors[i] = vec
		}

		b.ResetTimer()
		for i := 0; i < b.N; i += batchSize {
			// Handle last partial batch if b.N is not multiple of batchSize
			count := batchSize
			if i+count > b.N {
				count = b.N - i
			}
			e.BatchInsert(vectors[:count], nil, nil)
		}
	})
}

func BenchmarkBatchSearch(b *testing.B) {
	dim := 128
	numVecs := 1000
	batchSize := 10

	dir := b.TempDir()
	e, _ := vecgo.Open(vecgo.Local(dir), vecgo.Create(dim, vecgo.MetricL2))
	defer e.Close()

	rng := testutil.NewRNG(1)
	data := make([][]float32, numVecs)
	pks := make([]model.ID, numVecs)
	for i := 0; i < numVecs; i++ {
		vec := make([]float32, dim)
		rng.FillUniform(vec)
		data[i] = vec
		id, _ := e.Insert(vec, nil, nil)
		pks[i] = id
	}

	ctx := context.Background()
	queries := make([][]float32, batchSize)
	for i := 0; i < batchSize; i++ {
		queries[i] = make([]float32, dim)
		rng.FillUniform(queries[i])
	}

	b.ResetTimer()

	b.Run("Sequential", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			for _, q := range queries {
				e.Search(ctx, q, 10)
			}
		}
		b.StopTimer()
		var sum float64
		for _, q := range queries {
			truth := exactTopK_L2_WithIDs(data, pks, q, 10)
			res, _ := e.Search(ctx, q, 10)
			sum += recallAtK(res, truth)
		}
		b.ReportMetric(sum/float64(len(queries)), "recall@10")
	})

	b.Run("Batch", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			e.BatchSearch(ctx, queries, 10)
		}
		b.StopTimer()
		res, _ := e.BatchSearch(ctx, queries, 10)
		var sum float64
		for i, q := range queries {
			truth := exactTopK_L2_WithIDs(data, pks, q, 10)
			sum += recallAtK(res[i], truth)
		}
		b.ReportMetric(sum/float64(len(queries)), "recall@10")
	})
}
