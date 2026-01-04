package benchmark_test

import (
	"context"
	"testing"

	"github.com/hupe1980/vecgo/distance"
	"github.com/hupe1980/vecgo/engine"
	"github.com/hupe1980/vecgo/model"
	"github.com/hupe1980/vecgo/testutil"
)

func BenchmarkBatchInsert(b *testing.B) {
	dim := 128
	batchSize := 100

	b.Run("Sequential", func(b *testing.B) {
		dir := b.TempDir()
		e, _ := engine.Open(dir, dim, distance.MetricL2)
		defer e.Close()

		vec := make([]float32, dim)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			e.Insert(model.PrimaryKey(i), vec, nil, nil)
		}
	})

	b.Run("Batch", func(b *testing.B) {
		dir := b.TempDir()
		e, _ := engine.Open(dir, dim, distance.MetricL2)
		defer e.Close()

		vec := make([]float32, dim)
		records := make([]model.Record, batchSize)
		for i := 0; i < batchSize; i++ {
			records[i] = model.Record{Vector: vec}
		}

		b.ResetTimer()
		for i := 0; i < b.N; i += batchSize {
			// Update PKs
			for j := 0; j < batchSize; j++ {
				records[j].PK = model.PrimaryKey(i + j)
			}
			// Handle last partial batch if b.N is not multiple of batchSize
			count := batchSize
			if i+count > b.N {
				count = b.N - i
			}
			e.BatchInsert(records[:count])
		}
	})
}

func BenchmarkBatchSearch(b *testing.B) {
	dim := 128
	numVecs := 1000
	batchSize := 10

	dir := b.TempDir()
	e, _ := engine.Open(dir, dim, distance.MetricL2)
	defer e.Close()

	rng := testutil.NewRNG(1)
	data := make([][]float32, numVecs)
	for i := 0; i < numVecs; i++ {
		vec := make([]float32, dim)
		rng.FillUniform(vec)
		data[i] = vec
		e.Insert(model.PrimaryKey(i), vec, nil, nil)
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
			truth := exactTopK_L2(data, q, 10)
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
			truth := exactTopK_L2(data, q, 10)
			sum += recallAtK(res[i], truth)
		}
		b.ReportMetric(sum/float64(len(queries)), "recall@10")
	})
}
