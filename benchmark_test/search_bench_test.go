package benchmark_test

import (
	"context"
	"sync/atomic"
	"testing"

	"github.com/hupe1980/vecgo"
	"github.com/hupe1980/vecgo/distance"
	"github.com/hupe1980/vecgo/engine"
	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/model"
	"github.com/hupe1980/vecgo/testutil"
)

func BenchmarkSearch_Mixed(b *testing.B) {
	b.ReportAllocs()

	dir := b.TempDir()
	// Use small flush threshold to force multiple segments
	eng, err := vecgo.Open(dir, 128, vecgo.MetricL2,
		engine.WithFlushConfig(engine.FlushConfig{MaxMemTableSize: 1024 * 1024}), // 1MB
		engine.WithCompactionThreshold(4),
		engine.WithWALOptions(engine.WALOptions{Durability: engine.DurabilityAsync}),
	)
	if err != nil {
		b.Fatal(err)
	}
	defer eng.Close()

	const (
		dim      = 128
		n        = 1500
		k        = 10
		nQueries = 32
	)

	// Deterministic dataset and queries.
	rng := testutil.NewRNG(1)
	data := make([][]float32, n)
	for i := 0; i < n; i++ {
		v := make([]float32, dim)
		rng.FillUniform(v)
		data[i] = v
		if err := eng.Insert(model.PKUint64(uint64(i)), v, nil, nil); err != nil {
			b.Fatal(err)
		}

		// Force a stable "mixed" state: some data in immutable segments,
		// then keep the tail in L0.
		if i == 499 || i == 999 {
			if err := eng.Flush(); err != nil {
				b.Fatal(err)
			}
		}
	}

	queries := make([][]float32, nQueries)
	for i := range queries {
		q := make([]float32, dim)
		rng.FillUniform(q)
		queries[i] = q
	}

	// Compute exact ground-truth once (outside timed section).
	truth := make([][]model.PrimaryKey, len(queries))
	for i, q := range queries {
		truth[i] = exactTopK_L2(data, q, k)
	}

	var qIdx atomic.Uint64
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			i := int(qIdx.Add(1) % uint64(len(queries)))
			if _, err := eng.Search(context.Background(), queries[i], k); err != nil {
				b.Fatal(err)
			}
		}
	})

	// Report recall@k as a benchmark metric, computed outside the timed region.
	b.StopTimer()
	var sum float64
	for i := range queries {
		res, err := eng.Search(context.Background(), queries[i], k)
		if err != nil {
			b.Fatal(err)
		}
		sum += recallAtK(res, truth[i])
	}
	avg := sum / float64(len(queries))
	b.ReportMetric(avg, "recall@10")
}

func BenchmarkSearchFiltered(b *testing.B) {
	dir := b.TempDir()
	eng, err := engine.Open(dir, 128, distance.MetricL2, engine.WithWALOptions(engine.WALOptions{Durability: engine.DurabilityAsync}))
	if err != nil {
		b.Fatal(err)
	}
	defer eng.Close()

	n := 10000
	dim := 128
	rng := testutil.NewRNG(1)

	// Insert with metadata
	// 50% category "A", 50% "B"
	dataA := make([][]float32, 0, n/2)
	pksA := make([]model.PrimaryKey, 0, n/2)

	for i := 0; i < n; i++ {
		vec := make([]float32, dim)
		rng.FillUniform(vec)

		cat := "A"
		if i%2 == 1 {
			cat = "B"
		} else {
			dataA = append(dataA, vec)
			pksA = append(pksA, model.PKUint64(uint64(i)))
		}

		md := map[string]interface{}{
			"category": cat,
		}

		if err := eng.Insert(model.PKUint64(uint64(i)), vec, md, nil); err != nil {
			b.Fatal(err)
		}
	}

	q := make([]float32, dim)
	rng.FillUniform(q)

	// Compute ground truth for category "A"
	// We need a version of exactTopK_L2 that takes explicit PKs because dataA indices don't match PKs
	truth := exactTopK_L2_WithPKs(dataA, pksA, q, 10)

	filterA := metadata.NewFilterSet(metadata.Filter{
		Key:      "category",
		Operator: metadata.OpEqual,
		Value:    metadata.String("A"),
	})

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := eng.Search(context.Background(), q, 10, engine.WithFilter(filterA)); err != nil {
			b.Fatal(err)
		}
	}

	b.StopTimer()
	res, err := eng.Search(context.Background(), q, 10, engine.WithFilter(filterA))
	if err != nil {
		b.Fatal(err)
	}
	b.ReportMetric(recallAtK(res, truth), "recall@10")
}
