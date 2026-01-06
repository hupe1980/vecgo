package benchmark_test

import (
	"context"
	"math/rand"
	"sync/atomic"
	"testing"

	"github.com/hupe1980/vecgo"
	"github.com/hupe1980/vecgo/engine"
	"github.com/hupe1980/vecgo/model"
	"github.com/hupe1980/vecgo/testutil"
)

func BenchmarkIngest_Async(b *testing.B) {
	benchmarkIngest(b, engine.DurabilityAsync)
}

func BenchmarkIngest_Sync(b *testing.B) {
	benchmarkIngest(b, engine.DurabilitySync)
}

func BenchmarkIngest_Async_Parallel(b *testing.B) {
	benchmarkIngestParallel(b, engine.DurabilityAsync)
}

func BenchmarkIngest_Sync_Parallel(b *testing.B) {
	benchmarkIngestParallel(b, engine.DurabilitySync)
}

func benchmarkIngest(b *testing.B, durability engine.Durability) {
	b.ReportAllocs()

	dir := b.TempDir()

	eng, err := vecgo.Open(dir, 128, vecgo.MetricL2, engine.WithWALOptions(engine.WALOptions{Durability: durability}))
	if err != nil {
		b.Fatal(err)
	}
	defer eng.Close()

	rng := testutil.NewRNG(1)
	vec := make([]float32, 128)
	rng.FillUniform(vec)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		pk := model.PKUint64(uint64(i))
		if err := eng.Insert(pk, vec, nil, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func benchmarkIngestParallel(b *testing.B, durability engine.Durability) {
	b.ReportAllocs()

	dir := b.TempDir()

	eng, err := vecgo.Open(dir, 128, vecgo.MetricL2, engine.WithWALOptions(engine.WALOptions{Durability: durability}))
	if err != nil {
		b.Fatal(err)
	}
	defer eng.Close()

	rng := testutil.NewRNG(1)
	vec := make([]float32, 128)
	rng.FillUniform(vec)

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			pk := model.PKUint64(uint64(rand.Int63()))
			if err := eng.Insert(pk, vec, nil, nil); err != nil {
				b.Fatal(err)
			}
		}
	})
}

func BenchmarkSearch_L0(b *testing.B) {
	b.ReportAllocs()

	dir := b.TempDir()
	eng, err := vecgo.Open(dir, 128, vecgo.MetricL2)
	if err != nil {
		b.Fatal(err)
	}
	defer eng.Close()

	// Insert 10k vectors
	rng := testutil.NewRNG(1)
	data := make([][]float32, 10000)
	for i := 0; i < 10000; i++ {
		vec := make([]float32, 128)
		rng.FillUniform(vec)
		data[i] = vec
		if err := eng.Insert(model.PKUint64(uint64(i)), vec, nil, nil); err != nil {
			b.Fatal(err)
		}
	}

	// Pre-generate queries outside the timed region.
	queries := make([][]float32, 256)
	for i := range queries {
		q := make([]float32, 128)
		rng.FillUniform(q)
		queries[i] = q
	}

	var qIdx atomic.Uint64
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			q := queries[qIdx.Add(1)%uint64(len(queries))]
			if _, err := eng.Search(context.Background(), q, 10); err != nil {
				b.Fatal(err)
			}
		}
	})

	b.StopTimer()
	var sum float64
	for _, q := range queries {
		truth := exactTopK_L2(data, q, 10)
		res, err := eng.Search(context.Background(), q, 10)
		if err != nil {
			b.Fatal(err)
		}
		sum += recallAtK(res, truth)
	}
	b.ReportMetric(sum/float64(len(queries)), "recall@10")
}
