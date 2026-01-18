package benchmark_test

import (
	"context"
	"fmt"
	"runtime"
	"sort"
	"testing"
	"time"

	"github.com/hupe1980/vecgo"
	"github.com/hupe1980/vecgo/model"
	"github.com/hupe1980/vecgo/testutil"
)

// ============================================================================
// PRODUCTION METRICS BENCHMARKS — Measure What Matters for Tuning
// ============================================================================
//
// Standard Go benchmarks report mean latency. But production SLAs care about
// tail latency (P99) and throughput under load. These benchmarks measure:
//
// 1. Latency percentiles (P50, P95, P99, P99.9)
// 2. Throughput vs latency trade-off
// 3. Memory footprint per vector
// 4. Index build time
// 5. Cold start latency
// 6. Recall vs QPS Pareto frontier
//
// Run: go test -bench=BenchmarkMetrics -run=^$ -timeout=10m ./benchmark_test/...

// BenchmarkLatencyPercentiles measures P50/P95/P99/P99.9 search latency.
func BenchmarkLatencyPercentiles(b *testing.B) {
	ctx := context.Background()

	fixtureName := "uniform_128d_50k"
	if testing.Short() {
		fixtureName = "uniform_128d_10k"
	}

	if !FixtureExists(fixtureName) {
		b.Skipf("fixture %q not found", fixtureName)
	}

	db, err := OpenFixture(ctx, fixtureName)
	if err != nil {
		b.Fatalf("open fixture: %v", err)
	}
	defer db.Close()

	data, err := LoadFixtureData(fixtureName)
	if err != nil {
		b.Fatalf("load fixture data: %v", err)
	}

	queries := data.Queries
	const k = 10
	const numSamples = 1000 // Reduced from 10k for faster benchmarks

	latencies := make([]time.Duration, numSamples)

	// Warmup
	for i := 0; i < 100; i++ {
		_, _ = db.Search(ctx, queries[i%len(queries)], k, vecgo.WithoutData())
	}

	// Measure
	for i := 0; i < numSamples; i++ {
		q := queries[i%len(queries)]
		start := time.Now()
		_, err := db.Search(ctx, q, k, vecgo.WithoutData())
		latencies[i] = time.Since(start)
		if err != nil {
			b.Fatal(err)
		}
	}

	sort.Slice(latencies, func(i, j int) bool {
		return latencies[i] < latencies[j]
	})

	p50 := numSamples * 50 / 100
	p95 := numSamples * 95 / 100
	p99 := numSamples * 99 / 100
	p999 := min(numSamples-1, numSamples*999/1000)

	b.ReportMetric(float64(latencies[p50].Microseconds()), "P50_μs")
	b.ReportMetric(float64(latencies[p95].Microseconds()), "P95_μs")
	b.ReportMetric(float64(latencies[p99].Microseconds()), "P99_μs")
	b.ReportMetric(float64(latencies[p999].Microseconds()), "P99.9_μs")
}

// BenchmarkLatencyUnderLoad measures latency with concurrent load.
func BenchmarkLatencyUnderLoad(b *testing.B) {
	ctx := context.Background()

	fixtureName := "uniform_128d_50k"
	if testing.Short() {
		fixtureName = "uniform_128d_10k"
	}

	if !FixtureExists(fixtureName) {
		b.Skipf("fixture %q not found", fixtureName)
	}

	db, err := OpenFixture(ctx, fixtureName)
	if err != nil {
		b.Fatalf("open fixture: %v", err)
	}
	defer db.Close()

	data, err := LoadFixtureData(fixtureName)
	if err != nil {
		b.Fatalf("load fixture data: %v", err)
	}

	queries := data.Queries
	const k = 10
	concurrencies := []int{1, 2, 4, 8}

	for _, c := range concurrencies {
		b.Run(fmt.Sprintf("conc=%d", c), func(b *testing.B) {
			const samplesPerWorker = 500 // Reduced for faster benchmarks
			totalSamples := samplesPerWorker * c
			latencies := make([]time.Duration, totalSamples)

			b.SetParallelism(c)
			b.ResetTimer()

			done := make(chan struct{})
			for w := 0; w < c; w++ {
				go func(workerID int) {
					for i := 0; i < samplesPerWorker; i++ {
						q := queries[(workerID*samplesPerWorker+i)%len(queries)]
						start := time.Now()
						_, _ = db.Search(ctx, q, k, vecgo.WithoutData())
						latencies[workerID*samplesPerWorker+i] = time.Since(start)
					}
					done <- struct{}{}
				}(w)
			}

			for w := 0; w < c; w++ {
				<-done
			}

			b.StopTimer()

			sort.Slice(latencies, func(i, j int) bool {
				return latencies[i] < latencies[j]
			})

			b.ReportMetric(float64(latencies[totalSamples*50/100].Microseconds()), "P50_μs")
			b.ReportMetric(float64(latencies[totalSamples*95/100].Microseconds()), "P95_μs")
			b.ReportMetric(float64(latencies[totalSamples*99/100].Microseconds()), "P99_μs")
		})
	}
}

// BenchmarkMemoryFootprint measures memory usage per vector.
func BenchmarkMemoryFootprint(b *testing.B) {
	ctx := context.Background()

	configs := []struct {
		dim  int
		size int
	}{
		{128, 10_000},
		{768, 10_000},
	}

	if testing.Short() {
		configs = configs[:1]
	}

	for _, cfg := range configs {
		b.Run(fmt.Sprintf("dim=%d/n=%d", cfg.dim, cfg.size), func(b *testing.B) {
			runtime.GC()
			var m1 runtime.MemStats
			runtime.ReadMemStats(&m1)

			dir := b.TempDir()
			db, err := vecgo.Open(ctx, vecgo.Local(dir),
				vecgo.Create(cfg.dim, vecgo.MetricL2),
				vecgo.WithCompactionThreshold(1<<40),
				vecgo.WithFlushConfig(vecgo.FlushConfig{MaxMemTableSize: 64 << 20}),
				vecgo.WithDiskANNThreshold(cfg.size+1),
				vecgo.WithMemoryLimit(0),
			)
			if err != nil {
				b.Fatalf("open: %v", err)
			}

			rng := testutil.NewRNG(benchSeed)
			const batchSize = 1000
			for start := 0; start < cfg.size; start += batchSize {
				end := min(start+batchSize, cfg.size)
				batch := make([][]float32, end-start)
				for i := range batch {
					vec := make([]float32, cfg.dim)
					rng.FillUniform(vec)
					batch[i] = vec
				}
				_, err := db.BatchInsert(ctx, batch, nil, nil)
				if err != nil {
					b.Fatalf("insert: %v", err)
				}
			}

			runtime.GC()
			var m2 runtime.MemStats
			runtime.ReadMemStats(&m2)

			heapUsed := m2.HeapAlloc - m1.HeapAlloc
			bytesPerVec := float64(heapUsed) / float64(cfg.size)
			rawBytesPerVec := float64(cfg.dim * 4)

			b.ReportMetric(bytesPerVec, "bytes/vec")
			b.ReportMetric(rawBytesPerVec, "raw_bytes/vec")
			b.ReportMetric(bytesPerVec/rawBytesPerVec, "overhead_ratio")

			db.Close()
		})
	}
}

// BenchmarkIndexBuildTime measures time to build index from scratch.
func BenchmarkIndexBuildTime(b *testing.B) {
	ctx := context.Background()

	configs := []struct {
		dim  int
		size int
	}{
		{128, 10_000},
		{768, 10_000},
	}

	if testing.Short() {
		configs = configs[:1]
	}

	for _, cfg := range configs {
		b.Run(fmt.Sprintf("dim=%d/n=%d", cfg.dim, cfg.size), func(b *testing.B) {
			rng := testutil.NewRNG(benchSeed)
			data := make([][]float32, cfg.size)
			for i := range data {
				vec := make([]float32, cfg.dim)
				rng.FillUniform(vec)
				data[i] = vec
			}

			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				b.StopTimer()
				dir := b.TempDir()
				b.StartTimer()

				db, err := vecgo.Open(ctx, vecgo.Local(dir),
					vecgo.Create(cfg.dim, vecgo.MetricL2),
					vecgo.WithCompactionThreshold(1<<40),
					vecgo.WithFlushConfig(vecgo.FlushConfig{MaxMemTableSize: 64 << 20}),
					vecgo.WithDiskANNThreshold(cfg.size+1),
					vecgo.WithMemoryLimit(0),
				)
				if err != nil {
					b.Fatalf("open: %v", err)
				}

				const batchSize = 1000
				for start := 0; start < cfg.size; start += batchSize {
					end := min(start+batchSize, cfg.size)
					_, err := db.BatchInsert(ctx, data[start:end], nil, nil)
					if err != nil {
						b.Fatalf("insert: %v", err)
					}
				}

				db.Close()
			}

			b.StopTimer()
			b.ReportMetric(float64(cfg.size)/b.Elapsed().Seconds()*float64(b.N), "vecs/sec")
		})
	}
}

// BenchmarkColdStart measures first-query latency after Open().
func BenchmarkColdStart(b *testing.B) {
	ctx := context.Background()

	fixtureName := "uniform_128d_50k"
	if testing.Short() {
		fixtureName = "uniform_128d_10k"
	}

	if !FixtureExists(fixtureName) {
		b.Skipf("fixture %q not found", fixtureName)
	}

	data, err := LoadFixtureData(fixtureName)
	if err != nil {
		b.Fatalf("load fixture data: %v", err)
	}

	query := data.Queries[0]
	const k = 10
	const numTrials = 10

	openLatencies := make([]time.Duration, numTrials)
	firstQueryLatencies := make([]time.Duration, numTrials)

	for trial := 0; trial < numTrials; trial++ {
		runtime.GC()

		startOpen := time.Now()
		db, err := OpenFixture(ctx, fixtureName)
		openLatencies[trial] = time.Since(startOpen)
		if err != nil {
			b.Fatalf("open fixture: %v", err)
		}

		startQuery := time.Now()
		_, err = db.Search(ctx, query, k, vecgo.WithoutData())
		firstQueryLatencies[trial] = time.Since(startQuery)
		if err != nil {
			b.Fatal(err)
		}

		db.Close()
	}

	sort.Slice(openLatencies, func(i, j int) bool {
		return openLatencies[i] < openLatencies[j]
	})
	sort.Slice(firstQueryLatencies, func(i, j int) bool {
		return firstQueryLatencies[i] < firstQueryLatencies[j]
	})

	b.ReportMetric(float64(openLatencies[numTrials/2].Milliseconds()), "open_ms")
	b.ReportMetric(float64(firstQueryLatencies[numTrials/2].Microseconds()), "first_query_μs")
	b.ReportMetric(float64((openLatencies[numTrials/2] + firstQueryLatencies[numTrials/2]).Milliseconds()), "cold_start_ms")
}

// BenchmarkRecallQPSTradeoff measures recall/throughput trade-off.
func BenchmarkRecallQPSTradeoff(b *testing.B) {
	ctx := context.Background()

	fixtureName := "uniform_128d_50k"
	if testing.Short() {
		fixtureName = "uniform_128d_10k"
	}

	if !FixtureExists(fixtureName) {
		b.Skipf("fixture %q not found", fixtureName)
	}

	db, err := OpenFixture(ctx, fixtureName)
	if err != nil {
		b.Fatalf("open fixture: %v", err)
	}
	defer db.Close()

	data, err := LoadFixtureData(fixtureName)
	if err != nil {
		b.Fatalf("load fixture data: %v", err)
	}

	queries := data.Queries
	truth := data.GroundTruth["0.01"]
	if truth == nil {
		truth = data.GroundTruth["1.00"]
	}

	kValues := []int{1, 5, 10, 20, 50, 100}

	for _, k := range kValues {
		b.Run(fmt.Sprintf("k=%d", k), func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				q := queries[i%len(queries)]
				_, err := db.Search(ctx, q, k, vecgo.WithoutData())
				if err != nil {
					b.Fatal(err)
				}
			}

			b.StopTimer()

			if truth != nil && len(truth) > 0 {
				var totalRecall float64
				numSamples := min(50, len(queries))
				for qi := 0; qi < numSamples; qi++ {
					results, _ := db.Search(ctx, queries[qi], k, vecgo.WithoutData())
					gtK := min(k, len(truth[qi]))
					totalRecall += recallAtKWithIDs(results[:min(k, len(results))], truth[qi][:gtK])
				}
				b.ReportMetric(totalRecall/float64(numSamples), "recall")
			}
			b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "qps")
		})
	}
}

// BenchmarkDeletePerformance measures delete throughput.
func BenchmarkDeletePerformance(b *testing.B) {
	ctx := context.Background()

	const dim = 128
	const n = 10_000
	const deleteRatio = 0.10
	const k = 10

	dir := b.TempDir()
	db, err := vecgo.Open(ctx, vecgo.Local(dir),
		vecgo.Create(dim, vecgo.MetricL2),
		vecgo.WithCompactionThreshold(1<<40),
		vecgo.WithFlushConfig(vecgo.FlushConfig{MaxMemTableSize: 64 << 20}),
		vecgo.WithDiskANNThreshold(n+1),
		vecgo.WithMemoryLimit(0),
	)
	if err != nil {
		b.Fatalf("open: %v", err)
	}
	defer db.Close()

	rng := testutil.NewRNG(benchSeed)
	pks := make([]model.ID, n)
	const batchSize = 1000

	for start := 0; start < n; start += batchSize {
		end := min(start+batchSize, n)
		batch := make([][]float32, end-start)
		for i := range batch {
			vec := make([]float32, dim)
			rng.FillUniform(vec)
			batch[i] = vec
		}
		ids, err := db.BatchInsert(ctx, batch, nil, nil)
		if err != nil {
			b.Fatalf("insert: %v", err)
		}
		copy(pks[start:end], ids)
	}

	queries := MakeQueries(100, dim)

	for _, q := range queries[:10] {
		_, _ = db.Search(ctx, q, k, vecgo.WithoutData())
	}

	numDeletes := int(float64(n) * deleteRatio)
	deleteIDs := pks[:numDeletes]

	b.Run("delete", func(b *testing.B) {
		b.ResetTimer()
		start := time.Now()

		for _, id := range deleteIDs {
			if err := db.Delete(ctx, id); err != nil {
				b.Fatal(err)
			}
		}

		elapsed := time.Since(start)
		b.StopTimer()

		b.ReportMetric(float64(numDeletes)/elapsed.Seconds(), "deletes/sec")
		b.ReportMetric(float64(elapsed.Nanoseconds())/float64(numDeletes), "ns/delete")
	})

	b.Run("search_after_delete", func(b *testing.B) {
		b.ResetTimer()

		for i := 0; i < b.N; i++ {
			q := queries[i%len(queries)]
			_, err := db.Search(ctx, q, k, vecgo.WithoutData())
			if err != nil {
				b.Fatal(err)
			}
		}

		b.StopTimer()
		b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "qps")
	})
}

// BenchmarkGCPressure measures allocation patterns under sustained load.
func BenchmarkGCPressure(b *testing.B) {
	ctx := context.Background()

	fixtureName := "uniform_128d_50k"
	if testing.Short() {
		fixtureName = "uniform_128d_10k"
	}

	if !FixtureExists(fixtureName) {
		b.Skipf("fixture %q not found", fixtureName)
	}

	db, err := OpenFixture(ctx, fixtureName)
	if err != nil {
		b.Fatalf("open fixture: %v", err)
	}
	defer db.Close()

	data, err := LoadFixtureData(fixtureName)
	if err != nil {
		b.Fatalf("load fixture data: %v", err)
	}

	queries := data.Queries
	const k = 10
	const numOps = 1000 // Reduced for faster benchmarks

	runtime.GC()
	var m1 runtime.MemStats
	runtime.ReadMemStats(&m1)

	start := time.Now()
	for i := 0; i < numOps; i++ {
		q := queries[i%len(queries)]
		_, _ = db.Search(ctx, q, k, vecgo.WithoutData())
	}
	elapsed := time.Since(start)

	var m2 runtime.MemStats
	runtime.ReadMemStats(&m2)

	allocsPerOp := float64(m2.Mallocs-m1.Mallocs) / float64(numOps)
	bytesPerOp := float64(m2.TotalAlloc-m1.TotalAlloc) / float64(numOps)
	gcPauses := m2.NumGC - m1.NumGC

	b.ReportMetric(allocsPerOp, "allocs/op")
	b.ReportMetric(bytesPerOp, "bytes_alloc/op")
	b.ReportMetric(float64(gcPauses), "gc_cycles")
	b.ReportMetric(float64(numOps)/elapsed.Seconds(), "qps")
}
