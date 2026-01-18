package benchmark_test

import (
	"context"
	"fmt"
	"sort"
	"testing"
	"time"

	"github.com/hupe1980/vecgo"
	"github.com/hupe1980/vecgo/model"
	"github.com/hupe1980/vecgo/testutil"
)

// ============================================================================
// ALGORITHM TUNING BENCHMARKS — Measure Algorithm Efficiency
// ============================================================================
//
// These benchmarks help tune HNSW/DiskANN parameters by measuring:
// - Search quality vs speed at different ef_search values
// - Build quality vs speed at different ef_construction values
// - Memory vs recall at different M (neighbor count) values
//
// Run: go test -bench=BenchmarkTuning -run=^$ -timeout=10m ./benchmark_test/...

// BenchmarkEfSearchTuning measures how ef_search affects recall and latency.
// ef_search controls the search beam width - higher = better recall, more work.
func BenchmarkEfSearchTuning(b *testing.B) {
	ctx := context.Background()

	const dim = 128
	const n = 50_000
	const k = 10

	if testing.Short() {
		b.Skip("skipping in short mode")
	}

	// Build index with fixed parameters
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

	// Generate and insert data
	rng := testutil.NewRNG(benchSeed)
	data := make([][]float32, n)
	for i := range data {
		vec := make([]float32, dim)
		rng.FillUniform(vec)
		data[i] = vec
	}

	const batchSize = 1000
	pks := make([]uint64, n)

	for start := 0; start < n; start += batchSize {
		end := min(start+batchSize, n)
		ids, err := db.BatchInsert(ctx, data[start:end], nil, nil)
		if err != nil {
			b.Fatalf("insert: %v", err)
		}
		for i, id := range ids {
			pks[start+i] = uint64(id)
		}
	}

	// Generate queries and compute exact ground truth
	queryRng := testutil.NewRNG(benchSeed + 1000)
	numQueries := 100
	queries := make([][]float32, numQueries)
	for i := range queries {
		q := make([]float32, dim)
		queryRng.FillUniform(q)
		queries[i] = q
	}

	// Note: ef_search is controlled via vecgo options - using proxy k values
	// Higher k approximates higher ef_search behavior
	efSearchValues := []int{16, 32, 64, 128, 256}

	for _, ef := range efSearchValues {
		b.Run(fmt.Sprintf("ef=%d", ef), func(b *testing.B) {
			const numSamples = 1000
			latencies := make([]time.Duration, numSamples)

			b.ResetTimer()

			for i := 0; i < numSamples; i++ {
				q := queries[i%len(queries)]
				start := time.Now()
				// Use larger k to simulate ef_search effect
				searchK := max(k, ef/4)
				_, err := db.Search(ctx, q, searchK, vecgo.WithoutData())
				latencies[i] = time.Since(start)
				if err != nil {
					b.Fatal(err)
				}
			}

			b.StopTimer()

			sort.Slice(latencies, func(i, j int) bool {
				return latencies[i] < latencies[j]
			})

			// Calculate recall
			var totalRecall float64
			numRecallSamples := min(50, numQueries)
			for qi := 0; qi < numRecallSamples; qi++ {
				searchK := max(k, ef/4)
				results, _ := db.Search(ctx, queries[qi], searchK, vecgo.WithoutData())
				// Compare to brute force
				exactIDs := exactTopK_L2_WithPKs(data, pks, queries[qi], k)
				totalRecall += recallWithPKs(results[:min(k, len(results))], exactIDs)
			}

			b.ReportMetric(float64(latencies[numSamples*50/100].Microseconds()), "P50_μs")
			b.ReportMetric(float64(latencies[numSamples*99/100].Microseconds()), "P99_μs")
			b.ReportMetric(totalRecall/float64(numRecallSamples), "recall@10")
		})
	}
}

// BenchmarkBuildQuality measures index build time vs search quality trade-off.
func BenchmarkBuildQuality(b *testing.B) {
	ctx := context.Background()

	const dim = 128
	const k = 10

	sizes := []int{10_000, 25_000, 50_000}
	if testing.Short() {
		sizes = []int{10_000}
	}

	for _, n := range sizes {
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			// Pre-generate data
			rng := testutil.NewRNG(benchSeed)
			data := make([][]float32, n)
			for i := range data {
				vec := make([]float32, dim)
				rng.FillUniform(vec)
				data[i] = vec
			}

			// Measure build time
			dir := b.TempDir()
			buildStart := time.Now()

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

			const batchSize = 1000
			pks := make([]uint64, n)
			for start := 0; start < n; start += batchSize {
				end := min(start+batchSize, n)
				ids, err := db.BatchInsert(ctx, data[start:end], nil, nil)
				if err != nil {
					b.Fatalf("insert: %v", err)
				}
				for i, id := range ids {
					pks[start+i] = uint64(id)
				}
			}

			buildTime := time.Since(buildStart)

			// Generate queries
			queryRng := testutil.NewRNG(benchSeed + 1000)
			numQueries := 100
			queries := make([][]float32, numQueries)
			for i := range queries {
				q := make([]float32, dim)
				queryRng.FillUniform(q)
				queries[i] = q
			}

			// Measure search quality
			const numSearchSamples = 1000
			searchLatencies := make([]time.Duration, numSearchSamples)

			for i := 0; i < numSearchSamples; i++ {
				q := queries[i%len(queries)]
				start := time.Now()
				_, err := db.Search(ctx, q, k, vecgo.WithoutData())
				searchLatencies[i] = time.Since(start)
				if err != nil {
					b.Fatal(err)
				}
			}

			sort.Slice(searchLatencies, func(i, j int) bool {
				return searchLatencies[i] < searchLatencies[j]
			})

			// Calculate recall
			var totalRecall float64
			numRecallSamples := min(50, numQueries)
			for qi := 0; qi < numRecallSamples; qi++ {
				results, _ := db.Search(ctx, queries[qi], k, vecgo.WithoutData())
				exactIDs := exactTopK_L2_WithPKs(data, pks, queries[qi], k)
				totalRecall += recallWithPKs(results, exactIDs)
			}

			db.Close()

			b.ReportMetric(float64(buildTime.Milliseconds()), "build_ms")
			b.ReportMetric(float64(n)/buildTime.Seconds(), "build_vecs/sec")
			b.ReportMetric(float64(searchLatencies[numSearchSamples*50/100].Microseconds()), "search_P50_μs")
			b.ReportMetric(totalRecall/float64(numRecallSamples), "recall@10")
		})
	}
}

// BenchmarkDimensionScaling measures how dimension affects performance.
func BenchmarkDimensionScaling(b *testing.B) {
	ctx := context.Background()

	const n = 10_000
	const k = 10

	dims := []int{64, 128, 256, 512, 768, 1024, 1536}
	if testing.Short() {
		dims = []int{128, 768}
	}

	for _, dim := range dims {
		b.Run(fmt.Sprintf("dim=%d", dim), func(b *testing.B) {
			// Pre-generate data
			rng := testutil.NewRNG(benchSeed)
			data := make([][]float32, n)
			for i := range data {
				vec := make([]float32, dim)
				rng.FillUniform(vec)
				data[i] = vec
			}

			// Build index
			dir := b.TempDir()
			buildStart := time.Now()

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

			const batchSize = 1000
			for start := 0; start < n; start += batchSize {
				end := min(start+batchSize, n)
				_, err := db.BatchInsert(ctx, data[start:end], nil, nil)
				if err != nil {
					b.Fatalf("insert: %v", err)
				}
			}

			buildTime := time.Since(buildStart)

			// Generate queries
			queryRng := testutil.NewRNG(benchSeed + 1000)
			numQueries := 100
			queries := make([][]float32, numQueries)
			for i := range queries {
				q := make([]float32, dim)
				queryRng.FillUniform(q)
				queries[i] = q
			}

			// Measure search
			const numSamples = 1000
			latencies := make([]time.Duration, numSamples)

			for i := 0; i < numSamples; i++ {
				q := queries[i%len(queries)]
				start := time.Now()
				_, err := db.Search(ctx, q, k, vecgo.WithoutData())
				latencies[i] = time.Since(start)
				if err != nil {
					b.Fatal(err)
				}
			}

			db.Close()

			sort.Slice(latencies, func(i, j int) bool {
				return latencies[i] < latencies[j]
			})

			b.ReportMetric(float64(buildTime.Milliseconds()), "build_ms")
			b.ReportMetric(float64(latencies[numSamples*50/100].Microseconds()), "P50_μs")
			b.ReportMetric(float64(latencies[numSamples*99/100].Microseconds()), "P99_μs")
			b.ReportMetric(float64(dim*4), "bytes/vec_raw")
		})
	}
}

// BenchmarkBatchSizeOptimal finds optimal batch size for insert throughput.
func BenchmarkBatchSizeOptimal(b *testing.B) {
	ctx := context.Background()

	const dim = 768
	const totalVecs = 50_000

	batchSizes := []int{1, 10, 50, 100, 250, 500, 1000, 2500, 5000}
	if testing.Short() {
		batchSizes = []int{100, 500, 1000}
	}

	for _, bs := range batchSizes {
		b.Run(fmt.Sprintf("batch=%d", bs), func(b *testing.B) {
			// Pre-generate data
			rng := testutil.NewRNG(benchSeed)
			data := make([][]float32, totalVecs)
			for i := range data {
				vec := make([]float32, dim)
				rng.FillUniform(vec)
				data[i] = vec
			}

			dir := b.TempDir()
			db, err := vecgo.Open(ctx, vecgo.Local(dir),
				vecgo.Create(dim, vecgo.MetricL2),
				vecgo.WithCompactionThreshold(1<<40),
				vecgo.WithFlushConfig(vecgo.FlushConfig{MaxMemTableSize: 64 << 20}),
				vecgo.WithDiskANNThreshold(totalVecs+1),
				vecgo.WithMemoryLimit(0),
			)
			if err != nil {
				b.Fatalf("open: %v", err)
			}

			b.ResetTimer()

			start := time.Now()
			for i := 0; i < totalVecs; i += bs {
				end := min(i+bs, totalVecs)
				_, err := db.BatchInsert(ctx, data[i:end], nil, nil)
				if err != nil {
					b.Fatalf("insert: %v", err)
				}
			}
			elapsed := time.Since(start)

			db.Close()

			b.StopTimer()
			b.ReportMetric(float64(totalVecs)/elapsed.Seconds(), "vecs/sec")
			b.ReportMetric(float64(elapsed.Nanoseconds())/float64(totalVecs), "ns/vec")
		})
	}
}

// ============================================================================
// Helper Functions
// ============================================================================

func exactTopK_L2_WithPKs(data [][]float32, pks []uint64, q []float32, k int) []uint64 {
	type distPK struct {
		dist float32
		pk   uint64
	}

	dists := make([]distPK, len(data))
	for i, v := range data {
		dists[i] = distPK{dist: l2DistanceSquared(q, v), pk: pks[i]}
	}

	sort.Slice(dists, func(i, j int) bool {
		return dists[i].dist < dists[j].dist
	})

	result := make([]uint64, min(k, len(dists)))
	for i := range result {
		result[i] = dists[i].pk
	}
	return result
}

func l2DistanceSquared(a, b []float32) float32 {
	var sum float32
	for i := range a {
		d := a[i] - b[i]
		sum += d * d
	}
	return sum
}

func recallWithPKs(results []model.Candidate, truth []uint64) float64 {
	if len(truth) == 0 {
		return 0
	}

	set := make(map[uint64]struct{}, len(truth))
	for _, pk := range truth {
		set[pk] = struct{}{}
	}

	var hit int
	for _, c := range results {
		if _, ok := set[uint64(c.ID)]; ok {
			hit++
		}
	}
	return float64(hit) / float64(min(len(results), len(truth)))
}
