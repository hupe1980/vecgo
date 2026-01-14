package benchmark_test

import (
	"context"
	"fmt"
	"testing"

	"github.com/hupe1980/vecgo"
	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/model"
	"github.com/hupe1980/vecgo/testutil"
)

// BenchmarkFilteredSearchSelectivity tests filtered search at various selectivity levels.
// Default: dim=128, numVecs=50K (fast CI benchmark)
func BenchmarkFilteredSearchSelectivity(b *testing.B) {
	runFilteredBenchmark(b, 128, 50_000)
}

// BenchmarkFilteredSearch_768dim_100K tests with OpenAI-small/Cohere dimensions at 100K scale.
// Skipped by default - run with: go test -bench=FilteredSearch_768dim -benchtime=1x
func BenchmarkFilteredSearch_768dim_100K(b *testing.B) {
	if testing.Short() {
		b.Skip("skipping large benchmark in short mode")
	}
	runFilteredBenchmark(b, 768, 100_000)
}

// BenchmarkFilteredSearch_1536dim_100K tests with OpenAI-large dimensions at 100K scale.
// Skipped by default - run with: go test -bench=FilteredSearch_1536dim -benchtime=1x
func BenchmarkFilteredSearch_1536dim_100K(b *testing.B) {
	if testing.Short() {
		b.Skip("skipping large benchmark in short mode")
	}
	runFilteredBenchmark(b, 1536, 100_000)
}

// BenchmarkFilteredSearch_768dim_500K tests production-scale workload.
// WARNING: Requires ~3GB RAM and takes several minutes.
// Run with: go test -bench=FilteredSearch_768dim_500K -benchtime=1x -timeout=30m
func BenchmarkFilteredSearch_768dim_500K(b *testing.B) {
	if testing.Short() {
		b.Skip("skipping large benchmark in short mode")
	}
	runFilteredBenchmark(b, 768, 500_000)
}

func runFilteredBenchmark(b *testing.B, dim, numVecs int) {
	bucketCount := int64(100) // enables 1% steps via "bucket < threshold"
	k := 10
	numQueries := 10
	batchSize := 1000 // batch insert for performance

	dir := b.TempDir()
	// High thresholds prevent background compaction during load
	e, err := vecgo.Open(context.Background(), vecgo.Local(dir), vecgo.Create(dim, vecgo.MetricL2),
		vecgo.WithCompactionThreshold(1<<40),                               // ~1TB, effectively disables auto-compaction
		vecgo.WithFlushConfig(vecgo.FlushConfig{MaxMemTableSize: 1 << 30}), // 1GB memtable
		vecgo.WithDiskANNThreshold(numVecs+1),                              // disable DiskANN building
		vecgo.WithMemoryLimit(0),                                           // disable memory semaphore timeout
	)
	if err != nil {
		b.Fatalf("failed to open: %v", err)
	}
	defer e.Close()

	rng := testutil.NewRNG(1)
	data := make([][]float32, numVecs)
	pks := make([]model.ID, numVecs)
	buckets := make([]int64, numVecs)

	b.Logf("Loading %d vectors of dim=%d (batch size=%d)...", numVecs, dim, batchSize)

	// Generate all data first
	for i := 0; i < numVecs; i++ {
		vec := make([]float32, dim)
		rng.FillUniform(vec)
		data[i] = vec
		buckets[i] = int64(i) % bucketCount
	}

	// Standard batch insert (builds HNSW graph for searchability)
	ctx := context.Background()
	for start := 0; start < numVecs; start += batchSize {
		end := start + batchSize
		if end > numVecs {
			end = numVecs
		}

		batchVecs := data[start:end]
		batchMds := make([]metadata.Document, end-start)
		for i := range batchMds {
			batchMds[i] = metadata.Document{"bucket": metadata.Int(buckets[start+i])}
		}

		ids, err := e.BatchInsert(ctx, batchVecs, batchMds, nil)
		if err != nil {
			b.Fatalf("batch insert failed: %v", err)
		}

		copy(pks[start:end], ids)

		if (start+batchSize)%25000 == 0 || end == numVecs {
			b.Logf("  loaded %d/%d vectors", end, numVecs)
		}
	}
	b.Logf("Load complete. Starting benchmarks...")

	queries := make([][]float32, numQueries)
	for i := 0; i < numQueries; i++ {
		queries[i] = make([]float32, dim)
		rng.FillUniform(queries[i])
	}

	suites := []struct {
		name string
		sel  float64
	}{
		{name: "01pct", sel: 0.01},
		{name: "05pct", sel: 0.05},
		{name: "10pct", sel: 0.10},
		{name: "50pct", sel: 0.50},
		{name: "90pct", sel: 0.90},
	}

	for _, suite := range suites {
		threshold := int64(float64(bucketCount) * suite.sel)
		if threshold < 1 {
			threshold = 1
		}
		if threshold > bucketCount {
			threshold = bucketCount
		}

		filter := metadata.NewFilterSet(metadata.Filter{
			Key:      "bucket",
			Operator: metadata.OpLessThan,
			Value:    metadata.Int(threshold),
		})

		// Compute ground truth once per selectivity level
		filteredData := make([][]float32, 0, numVecs)
		filteredPKs := make([]model.ID, 0, numVecs)
		for i := 0; i < numVecs; i++ {
			if buckets[i] < threshold {
				filteredData = append(filteredData, data[i])
				filteredPKs = append(filteredPKs, pks[i])
			}
		}
		matches := float64(len(filteredData))
		selectivity := matches / float64(numVecs)

		// Auto mode (engine decides pre vs post filter)
		b.Run(fmt.Sprintf("d%d_n%dk/%s", dim, numVecs/1000, suite.name), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				for _, q := range queries {
					_, _ = e.Search(ctx, q, k, vecgo.WithFilter(filter), vecgo.WithoutData())
				}
			}
			b.StopTimer()

			var sum float64
			for _, q := range queries {
				truth := exactTopK_L2_WithIDs(filteredData, filteredPKs, q, k)
				res, _ := e.Search(ctx, q, k, vecgo.WithFilter(filter), vecgo.WithoutData())
				sum += recallAtK(res, truth)
			}
			b.ReportMetric(matches, "matches")
			b.ReportMetric(selectivity, "selectivity")
			b.ReportMetric(sum/float64(len(queries)), "recall@10")
		})
	}
}
