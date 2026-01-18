package benchmark_test

import (
	"context"
	"fmt"
	"math"
	"runtime"
	"testing"

	"github.com/hupe1980/vecgo"
	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/model"
	"github.com/hupe1980/vecgo/testutil"
)

// ============================================================================
// BENCHMARK METHODOLOGY (CRITICAL FOR CLEAN MEASUREMENTS)
// ============================================================================
//
// Go's b.ResetTimer() resets time and allocations but doesn't:
// 1. Warm CPU caches (L1/L2/L3)
// 2. Warm branch predictors
// 3. Warm Go runtime caches
// 4. Clear GC pressure from setup
//
// Our methodology:
// 1. Setup (before warmup)
// 2. WARMUP: Run WarmupIterations queries to warm caches
// 3. runtime.GC() to clear setup allocations
// 4. b.ResetTimer() to reset counters
// 5. Measure b.N iterations
// 6. NO b.StopTimer() - let benchmark complete naturally
// 7. Validation done in separate benchmark or deferred
//
// This eliminates:
// - First-iteration cold cache effects
// - GC pauses from setup allocations
// - b.StopTimer()/b.StartTimer() overhead (~50-100ns each)
// - Timer drift from post-loop validation

// BenchmarkSearchOnly runs search-only benchmarks on pre-built indexes.
// The index is built once per sub-benchmark and reused across all iterations.
func BenchmarkSearchOnly(b *testing.B) {
	const dim = 128
	const numVecs = 50_000
	const bucketCount = 100
	const k = 10
	const numQueries = 100
	const batchSize = 1000
	const selectivity = 0.10

	threshold := int64(float64(bucketCount) * selectivity)
	ctx := context.Background()

	filter := metadata.NewFilterSet(metadata.Filter{
		Key:      "bucket",
		Operator: metadata.OpLessThan,
		Value:    metadata.Int(threshold),
	})

	// ========== UNIFORM SEARCH BENCHMARK ==========
	b.Run("uniform", func(b *testing.B) {
		rng := testutil.NewRNG(1)
		dir := b.TempDir()
		e, err := vecgo.Open(ctx, vecgo.Local(dir), vecgo.Create(dim, vecgo.MetricL2),
			vecgo.WithCompactionThreshold(math.MaxInt),
			vecgo.WithFlushConfig(vecgo.FlushConfig{MaxMemTableSize: 64 << 20}),
			vecgo.WithDiskANNThreshold(numVecs+1),
			vecgo.WithMemoryLimit(0),
		)
		if err != nil {
			b.Fatalf("failed to open: %v", err)
		}
		defer e.Close()

		data := make([][]float32, numVecs)
		pks := make([]model.ID, numVecs)
		buckets := make([]int64, numVecs)

		for i := range numVecs {
			vec := make([]float32, dim)
			rng.FillUniform(vec)
			data[i] = vec
			buckets[i] = int64(i) % bucketCount
		}

		for start := 0; start < numVecs; start += batchSize {
			end := min(start+batchSize, numVecs)
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
		}

		queries := make([][]float32, numQueries)
		for i := range numQueries {
			queries[i] = make([]float32, dim)
			rng.FillUniform(queries[i])
		}

		// Pre-compute ground truth BEFORE measurement
		filteredData := make([][]float32, 0, numVecs/10)
		filteredPKs := make([]model.ID, 0, numVecs/10)
		for i := range numVecs {
			if buckets[i] < threshold {
				filteredData = append(filteredData, data[i])
				filteredPKs = append(filteredPKs, pks[i])
			}
		}

		// PHASE 1: WARMUP (critical for clean measurements)
		for i := 0; i < WarmupIterations; i++ {
			q := queries[i%len(queries)]
			_, _ = e.Search(ctx, q, k, vecgo.WithFilter(filter), vecgo.WithoutData())
		}

		// PHASE 2: Clear GC pressure from setup
		runtime.GC()

		// PHASE 3: Measure
		b.ReportAllocs()
		b.ResetTimer()

		for i := 0; i < b.N; i++ {
			q := queries[i%len(queries)]
			_, _ = e.Search(ctx, q, k, vecgo.WithFilter(filter), vecgo.WithoutData())
		}
		// NO b.StopTimer() - let benchmark complete naturally

		// Report metrics (computed before measurement, just reporting values)
		b.ReportMetric(float64(len(filteredData)), "matches")
		b.ReportMetric(float64(len(filteredData))/float64(numVecs), "selectivity")

		// Recall validation in deferred function (after benchmark timing)
		b.Cleanup(func() {
			var sumRecall float64
			for _, q := range queries[:min(20, len(queries))] {
				truth := exactTopK_L2_WithIDs(filteredData, filteredPKs, q, k)
				res, _ := e.Search(ctx, q, k, vecgo.WithFilter(filter), vecgo.WithoutData())
				sumRecall += recallAtK(res, truth)
			}
			b.ReportMetric(sumRecall/float64(min(20, len(queries))), "recall@10")
		})
	})

	// ========== REALISTIC SEARCH BENCHMARK ==========
	b.Run("realistic", func(b *testing.B) {
		rng := testutil.NewRNG(42)
		dir := b.TempDir()
		e, err := vecgo.Open(ctx, vecgo.Local(dir), vecgo.Create(dim, vecgo.MetricL2),
			vecgo.WithCompactionThreshold(math.MaxInt),
			vecgo.WithFlushConfig(vecgo.FlushConfig{MaxMemTableSize: 64 << 20}),
			vecgo.WithDiskANNThreshold(numVecs+1),
			vecgo.WithMemoryLimit(0),
		)
		if err != nil {
			b.Fatalf("failed to open: %v", err)
		}
		defer e.Close()

		buckets := rng.ZipfBuckets(numVecs, bucketCount, 1.5)
		present := rng.SparseMetadata(numVecs, 0.30)
		data := rng.ClusteredVectorsWithBuckets(numVecs, dim, bucketCount, buckets, 0.15)

		pks := make([]model.ID, numVecs)
		for start := 0; start < numVecs; start += batchSize {
			end := min(start+batchSize, numVecs)
			batchVecs := data[start:end]
			batchMds := make([]metadata.Document, end-start)
			for i := range batchMds {
				if present[start+i] {
					batchMds[i] = metadata.Document{"bucket": metadata.Int(buckets[start+i])}
				} else {
					batchMds[i] = metadata.Document{}
				}
			}
			ids, err := e.BatchInsert(ctx, batchVecs, batchMds, nil)
			if err != nil {
				b.Fatalf("batch insert failed: %v", err)
			}
			copy(pks[start:end], ids)
		}

		// Pre-compute ground truth BEFORE measurement
		filteredData := make([][]float32, 0)
		filteredPKs := make([]model.ID, 0)
		for i := range numVecs {
			if present[i] && int64(buckets[i]) < threshold {
				filteredData = append(filteredData, data[i])
				filteredPKs = append(filteredPKs, pks[i])
			}
		}
		actualSel := float64(len(filteredData)) / float64(numVecs)

		queries := make([][]float32, numQueries)
		for i := range numQueries {
			queries[i] = make([]float32, dim)
			rng.FillUniform(queries[i])
		}

		// PHASE 1: WARMUP
		for i := 0; i < WarmupIterations; i++ {
			q := queries[i%len(queries)]
			_, _ = e.Search(ctx, q, k, vecgo.WithFilter(filter), vecgo.WithoutData())
		}

		// PHASE 2: Clear GC pressure
		runtime.GC()

		// PHASE 3: Measure
		b.ReportAllocs()
		b.ResetTimer()

		for i := 0; i < b.N; i++ {
			q := queries[i%len(queries)]
			_, _ = e.Search(ctx, q, k, vecgo.WithFilter(filter), vecgo.WithoutData())
		}

		b.ReportMetric(float64(len(filteredData)), "matches")
		b.ReportMetric(actualSel, "selectivity")

		b.Cleanup(func() {
			var sumRecall float64
			for _, q := range queries[:min(20, len(queries))] {
				truth := exactTopK_L2_WithIDs(filteredData, filteredPKs, q, k)
				res, _ := e.Search(ctx, q, k, vecgo.WithFilter(filter), vecgo.WithoutData())
				sumRecall += recallAtK(res, truth)
			}
			b.ReportMetric(sumRecall/float64(min(20, len(queries))), "recall@10")
		})
	})

	// ========== NO FILTER BASELINE ==========
	b.Run("no_filter", func(b *testing.B) {
		rng := testutil.NewRNG(1)
		dir := b.TempDir()
		e, err := vecgo.Open(ctx, vecgo.Local(dir), vecgo.Create(dim, vecgo.MetricL2),
			vecgo.WithCompactionThreshold(math.MaxInt),
			vecgo.WithFlushConfig(vecgo.FlushConfig{MaxMemTableSize: 64 << 20}),
			vecgo.WithDiskANNThreshold(numVecs+1),
			vecgo.WithMemoryLimit(0),
		)
		if err != nil {
			b.Fatalf("failed to open: %v", err)
		}
		defer e.Close()

		data := make([][]float32, numVecs)
		pks := make([]model.ID, numVecs)

		for i := range numVecs {
			vec := make([]float32, dim)
			rng.FillUniform(vec)
			data[i] = vec
		}

		for start := 0; start < numVecs; start += batchSize {
			end := min(start+batchSize, numVecs)
			batchVecs := data[start:end]
			batchMds := make([]metadata.Document, end-start)
			for i := range batchMds {
				batchMds[i] = metadata.Document{"bucket": metadata.Int(int64(start + i))}
			}
			ids, err := e.BatchInsert(ctx, batchVecs, batchMds, nil)
			if err != nil {
				b.Fatalf("batch insert failed: %v", err)
			}
			copy(pks[start:end], ids)
		}

		queries := make([][]float32, numQueries)
		for i := range numQueries {
			queries[i] = make([]float32, dim)
			rng.FillUniform(queries[i])
		}

		// PHASE 1: WARMUP
		for i := 0; i < WarmupIterations; i++ {
			q := queries[i%len(queries)]
			_, _ = e.Search(ctx, q, k, vecgo.WithoutData())
		}

		// PHASE 2: Clear GC pressure
		runtime.GC()

		// PHASE 3: Measure
		b.ReportAllocs()
		b.ResetTimer()

		for i := 0; i < b.N; i++ {
			q := queries[i%len(queries)]
			_, _ = e.Search(ctx, q, k, vecgo.WithoutData())
		}

		b.Cleanup(func() {
			var sumRecall float64
			for _, q := range queries[:min(20, len(queries))] {
				truth := exactTopK_L2_WithIDs(data, pks, q, k)
				res, _ := e.Search(ctx, q, k, vecgo.WithoutData())
				sumRecall += recallAtK(res, truth)
			}
			b.ReportMetric(sumRecall/float64(min(20, len(queries))), "recall@10")
		})
	})
}

// BenchmarkSearchOnlySelectivity tests search at various selectivities.
func BenchmarkSearchOnlySelectivity(b *testing.B) {
	const dim = 128
	const numVecs = 50_000
	const bucketCount = 100
	const k = 10
	const numQueries = 100
	const batchSize = 1000

	ctx := context.Background()
	rng := testutil.NewRNG(123)

	dir := b.TempDir()
	e, err := vecgo.Open(ctx, vecgo.Local(dir), vecgo.Create(dim, vecgo.MetricL2),
		vecgo.WithCompactionThreshold(math.MaxInt),
		vecgo.WithFlushConfig(vecgo.FlushConfig{MaxMemTableSize: 64 << 20}),
		vecgo.WithDiskANNThreshold(numVecs+1),
		vecgo.WithMemoryLimit(0),
	)
	if err != nil {
		b.Fatalf("failed to open: %v", err)
	}
	defer e.Close()

	buckets := make([]int64, numVecs)
	data := make([][]float32, numVecs)
	for i := range numVecs {
		vec := make([]float32, dim)
		rng.FillUniform(vec)
		data[i] = vec
		buckets[i] = int64(i) % bucketCount
	}

	// Batch insert for fast setup
	for start := 0; start < numVecs; start += batchSize {
		end := min(start+batchSize, numVecs)
		batchVecs := data[start:end]
		batchMds := make([]metadata.Document, end-start)
		for i := range batchMds {
			batchMds[i] = metadata.Document{"bucket": metadata.Int(buckets[start+i])}
		}
		_, err := e.BatchInsert(ctx, batchVecs, batchMds, nil)
		if err != nil {
			b.Fatalf("batch insert failed: %v", err)
		}
	}

	queries := make([][]float32, numQueries)
	for i := range numQueries {
		queries[i] = make([]float32, dim)
		rng.FillUniform(queries[i])
	}

	selectivities := []float64{0.01, 0.05, 0.10, 0.30, 0.50, 0.70, 0.90}

	for _, sel := range selectivities {
		threshold := int64(float64(bucketCount) * sel)
		filter := metadata.NewFilterSet(metadata.Filter{
			Key:      "bucket",
			Operator: metadata.OpLessThan,
			Value:    metadata.Int(threshold),
		})

		matches := 0
		for _, bucket := range buckets {
			if bucket < threshold {
				matches++
			}
		}
		actualSel := float64(matches) / float64(numVecs)

		b.Run(fmt.Sprintf("sel=%.0f%%", sel*100), func(b *testing.B) {
			// PHASE 1: WARMUP
			for i := 0; i < WarmupIterations; i++ {
				q := queries[i%len(queries)]
				_, _ = e.Search(ctx, q, k, vecgo.WithFilter(filter), vecgo.WithoutData())
			}

			// PHASE 2: Clear GC pressure
			runtime.GC()

			// PHASE 3: Measure
			b.ReportAllocs()
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				q := queries[i%len(queries)]
				_, _ = e.Search(ctx, q, k, vecgo.WithFilter(filter), vecgo.WithoutData())
			}

			b.ReportMetric(actualSel, "selectivity")
			b.ReportMetric(float64(matches), "matches")
		})
	}
}
