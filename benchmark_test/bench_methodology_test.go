package benchmark_test

import (
	"context"
	"math"
	"runtime"
	"testing"

	"github.com/hupe1980/vecgo"
	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/model"
)

// ============================================================================
// BENCHMARK METHODOLOGY — Clean, Reproducible Measurements
// ============================================================================
//
// This file provides utilities for noise-free benchmarks. Key principles:
//
// 1. WARMUP PHASE: Run N iterations before measurement to:
//    - Warm CPU branch predictors
//    - Populate caches (L1/L2/L3, Go runtime, internal caches)
//    - Stabilize JIT-like optimizations
//
// 2. GC CONTROL: Force GC before measurement to:
//    - Clear allocation pressure from setup
//    - Prevent GC pauses during measurement
//
// 3. ONE QUERY PER ITERATION: Each b.N iteration = exactly 1 query
//    - Enables accurate per-query latency
//    - Allocations are per-query, not per-batch
//
// 4. NO StopTimer/StartTimer: These have overhead (~50-100ns) and
//    cause timing instability. Use RunParallel or separate benchmarks.
//
// 5. VALIDATION SEPARATE: Recall computation is done in a separate
//    benchmark or after b.StopTimer() outside the measurement loop.
//
// Usage:
//
//   func BenchmarkSearch(b *testing.B) {
//       // Setup
//       db, queries := setupDatabase(b)
//       defer db.Close()
//
//       // Use BenchLoop for clean measurements
//       BenchLoop(b, len(queries), func(i int) {
//           _, _ = db.Search(ctx, queries[i], k)
//       })
//   }

// WarmupIterations is the number of warmup iterations before measurement.
// This should be enough to warm caches and branch predictors.
const WarmupIterations = 10

// BenchLoop runs a benchmark with proper methodology:
// 1. Warmup phase (WarmupIterations)
// 2. GC to clear allocation pressure
// 3. Reset timer
// 4. Run b.N iterations
//
// The queryCount parameter is used to cycle through queries (i % queryCount).
// The fn receives the current iteration index.
func BenchLoop(b *testing.B, queryCount int, fn func(i int)) {
	b.Helper()

	// Phase 1: Warmup
	for i := 0; i < WarmupIterations; i++ {
		fn(i % queryCount)
	}

	// Phase 2: GC to clear setup allocations
	runtime.GC()

	// Phase 3: Reset and run
	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		fn(i % queryCount)
	}
}

// BenchLoopFiltered runs a filtered search benchmark with proper methodology.
// Same as BenchLoop but accepts a filter.
func BenchLoopFiltered(b *testing.B, db *vecgo.DB, queries [][]float32, k int, filter *metadata.FilterSet) {
	b.Helper()
	ctx := context.Background()

	// Phase 1: Warmup
	for i := 0; i < WarmupIterations; i++ {
		q := queries[i%len(queries)]
		_, _ = db.Search(ctx, q, k, vecgo.WithFilter(filter), vecgo.WithoutData())
	}

	// Phase 2: GC
	runtime.GC()

	// Phase 3: Measure
	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		q := queries[i%len(queries)]
		_, _ = db.Search(ctx, q, k, vecgo.WithFilter(filter), vecgo.WithoutData())
	}
}

// BenchLoopWithCallback runs a benchmark with an optional post-measurement callback.
// The callback is called AFTER measurement completes and should be used for validation.
func BenchLoopWithCallback(b *testing.B, queryCount int, fn func(i int), postMeasure func()) {
	b.Helper()

	// Phase 1: Warmup
	for i := 0; i < WarmupIterations; i++ {
		fn(i % queryCount)
	}

	// Phase 2: GC
	runtime.GC()

	// Phase 3: Measure
	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		fn(i % queryCount)
	}

	// Phase 4: Post-measurement validation (not timed)
	if postMeasure != nil {
		postMeasure()
	}
}

// ValidateRecall computes and reports recall@k separately from the benchmark.
// Call this AFTER the benchmark loop completes.
func ValidateRecall(b *testing.B, db *vecgo.DB, queries [][]float32, k int, filter *metadata.FilterSet, groundTruth [][]model.ID) {
	b.Helper()
	ctx := context.Background()

	if len(groundTruth) == 0 {
		return
	}

	var sumRecall float64
	numSamples := min(len(queries), len(groundTruth))
	for i := 0; i < numSamples; i++ {
		var results []model.Candidate
		var err error
		if filter != nil {
			results, err = db.Search(ctx, queries[i], k, vecgo.WithFilter(filter), vecgo.WithoutData())
		} else {
			results, err = db.Search(ctx, queries[i], k, vecgo.WithoutData())
		}
		if err != nil {
			continue
		}
		sumRecall += recallAtKWithIDs(results, groundTruth[i])
	}
	b.ReportMetric(sumRecall/float64(numSamples), "recall@10")
}

// recallAtKWithIDs computes recall given results and ground truth IDs.
func recallAtKWithIDs(results []model.Candidate, truth []model.ID) float64 {
	if len(truth) == 0 {
		return 0
	}
	set := make(map[model.ID]struct{}, len(truth))
	for _, pk := range truth {
		set[pk] = struct{}{}
	}
	var hit int
	for _, c := range results {
		if _, ok := set[c.ID]; ok {
			hit++
		}
	}
	return float64(hit) / float64(len(truth))
}

// BenchmarkSetup holds reusable benchmark setup.
type BenchmarkSetup struct {
	DB      *vecgo.DB
	Data    [][]float32
	PKs     []model.ID
	Buckets []int64
	Queries [][]float32
	Dim     int
	NumVecs int
}

// SetupFilteredBenchmark creates a standardized benchmark setup.
func SetupFilteredBenchmark(b *testing.B, dim, numVecs, bucketCount, numQueries int) *BenchmarkSetup {
	b.Helper()

	dir := b.TempDir()
	db, err := vecgo.Open(context.Background(), vecgo.Local(dir), vecgo.Create(dim, vecgo.MetricL2),
		vecgo.WithCompactionThreshold(math.MaxInt),                          // Disable compaction
		vecgo.WithFlushConfig(vecgo.FlushConfig{MaxMemTableSize: 64 << 20}), // 64MB memtable
		vecgo.WithDiskANNThreshold(numVecs+1),                               // Keep in HNSW
		vecgo.WithMemoryLimit(0),                                            // No memory limit
	)
	if err != nil {
		b.Fatalf("failed to open: %v", err)
	}

	rng := newDeterministicRNG(benchSeed)

	// Generate data
	data := make([][]float32, numVecs)
	buckets := make([]int64, numVecs)
	for i := range numVecs {
		vec := make([]float32, dim)
		rng.FillUniform(vec)
		data[i] = vec
		buckets[i] = int64(i % bucketCount)
	}

	// Batch insert
	ctx := context.Background()
	pks := make([]model.ID, numVecs)
	const batchSize = 1000
	for start := 0; start < numVecs; start += batchSize {
		end := min(start+batchSize, numVecs)
		batchVecs := data[start:end]
		batchMds := make([]metadata.Document, end-start)
		for i := range batchMds {
			batchMds[i] = metadata.Document{"bucket": metadata.Int(buckets[start+i])}
		}
		ids, err := db.BatchInsert(ctx, batchVecs, batchMds, nil)
		if err != nil {
			b.Fatalf("batch insert failed: %v", err)
		}
		copy(pks[start:end], ids)
	}

	// Generate queries
	queries := make([][]float32, numQueries)
	for i := range numQueries {
		queries[i] = make([]float32, dim)
		rng.FillUniform(queries[i])
	}

	return &BenchmarkSetup{
		DB:      db,
		Data:    data,
		PKs:     pks,
		Buckets: buckets,
		Queries: queries,
		Dim:     dim,
		NumVecs: numVecs,
	}
}

// Close releases benchmark resources.
func (s *BenchmarkSetup) Close() {
	if s.DB != nil {
		s.DB.Close()
	}
}

// ComputeGroundTruth computes exact top-k for filtered queries.
func (s *BenchmarkSetup) ComputeGroundTruth(k int, bucketThreshold int64) [][]model.ID {
	// Filter data by bucket
	filteredData := make([][]float32, 0)
	filteredPKs := make([]model.ID, 0)
	for i := range s.NumVecs {
		if s.Buckets[i] < bucketThreshold {
			filteredData = append(filteredData, s.Data[i])
			filteredPKs = append(filteredPKs, s.PKs[i])
		}
	}

	// Compute ground truth for each query
	groundTruth := make([][]model.ID, len(s.Queries))
	for i, q := range s.Queries {
		groundTruth[i] = exactTopK_L2_WithIDs(filteredData, filteredPKs, q, k)
	}
	return groundTruth
}

// deterministicRNG wraps testutil.RNG for benchmarks.
type deterministicRNG struct {
	seed int64
	idx  int
}

func newDeterministicRNG(seed int64) *deterministicRNG {
	return &deterministicRNG{seed: seed}
}

func (r *deterministicRNG) FillUniform(v []float32) {
	// Simple deterministic fill using LCG
	x := uint64(r.seed) + uint64(r.idx)*6364136223846793005
	for i := range v {
		x = x*6364136223846793005 + 1442695040888963407
		v[i] = float32(x>>33) / float32(1<<31) // [0, 1)
	}
	r.idx++
}
