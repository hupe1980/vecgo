package benchmark_test

import (
	"context"
	"math"
	"testing"

	"github.com/hupe1980/vecgo"
	"github.com/hupe1980/vecgo/model"
	"github.com/hupe1980/vecgo/testutil"
)

// ============================================================================
// Benchmark Configuration
// ============================================================================

// Standard dimensions used across benchmarks for consistency.
const (
	dimSmall  = 128  // Fast CI benchmarks
	dimMedium = 768  // OpenAI text-embedding-3-small, Cohere v3
	dimLarge  = 1536 // OpenAI text-embedding-3-large
)

// Standard dataset sizes.
const (
	sizeSmall  = 10_000  // Quick iteration
	sizeMedium = 50_000  // Default CI
	sizeLarge  = 100_000 // Production-scale
)

// Seed for deterministic benchmarks - enables reproducible comparisons.
const benchSeed = 42

// ============================================================================
// Benchmark Helpers
// ============================================================================

// BenchEngine wraps DB creation with standardized config.
type BenchEngine struct {
	*vecgo.DB
	dir string
}

// OpenBenchEngine creates a DB optimized for benchmark isolation.
func OpenBenchEngine(b *testing.B, dim int, opts ...vecgo.Option) *BenchEngine {
	dir := b.TempDir()
	defaultOpts := []vecgo.Option{
		vecgo.Create(dim, vecgo.MetricL2),
		vecgo.WithCompactionThreshold(math.MaxInt),                          // Disable auto-compaction
		vecgo.WithFlushConfig(vecgo.FlushConfig{MaxMemTableSize: 64 << 20}), // 1GB memtable
		vecgo.WithMemoryLimit(0),                                            // No memory semaphore
	}
	allOpts := append(defaultOpts, opts...)
	db, err := vecgo.Open(context.Background(), vecgo.Local(dir), allOpts...)
	if err != nil {
		b.Fatalf("failed to open engine: %v", err)
	}
	return &BenchEngine{DB: db, dir: dir}
}

// LoadData inserts random vectors and returns them for ground truth computation.
func (e *BenchEngine) LoadData(b *testing.B, n, dim int) ([][]float32, []model.ID) {
	b.Helper()
	rng := testutil.NewRNG(benchSeed)
	data := make([][]float32, n)
	pks := make([]model.ID, n)

	const batchSize = 1000
	for start := 0; start < n; start += batchSize {
		end := start + batchSize
		if end > n {
			end = n
		}

		batchVecs := make([][]float32, end-start)
		for i := range batchVecs {
			vec := make([]float32, dim)
			rng.FillUniform(vec)
			batchVecs[i] = vec
			data[start+i] = vec
		}

		ids, err := e.BatchInsert(context.Background(), batchVecs, nil, nil)
		if err != nil {
			b.Fatalf("batch insert failed: %v", err)
		}
		copy(pks[start:end], ids)
	}
	return data, pks
}

// MakeQueries generates n random query vectors.
func MakeQueries(n, dim int) [][]float32 {
	rng := testutil.NewRNG(benchSeed + 1) // Different seed from data
	queries := make([][]float32, n)
	for i := range queries {
		q := make([]float32, dim)
		rng.FillUniform(q)
		queries[i] = q
	}
	return queries
}

// WarmupSearch runs a few search iterations to warm caches.
func (e *BenchEngine) WarmupSearch(b *testing.B, queries [][]float32, k int) {
	b.Helper()
	ctx := context.Background()
	for _, q := range queries[:min(3, len(queries))] {
		_, _ = e.Search(ctx, q, k)
	}
}
