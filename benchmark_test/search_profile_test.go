//go:build profile

package benchmark_test

import (
	"context"
	"fmt"
	"math"
	"os"
	"runtime"
	"runtime/pprof"
	"testing"

	"github.com/hupe1980/vecgo"
	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/testutil"
)

// TestSearchProfile creates a clean CPU profile of ONLY search operations.
// This avoids the -cpuprofile artifact where setup (inserts) dominate the profile.
//
// METHODOLOGY for clean profiles:
// 1. Setup index (not profiled)
// 2. WARMUP: Run queries to warm CPU caches, branch predictors, Go runtime
// 3. runtime.GC(): Clear allocation pressure from setup/warmup
// 4. pprof.StartCPUProfile(): Start profiling AFTER warmup
// 5. Run measured iterations
// 6. pprof.StopCPUProfile(): Stop profiling
//
// This ensures the profile reflects steady-state performance, not cold-start.
//
// Usage:
//
//	go test -tags=profile -run=TestSearchProfile -v ./benchmark_test/
//
// Then analyze:
//
//	go tool pprof -top /tmp/search_pure_sel70.prof
//	go tool pprof -web /tmp/search_pure_sel70.prof
//	go tool pprof -list=Search /tmp/search_pure_sel70.prof
func TestSearchProfile(t *testing.T) {
	const dim = 128
	const numVecs = 50_000
	const bucketCount = 100
	const k = 10
	const numQueries = 100
	const batchSize = 1000
	const warmupIterations = 20 // Warmup iterations (not profiled)
	const numIterations = 100   // Profiled iterations

	ctx := context.Background()
	rng := testutil.NewRNG(42)

	// ========== SETUP (not profiled) ==========
	t.Log("Setting up index (not profiled)...")

	dir := t.TempDir()
	e, err := vecgo.Open(ctx, vecgo.Local(dir), vecgo.Create(dim, vecgo.MetricL2),
		vecgo.WithCompactionThreshold(math.MaxInt),
		vecgo.WithFlushConfig(vecgo.FlushConfig{MaxMemTableSize: 64 << 20}),
		vecgo.WithDiskANNThreshold(numVecs+1),
		vecgo.WithMemoryLimit(0),
	)
	if err != nil {
		t.Fatalf("failed to open: %v", err)
	}
	defer e.Close()

	// Insert vectors
	buckets := make([]int64, numVecs)
	for i := range numVecs {
		buckets[i] = int64(i) % bucketCount
	}

	for start := 0; start < numVecs; start += batchSize {
		end := min(start+batchSize, numVecs)
		batchVecs := make([][]float32, end-start)
		batchMds := make([]metadata.Document, end-start)
		for i := range batchVecs {
			vec := make([]float32, dim)
			rng.FillUniform(vec)
			batchVecs[i] = vec
			batchMds[i] = metadata.Document{"bucket": metadata.Int(buckets[start+i])}
		}
		if _, err := e.BatchInsert(ctx, batchVecs, batchMds, nil); err != nil {
			t.Fatalf("batch insert failed: %v", err)
		}
	}
	t.Logf("Inserted %d vectors", numVecs)

	// Create queries
	queries := make([][]float32, numQueries)
	for i := range numQueries {
		queries[i] = make([]float32, dim)
		rng.FillUniform(queries[i])
	}

	// Test different selectivities
	selectivities := []float64{0.01, 0.10, 0.50, 0.70, 0.90}

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

		profilePath := fmt.Sprintf("/tmp/search_pure_sel%.0f.prof", sel*100)

		t.Run(fmt.Sprintf("sel=%.0f%%", sel*100), func(t *testing.T) {
			t.Logf("Profiling search at %.0f%% selectivity (%d matches)...", sel*100, matches)

			// ========== WARMUP (not profiled) ==========
			// Critical: Warm CPU caches, branch predictors, Go runtime pools
			t.Logf("  Warming up (%d iterations)...", warmupIterations)
			for iter := 0; iter < warmupIterations; iter++ {
				for _, q := range queries {
					_, _ = e.Search(ctx, q, k, vecgo.WithFilter(filter), vecgo.WithoutData())
				}
			}

			// ========== CLEAR GC PRESSURE ==========
			// Force GC to clear allocations from setup/warmup
			runtime.GC()

			// ========== START PROFILING (search only, steady-state) ==========
			f, err := os.Create(profilePath)
			if err != nil {
				t.Fatalf("failed to create profile: %v", err)
			}

			if err := pprof.StartCPUProfile(f); err != nil {
				f.Close()
				t.Fatalf("failed to start profile: %v", err)
			}

			// Run search iterations (profiled)
			for iter := 0; iter < numIterations; iter++ {
				for _, q := range queries {
					_, _ = e.Search(ctx, q, k, vecgo.WithFilter(filter), vecgo.WithoutData())
				}
			}

			// ========== STOP PROFILING ==========
			pprof.StopCPUProfile()
			f.Close()

			t.Logf("Profile written to: %s", profilePath)
			t.Logf("Analyze with: go tool pprof -top %s", profilePath)
		})
	}

	// Also profile no-filter case
	t.Run("no_filter", func(t *testing.T) {
		profilePath := "/tmp/search_pure_nofilter.prof"
		t.Log("Profiling search without filter...")

		// Warmup
		t.Logf("  Warming up (%d iterations)...", warmupIterations)
		for iter := 0; iter < warmupIterations; iter++ {
			for _, q := range queries {
				_, _ = e.Search(ctx, q, k, vecgo.WithoutData())
			}
		}

		// Clear GC
		runtime.GC()

		f, err := os.Create(profilePath)
		if err != nil {
			t.Fatalf("failed to create profile: %v", err)
		}

		if err := pprof.StartCPUProfile(f); err != nil {
			f.Close()
			t.Fatalf("failed to start profile: %v", err)
		}

		for iter := 0; iter < numIterations; iter++ {
			for _, q := range queries {
				_, _ = e.Search(ctx, q, k, vecgo.WithoutData())
			}
		}

		pprof.StopCPUProfile()
		f.Close()

		t.Logf("Profile written to: %s", profilePath)
	})
}

// TestSearchMemProfile creates a clean MEMORY profile of search operations.
// This is useful for finding allocation hotspots.
//
// Usage:
//
//	go test -tags=profile -run=TestSearchMemProfile -v ./benchmark_test/
//
// Then analyze:
//
//	go tool pprof -alloc_objects /tmp/search_mem_sel10.prof
//	go tool pprof -alloc_space /tmp/search_mem_sel10.prof
func TestSearchMemProfile(t *testing.T) {
	const dim = 128
	const numVecs = 50_000
	const bucketCount = 100
	const k = 10
	const numQueries = 100
	const batchSize = 1000
	const warmupIterations = 20
	const numIterations = 50 // Fewer iterations for memory profile

	ctx := context.Background()
	rng := testutil.NewRNG(42)

	// ========== SETUP ==========
	t.Log("Setting up index...")

	dir := t.TempDir()
	e, err := vecgo.Open(ctx, vecgo.Local(dir), vecgo.Create(dim, vecgo.MetricL2),
		vecgo.WithCompactionThreshold(math.MaxInt),
		vecgo.WithFlushConfig(vecgo.FlushConfig{MaxMemTableSize: 64 << 20}),
		vecgo.WithDiskANNThreshold(numVecs+1),
		vecgo.WithMemoryLimit(0),
	)
	if err != nil {
		t.Fatalf("failed to open: %v", err)
	}
	defer e.Close()

	buckets := make([]int64, numVecs)
	for i := range numVecs {
		buckets[i] = int64(i) % bucketCount
	}

	for start := 0; start < numVecs; start += batchSize {
		end := min(start+batchSize, numVecs)
		batchVecs := make([][]float32, end-start)
		batchMds := make([]metadata.Document, end-start)
		for i := range batchVecs {
			vec := make([]float32, dim)
			rng.FillUniform(vec)
			batchVecs[i] = vec
			batchMds[i] = metadata.Document{"bucket": metadata.Int(buckets[start+i])}
		}
		if _, err := e.BatchInsert(ctx, batchVecs, batchMds, nil); err != nil {
			t.Fatalf("batch insert failed: %v", err)
		}
	}
	t.Logf("Inserted %d vectors", numVecs)

	queries := make([][]float32, numQueries)
	for i := range numQueries {
		queries[i] = make([]float32, dim)
		rng.FillUniform(queries[i])
	}

	selectivities := []float64{0.01, 0.10, 0.50}

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

		profilePath := fmt.Sprintf("/tmp/search_mem_sel%.0f.prof", sel*100)

		t.Run(fmt.Sprintf("sel=%.0f%%", sel*100), func(t *testing.T) {
			t.Logf("Memory profiling at %.0f%% selectivity (%d matches)...", sel*100, matches)

			// Warmup
			for iter := 0; iter < warmupIterations; iter++ {
				for _, q := range queries {
					_, _ = e.Search(ctx, q, k, vecgo.WithFilter(filter), vecgo.WithoutData())
				}
			}

			// Clear GC and reset memory stats
			runtime.GC()

			// Run iterations (memory profile captures all allocations)
			for iter := 0; iter < numIterations; iter++ {
				for _, q := range queries {
					_, _ = e.Search(ctx, q, k, vecgo.WithFilter(filter), vecgo.WithoutData())
				}
			}

			// Write heap profile
			runtime.GC() // Ensure all allocations are visible
			f, err := os.Create(profilePath)
			if err != nil {
				t.Fatalf("failed to create profile: %v", err)
			}
			if err := pprof.WriteHeapProfile(f); err != nil {
				f.Close()
				t.Fatalf("failed to write heap profile: %v", err)
			}
			f.Close()

			t.Logf("Memory profile written to: %s", profilePath)
			t.Logf("Analyze with: go tool pprof -alloc_objects %s", profilePath)
		})
	}
}
