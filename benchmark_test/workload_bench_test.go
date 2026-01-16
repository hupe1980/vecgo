package benchmark_test

import (
	"context"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/hupe1980/vecgo"
	"github.com/hupe1980/vecgo/testutil"
)

// ============================================================================
// Workload Benchmarks - Realistic Production Patterns
// ============================================================================

// BenchmarkMixedWorkload simulates a real production workload with concurrent
// reads and writes at various ratios.
func BenchmarkMixedWorkload(b *testing.B) {
	// Read:Write ratios (% reads)
	ratios := []int{50, 80, 95, 99}
	const dim = dimMedium
	const initialSize = sizeSmall // Pre-populate with 10k vectors
	const k = 10

	for _, readPct := range ratios {
		b.Run("read="+strconv.Itoa(readPct)+"%", func(b *testing.B) {
			e := OpenBenchEngine(b, dim)
			defer e.Close()

			// Pre-populate
			e.LoadData(b, initialSize, dim)
			queries := MakeQueries(100, dim)
			e.WarmupSearch(b, queries, k)

			ctx := context.Background()
			rng := testutil.NewRNG(benchSeed + 100)

			var reads, writes int64

			b.ReportAllocs()
			b.ResetTimer()

			b.RunParallel(func(pb *testing.PB) {
				localRng := testutil.NewRNG(benchSeed + time.Now().UnixNano())
				vec := make([]float32, dim)
				queryIdx := 0

				for pb.Next() {
					if localRng.Intn(100) < readPct {
						// Read operation
						q := queries[queryIdx%len(queries)]
						queryIdx++
						_, err := e.Search(ctx, q, k)
						if err != nil {
							b.Error(err)
							return
						}
						atomic.AddInt64(&reads, 1)
					} else {
						// Write operation
						rng.FillUniform(vec)
						_, err := e.Insert(ctx, vec, nil, nil)
						if err != nil {
							b.Error(err)
							return
						}
						atomic.AddInt64(&writes, 1)
					}
				}
			})

			b.StopTimer()
			totalOps := float64(reads + writes)
			b.ReportMetric(totalOps/b.Elapsed().Seconds(), "ops/sec")
			b.ReportMetric(float64(reads)/b.Elapsed().Seconds(), "reads/sec")
			b.ReportMetric(float64(writes)/b.Elapsed().Seconds(), "writes/sec")
		})
	}
}

// BenchmarkBurstWorkload simulates bursty traffic patterns where operations
// come in waves with quiet periods.
func BenchmarkBurstWorkload(b *testing.B) {
	const dim = dimMedium
	const initialSize = sizeSmall
	const k = 10
	const burstSize = 100
	const workers = 4

	e := OpenBenchEngine(b, dim)
	defer e.Close()

	e.LoadData(b, initialSize, dim)
	queries := MakeQueries(100, dim)
	e.WarmupSearch(b, queries, k)

	ctx := context.Background()

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		// Simulate a burst: 4 workers each doing 100 searches
		var wg sync.WaitGroup
		for w := range workers {
			wg.Add(1)
			go func(workerID int) {
				defer wg.Done()
				for j := 0; j < burstSize; j++ {
					q := queries[(workerID*burstSize+j)%len(queries)]
					_, _ = e.Search(ctx, q, k)
				}
			}(w)
		}
		wg.Wait()
	}

	b.StopTimer()
	totalQueries := float64(b.N * workers * burstSize)
	b.ReportMetric(totalQueries/b.Elapsed().Seconds(), "qps")
}

// BenchmarkReadAfterWrite measures the latency of reads that immediately follow writes.
// This tests the memtable search path which is critical for real-time applications.
func BenchmarkReadAfterWrite(b *testing.B) {
	const dim = dimMedium
	const k = 10

	e := OpenBenchEngine(b, dim)
	defer e.Close()

	// Pre-populate with some data
	e.LoadData(b, 1000, dim)

	rng := testutil.NewRNG(benchSeed + 200)
	ctx := context.Background()

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		// Write a new vector
		vec := make([]float32, dim)
		rng.FillUniform(vec)
		id, err := e.Insert(ctx, vec, nil, nil)
		if err != nil {
			b.Fatal(err)
		}

		// Immediately search for it - should find it in memtable
		results, err := e.Search(ctx, vec, k)
		if err != nil {
			b.Fatal(err)
		}

		// Verify the just-inserted vector is found
		found := false
		for _, r := range results {
			if r.ID == id {
				found = true
				break
			}
		}
		if !found {
			b.Fatalf("just-inserted vector not found in search results")
		}
	}

	b.StopTimer()
	b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "ops/sec")
}

// BenchmarkThroughputUnderLoad measures sustained throughput with background pressure.
func BenchmarkThroughputUnderLoad(b *testing.B) {
	const dim = dimMedium
	const k = 10

	e := OpenBenchEngine(b, dim,
		vecgo.WithFlushConfig(vecgo.FlushConfig{MaxMemTableSize: 1 << 20}), // 1MB memtable for more flushes
	)
	defer e.Close()

	e.LoadData(b, 5000, dim)
	queries := MakeQueries(100, dim)

	ctx := context.Background()
	rng := testutil.NewRNG(benchSeed + 300)

	// Background writer goroutine
	done := make(chan struct{})
	var writeCount int64
	go func() {
		vec := make([]float32, dim)
		for {
			select {
			case <-done:
				return
			default:
				rng.FillUniform(vec)
				_, _ = e.Insert(ctx, vec, nil, nil)
				atomic.AddInt64(&writeCount, 1)
			}
		}
	}()

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		q := queries[i%len(queries)]
		_, err := e.Search(ctx, q, k)
		if err != nil {
			b.Fatal(err)
		}
	}

	b.StopTimer()
	close(done)

	b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "qps")
	b.ReportMetric(float64(writeCount)/b.Elapsed().Seconds(), "bg_writes/sec")
}
