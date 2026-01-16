package benchmark_test

import (
	"context"
	"strconv"
	"testing"

	"github.com/hupe1980/vecgo/testutil"
)

// ============================================================================
// Insert Benchmarks
// ============================================================================

// BenchmarkInsert measures single-insert throughput.
// Reports: ns/op, allocs, and vectors/sec.
func BenchmarkInsert(b *testing.B) {
	dims := []int{dimSmall, dimMedium, dimLarge}

	for _, dim := range dims {
		b.Run("dim="+strconv.Itoa(dim), func(b *testing.B) {
			e := OpenBenchEngine(b, dim)
			defer e.Close()

			rng := testutil.NewRNG(benchSeed)
			vecs := make([][]float32, b.N)
			for i := range vecs {
				v := make([]float32, dim)
				rng.FillUniform(v)
				vecs[i] = v
			}

			ctx := context.Background()
			b.ReportAllocs()
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				_, err := e.Insert(ctx, vecs[i], nil, nil)
				if err != nil {
					b.Fatal(err)
				}
			}

			b.StopTimer()
			b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "vectors/sec")
		})
	}
}

// BenchmarkBatchInsert measures batch-insert throughput with various batch sizes.
func BenchmarkBatchInsert(b *testing.B) {
	batchSizes := []int{10, 100, 1000}

	for _, bs := range batchSizes {
		b.Run("batch="+strconv.Itoa(bs), func(b *testing.B) {
			e := OpenBenchEngine(b, dimMedium)
			defer e.Close()

			rng := testutil.NewRNG(benchSeed)
			batch := make([][]float32, bs)
			for i := range batch {
				v := make([]float32, dimMedium)
				rng.FillUniform(v)
				batch[i] = v
			}

			ctx := context.Background()
			b.ReportAllocs()
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				_, err := e.BatchInsert(ctx, batch, nil, nil)
				if err != nil {
					b.Fatal(err)
				}
			}

			b.StopTimer()
			totalVecs := float64(b.N * bs)
			b.ReportMetric(totalVecs/b.Elapsed().Seconds(), "vectors/sec")
			b.ReportMetric(float64(bs), "batch_size")
		})
	}
}

// BenchmarkDeferredInsert measures insert throughput with deferred indexing.
// This simulates bulk loading scenarios - measures time to load N vectors once.
// Use -benchtime=1x since each iteration loads a full dataset.
func BenchmarkDeferredInsert(b *testing.B) {
	sizes := []int{1000, 10_000}

	for _, n := range sizes {
		b.Run("n="+strconv.Itoa(n), func(b *testing.B) {
			rng := testutil.NewRNG(benchSeed)
			batch := make([][]float32, n)
			for i := range batch {
				v := make([]float32, dimMedium)
				rng.FillUniform(v)
				batch[i] = v
			}

			b.ReportAllocs()
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				b.StopTimer()
				// Create fresh engine per iteration to avoid memory accumulation
				e := OpenBenchEngine(b, dimMedium)
				b.StartTimer()

				ctx := context.Background()
				_, err := e.BatchInsertDeferred(ctx, batch, nil, nil)
				if err != nil {
					b.Fatal(err)
				}

				b.StopTimer()
				e.Close()
				b.StartTimer()
			}

			b.StopTimer()
			totalVecs := float64(b.N * n)
			b.ReportMetric(totalVecs/b.Elapsed().Seconds(), "vectors/sec")
		})
	}
}

// BenchmarkConcurrentInsert measures insert throughput under concurrent load.
func BenchmarkConcurrentInsert(b *testing.B) {
	parallelism := []int{1, 2, 4, 8}

	for _, p := range parallelism {
		b.Run("goroutines="+strconv.Itoa(p), func(b *testing.B) {
			e := OpenBenchEngine(b, dimMedium)
			defer e.Close()

			rng := testutil.NewRNG(benchSeed)

			b.SetParallelism(p)
			b.ReportAllocs()
			b.ResetTimer()

			b.RunParallel(func(pb *testing.PB) {
				ctx := context.Background()
				vec := make([]float32, dimMedium)
				for pb.Next() {
					rng.FillUniform(vec) // Note: not thread-safe but acceptable for benchmarks
					_, err := e.Insert(ctx, vec, nil, nil)
					if err != nil {
						b.Error(err)
						return
					}
				}
			})

			b.StopTimer()
			b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "vectors/sec")
		})
	}
}
