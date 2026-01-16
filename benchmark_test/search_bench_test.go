package benchmark_test

import (
	"context"
	"strconv"
	"testing"

	"github.com/hupe1980/vecgo"
	"github.com/hupe1980/vecgo/testutil"
)

// ============================================================================
// Search Benchmarks
// ============================================================================

// BenchmarkSearchDim measures search latency and recall across dimensions.
func BenchmarkSearchDim(b *testing.B) {
	dims := []int{dimSmall, dimMedium, dimLarge}
	const n = sizeSmall // 10k vectors - fast CI
	const k = 10

	for _, dim := range dims {
		b.Run("dim="+strconv.Itoa(dim), func(b *testing.B) {
			e := OpenBenchEngine(b, dim)
			defer e.Close()

			data, pks := e.LoadData(b, n, dim)
			queries := MakeQueries(100, dim)
			e.WarmupSearch(b, queries, k)

			ctx := context.Background()
			var totalRecall float64

			b.ReportAllocs()
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				q := queries[i%len(queries)]
				results, err := e.Search(ctx, q, k)
				if err != nil {
					b.Fatal(err)
				}

				// Compute recall for a subset to avoid slowing the benchmark
				if i < 50 {
					truth := exactTopK_L2_WithIDs(data, pks, q, k)
					totalRecall += recallAtK(results, truth)
				}
			}

			b.StopTimer()
			avgRecall := totalRecall / float64(min(50, b.N))
			b.ReportMetric(avgRecall, "recall@10")
			b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "qps")
		})
	}
}

// BenchmarkSearchScaling measures search latency scaling with dataset size.
func BenchmarkSearchScaling(b *testing.B) {
	sizes := []int{1_000, 10_000, 50_000}
	const dim = dimMedium
	const k = 10

	for _, n := range sizes {
		b.Run("n="+strconv.Itoa(n), func(b *testing.B) {
			e := OpenBenchEngine(b, dim)
			defer e.Close()

			data, pks := e.LoadData(b, n, dim)
			queries := MakeQueries(100, dim)
			e.WarmupSearch(b, queries, k)

			ctx := context.Background()
			var totalRecall float64

			b.ReportAllocs()
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				q := queries[i%len(queries)]
				results, err := e.Search(ctx, q, k)
				if err != nil {
					b.Fatal(err)
				}
				if i < 20 {
					truth := exactTopK_L2_WithIDs(data, pks, q, k)
					totalRecall += recallAtK(results, truth)
				}
			}

			b.StopTimer()
			avgRecall := totalRecall / float64(min(20, b.N))
			b.ReportMetric(avgRecall, "recall@10")
			b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "qps")
		})
	}
}

// BenchmarkConcurrentSearch measures search throughput under concurrent load.
func BenchmarkConcurrentSearch(b *testing.B) {
	parallelism := []int{1, 2, 4, 8}
	const dim = dimMedium
	const n = sizeMedium
	const k = 10

	for _, p := range parallelism {
		b.Run("goroutines="+strconv.Itoa(p), func(b *testing.B) {
			e := OpenBenchEngine(b, dim)
			defer e.Close()

			e.LoadData(b, n, dim)
			queries := MakeQueries(100, dim)
			e.WarmupSearch(b, queries, k)

			ctx := context.Background()

			b.SetParallelism(p)
			b.ReportAllocs()
			b.ResetTimer()

			b.RunParallel(func(pb *testing.PB) {
				i := 0
				for pb.Next() {
					q := queries[i%len(queries)]
					i++
					_, err := e.Search(ctx, q, k)
					if err != nil {
						b.Error(err)
						return
					}
				}
			})

			b.StopTimer()
			b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "qps")
		})
	}
}

// BenchmarkBatchSearch measures batched search throughput.
func BenchmarkBatchSearch(b *testing.B) {
	batchSizes := []int{1, 10, 50}
	const dim = dimMedium
	const n = sizeMedium
	const k = 10

	for _, bs := range batchSizes {
		b.Run("batch="+strconv.Itoa(bs), func(b *testing.B) {
			e := OpenBenchEngine(b, dim)
			defer e.Close()

			e.LoadData(b, n, dim)

			rng := testutil.NewRNG(benchSeed + 2)
			batch := make([][]float32, bs)
			for i := range batch {
				v := make([]float32, dim)
				rng.FillUniform(v)
				batch[i] = v
			}

			ctx := context.Background()
			e.WarmupSearch(b, batch, k)

			b.ReportAllocs()
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				_, err := e.BatchSearch(ctx, batch, k)
				if err != nil {
					b.Fatal(err)
				}
			}

			b.StopTimer()
			totalQueries := float64(b.N * bs)
			b.ReportMetric(totalQueries/b.Elapsed().Seconds(), "qps")
			b.ReportMetric(float64(bs), "batch_size")
		})
	}
}

// BenchmarkSearchRefineFactor measures how search quality/latency trades off with refine factor.
func BenchmarkSearchRefineFactor(b *testing.B) {
	refineFactors := []float32{1.0, 1.5, 2.0, 3.0, 5.0}
	const dim = dimMedium
	const n = sizeMedium
	const k = 10

	for _, rf := range refineFactors {
		b.Run("refine="+strconv.FormatFloat(float64(rf), 'f', 1, 32), func(b *testing.B) {
			e := OpenBenchEngine(b, dim)
			defer e.Close()

			data, pks := e.LoadData(b, n, dim)
			queries := MakeQueries(100, dim)
			e.WarmupSearch(b, queries, k)

			ctx := context.Background()
			var totalRecall float64

			b.ReportAllocs()
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				q := queries[i%len(queries)]
				results, err := e.Search(ctx, q, k, vecgo.WithRefineFactor(rf))
				if err != nil {
					b.Fatal(err)
				}
				if i < 20 {
					truth := exactTopK_L2_WithIDs(data, pks, q, k)
					totalRecall += recallAtK(results, truth)
				}
			}

			b.StopTimer()
			avgRecall := totalRecall / float64(min(20, b.N))
			b.ReportMetric(avgRecall, "recall@10")
			b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "qps")
		})
	}
}
