package benchmark_test

import (
	"context"
	"fmt"
	"testing"

	"github.com/hupe1980/vecgo"
)

// ============================================================================
// FIXTURE-BASED BENCHMARKS — Fast, Reproducible, Production-Grade
// ============================================================================
//
// These benchmarks use pre-built fixtures for instant startup.
// Generate fixtures once: go test -tags=genfixtures -run=TestGenerateFixtures -v ./benchmark_test/...
// Run benchmarks:        go test -bench=BenchmarkFast -run=^$ ./benchmark_test/...
//
// Benefits:
// - 10-100x faster benchmark startup (Open vs Insert)
// - Reproducible results (deterministic fixtures)
// - Pre-computed ground truth (no recall computation overhead)
// - Realistic data distributions (Zipfian, clustered vectors)

// BenchmarkFastSearch benchmarks search performance using pre-built fixtures.
// This is the primary benchmark for search optimization work.
//
// The benchmark suite covers 5 distributions (best-in-class methodology):
// - Uniform: Baseline, algorithmic efficiency
// - Zipfian: Hot keys, cache behavior, planner lies
// - Segment-local skew: Planner correctness (Snowflake killer)
// - Correlated: ANN + filter interaction
// - Boolean adversarial: Bitmap operations, allocation stress
func BenchmarkFastSearch(b *testing.B) {
	ctx := context.Background()

	// Use quick fixtures for CI, full adversarial set for benchmarking
	fixtures := []string{"uniform_128d_50k", "zipfian_128d_50k", "seglocal_128d_50k", "correlated_128d_50k", "booladv_128d_50k"}
	if testing.Short() {
		fixtures = []string{"uniform_128d_10k", "zipfian_128d_10k"}
	}

	for _, fixtureName := range fixtures {
		b.Run(fixtureName, func(b *testing.B) {
			// Check if fixture exists
			if !FixtureExists(fixtureName) {
				b.Skipf("fixture %q not found; run 'go test -tags=genfixtures -run=TestGenerateFixtures' to generate", fixtureName)
			}

			// Open pre-built database (instant, no insert)
			db, err := OpenFixture(ctx, fixtureName)
			if err != nil {
				b.Fatalf("open fixture: %v", err)
			}
			defer db.Close()

			// Load pre-computed queries and ground truth
			data, err := LoadFixtureData(fixtureName)
			if err != nil {
				b.Fatalf("load fixture data: %v", err)
			}

			queries := data.Queries
			const k = 10

			// No-filter baseline
			b.Run("nofilter", func(b *testing.B) {
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
				b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "qps")
			})

			// Skip filtered benchmarks for nofilter fixtures
			if data.Config.Distribution == "nofilter" {
				return
			}

			// Filtered search at various selectivities
			selectivities := []float64{0.01, 0.05, 0.10, 0.50, 0.90}
			for _, sel := range selectivities {
				b.Run(fmt.Sprintf("sel=%.0f%%", sel*100), func(b *testing.B) {
					filter := CreateSelectivityFilter(sel, data.Config.BucketCount)
					truthKey := SelectivityKey(sel)
					truth := data.GroundTruth[truthKey]

					b.ReportAllocs()
					b.ResetTimer()

					for i := 0; i < b.N; i++ {
						q := queries[i%len(queries)]
						_, err := db.Search(ctx, q, k, vecgo.WithFilter(filter), vecgo.WithoutData())
						if err != nil {
							b.Fatal(err)
						}
					}

					b.StopTimer()

					// Compute recall using pre-computed ground truth
					if len(truth) > 0 {
						var sumRecall float64
						numSamples := min(20, len(queries))
						for qi := 0; qi < numSamples; qi++ {
							results, _ := db.Search(ctx, queries[qi], k, vecgo.WithFilter(filter), vecgo.WithoutData())
							sumRecall += recallAtKWithIDs(results, truth[qi])
						}
						b.ReportMetric(sumRecall/float64(numSamples), "recall@10")
					}
					b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "qps")
				})
			}
		})
	}
}

// BenchmarkFastSearchWithData benchmarks search WITH metadata/vectors returned.
// This exercises the FetchInto/Clone path which is the allocation hotspot.
func BenchmarkFastSearchWithData(b *testing.B) {
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

	// Test different data inclusion patterns
	// Default includes metadata+payload, WithoutData excludes all, WithVector adds vector
	b.Run("id_only", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()

		for i := 0; i < b.N; i++ {
			q := queries[i%len(queries)]
			results, err := db.Search(ctx, q, k, vecgo.WithoutData())
			if err != nil {
				b.Fatal(err)
			}
			if len(results) == 0 {
				b.Fatal("no results")
			}
		}

		b.StopTimer()
		b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "qps")
	})

	b.Run("with_metadata_payload", func(b *testing.B) {
		// Default behavior - includes metadata and payload
		b.ReportAllocs()
		b.ResetTimer()

		for i := 0; i < b.N; i++ {
			q := queries[i%len(queries)]
			results, err := db.Search(ctx, q, k)
			if err != nil {
				b.Fatal(err)
			}
			if len(results) == 0 {
				b.Fatal("no results")
			}
		}

		b.StopTimer()
		b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "qps")
	})

	b.Run("with_vector", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()

		for i := 0; i < b.N; i++ {
			q := queries[i%len(queries)]
			results, err := db.Search(ctx, q, k, vecgo.WithVector())
			if err != nil {
				b.Fatal(err)
			}
			if len(results) == 0 {
				b.Fatal("no results")
			}
		}

		b.StopTimer()
		b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "qps")
	})
}

// BenchmarkFastBatchSearch benchmarks batch search performance.
func BenchmarkFastBatchSearch(b *testing.B) {
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

	batchSizes := []int{1, 10, 50, 100}
	const k = 10

	for _, bs := range batchSizes {
		b.Run(fmt.Sprintf("batch=%d", bs), func(b *testing.B) {
			// Prepare batch from queries
			batch := make([][]float32, bs)
			for i := range batch {
				batch[i] = data.Queries[i%len(data.Queries)]
			}

			b.ReportAllocs()
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				_, err := db.BatchSearch(ctx, batch, k)
				if err != nil {
					b.Fatal(err)
				}
			}

			b.StopTimer()
			totalQueries := float64(b.N * bs)
			b.ReportMetric(totalQueries/b.Elapsed().Seconds(), "qps")
		})
	}
}

// BenchmarkFastConcurrentSearch benchmarks concurrent search throughput.
func BenchmarkFastConcurrentSearch(b *testing.B) {
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

	parallelism := []int{1, 2, 4, 8}
	const k = 10

	for _, p := range parallelism {
		b.Run(fmt.Sprintf("goroutines=%d", p), func(b *testing.B) {
			b.SetParallelism(p)
			b.ReportAllocs()
			b.ResetTimer()

			b.RunParallel(func(pb *testing.PB) {
				i := 0
				for pb.Next() {
					q := data.Queries[i%len(data.Queries)]
					i++
					_, err := db.Search(ctx, q, k, vecgo.WithoutData())
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

// BenchmarkFastFiltered benchmarks filtered search at all selectivity levels.
// This is the primary benchmark for filter optimization work.
func BenchmarkFastFiltered(b *testing.B) {
	ctx := context.Background()

	fixtures := []string{"uniform_128d_50k", "zipfian_128d_50k"}
	if testing.Short() {
		fixtures = []string{"uniform_128d_10k", "zipfian_128d_10k"}
	}

	selectivities := []float64{0.01, 0.05, 0.10, 0.30, 0.50, 0.90}
	const k = 10

	for _, fixtureName := range fixtures {
		b.Run(fixtureName, func(b *testing.B) {
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

			for _, sel := range selectivities {
				b.Run(fmt.Sprintf("sel=%.0f%%", sel*100), func(b *testing.B) {
					filter := CreateSelectivityFilter(sel, data.Config.BucketCount)

					b.ReportAllocs()
					b.ResetTimer()

					for i := 0; i < b.N; i++ {
						q := data.Queries[i%len(data.Queries)]
						_, err := db.Search(ctx, q, k, vecgo.WithFilter(filter), vecgo.WithoutData())
						if err != nil {
							b.Fatal(err)
						}
					}

					b.StopTimer()
					b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "qps")
				})
			}
		})
	}
}

// BenchmarkFastDimensions benchmarks search across different embedding dimensions.
func BenchmarkFastDimensions(b *testing.B) {
	ctx := context.Background()

	dimFixtures := map[int]string{
		128: "uniform_128d_50k",
		768: "uniform_768d_50k",
	}
	if testing.Short() {
		dimFixtures = map[int]string{
			128: "uniform_128d_10k",
		}
	}

	const k = 10

	for dim, fixtureName := range dimFixtures {
		b.Run(fmt.Sprintf("dim=%d", dim), func(b *testing.B) {
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

			b.ReportAllocs()
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				q := data.Queries[i%len(data.Queries)]
				_, err := db.Search(ctx, q, k, vecgo.WithoutData())
				if err != nil {
					b.Fatal(err)
				}
			}

			b.StopTimer()
			b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "qps")
		})
	}
}

// ============================================================================
// Helper Functions
// ============================================================================

// Note: recallAtKWithIDs is defined in bench_methodology_test.go
