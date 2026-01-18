package benchmark_test

import (
	"context"
	"fmt"
	"testing"

	"github.com/hupe1980/vecgo"
	"github.com/hupe1980/vecgo/metadata"
)

// ============================================================================
// ADVERSARIAL DISTRIBUTION BENCHMARKS
// ============================================================================
//
// These benchmarks test performance under pathological/real-world distributions
// that expose planner weaknesses and allocation cliffs.
//
// 5-Distribution Framework (Best-in-Class Methodology):
// 1. Uniform      - Baseline, algorithmic efficiency
// 2. Zipfian      - Hot keys, cache behavior
// 3. Segment-local - Tests global vs local selectivity (Snowflake killer)
// 4. Correlated   - Metadata correlates with vector space
// 5. Boolean adversarial - Bitmap stress, compound filters

// BenchmarkAdversarialSearch runs comprehensive tests across all 5 distributions.
func BenchmarkAdversarialSearch(b *testing.B) {
	ctx := context.Background()

	distributions := []string{
		"uniform_128d_50k",
		"zipfian_128d_50k",
		"seglocal_128d_50k",
		"correlated_128d_50k",
		"booladv_128d_50k",
	}

	selectivities := []float64{0.01, 0.10, 0.50}
	const k = 10

	for _, dist := range distributions {
		b.Run(dist, func(b *testing.B) {
			if !FixtureExists(dist) {
				b.Skipf("fixture %q not found", dist)
			}

			db, err := OpenFixture(ctx, dist)
			if err != nil {
				b.Fatalf("open fixture: %v", err)
			}
			defer db.Close()

			data, err := LoadFixtureData(dist)
			if err != nil {
				b.Fatalf("load fixture data: %v", err)
			}

			queries := data.Queries

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

					// Compute recall
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

// BenchmarkSegmentLocalSkew specifically tests the "Snowflake killer" scenario:
// Global stats say 1% selectivity, but per-segment selectivity is 90%.
func BenchmarkSegmentLocalSkew(b *testing.B) {
	ctx := context.Background()

	fixtureName := "seglocal_128d_50k"
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

	// Test the critical 1% selectivity point
	// This is where global stats lie most dramatically
	b.Run("global_1pct_local_90pct", func(b *testing.B) {
		filter := CreateSelectivityFilter(0.01, data.Config.BucketCount)
		queries := data.Queries
		const k = 10

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
		b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "qps")
	})
}

// BenchmarkCorrelatedVectors tests when metadata predicts vector similarity.
// This is the realistic case: "category=shoes" → vectors cluster.
func BenchmarkCorrelatedVectors(b *testing.B) {
	ctx := context.Background()

	fixtureName := "correlated_128d_50k"
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

	selectivities := []float64{0.01, 0.10, 0.50}
	const k = 10

	for _, sel := range selectivities {
		b.Run(fmt.Sprintf("sel=%.0f%%", sel*100), func(b *testing.B) {
			filter := CreateSelectivityFilter(sel, data.Config.BucketCount)
			queries := data.Queries

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
			b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "qps")
		})
	}
}

// BenchmarkBooleanAdversarial tests compound filters with bimodal distribution.
// Tests: (bucket < X OR bucket_b = Y) AND bucket > Z
func BenchmarkBooleanAdversarial(b *testing.B) {
	ctx := context.Background()

	fixtureName := "booladv_128d_50k"
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

	// Simple filter (baseline)
	b.Run("simple_filter", func(b *testing.B) {
		filter := CreateSelectivityFilter(0.01, data.Config.BucketCount)

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
		b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "qps")
	})

	// Range filter (tests NumericIndex)
	b.Run("range_filter", func(b *testing.B) {
		// bucket > 10 AND bucket < 50 (roughly 40% selectivity)
		filter := &metadata.FilterSet{
			Filters: []metadata.Filter{
				{Key: "bucket", Operator: metadata.OpGreaterThan, Value: metadata.Int(10)},
				{Key: "bucket", Operator: metadata.OpLessThan, Value: metadata.Int(50)},
			},
		}

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
		b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "qps")
	})

	// Multi-field filter (tests bitmap AND operations)
	b.Run("multi_field_and", func(b *testing.B) {
		// bucket < 10 AND bucket_b < 20
		filter := &metadata.FilterSet{
			Filters: []metadata.Filter{
				{Key: "bucket", Operator: metadata.OpLessThan, Value: metadata.Int(10)},
				{Key: "bucket_b", Operator: metadata.OpLessThan, Value: metadata.Int(20)},
			},
		}

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
		b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "qps")
	})
}

// BenchmarkDistributionComparison provides a side-by-side comparison table.
func BenchmarkDistributionComparison(b *testing.B) {
	ctx := context.Background()

	// All 5 distributions at 1% selectivity
	distributions := []struct {
		name    string
		fixture string
	}{
		{"uniform", "uniform_128d_50k"},
		{"zipfian", "zipfian_128d_50k"},
		{"seglocal", "seglocal_128d_50k"},
		{"correlated", "correlated_128d_50k"},
		{"booladv", "booladv_128d_50k"},
	}

	const sel = 0.01
	const k = 10

	for _, dist := range distributions {
		b.Run(dist.name, func(b *testing.B) {
			if !FixtureExists(dist.fixture) {
				b.Skipf("fixture %q not found", dist.fixture)
			}

			db, err := OpenFixture(ctx, dist.fixture)
			if err != nil {
				b.Fatalf("open fixture: %v", err)
			}
			defer db.Close()

			data, err := LoadFixtureData(dist.fixture)
			if err != nil {
				b.Fatalf("load fixture data: %v", err)
			}

			filter := CreateSelectivityFilter(sel, data.Config.BucketCount)
			queries := data.Queries
			truth := data.GroundTruth[SelectivityKey(sel)]

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

			// Compute recall
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
}
