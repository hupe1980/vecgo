package vecgo_bench_test

import (
	"context"
	"testing"

	"github.com/hupe1980/vecgo"
	"github.com/hupe1980/vecgo/index"
	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/testutil"
)

// BenchmarkHNSWInsert benchmarks single vector insertion with HNSW
func BenchmarkHNSWInsert(b *testing.B) {
	dimensions := []int{128, 384, 768}

	for _, dim := range dimensions {
		b.Run(formatDim(dim), func(b *testing.B) {
			db, err := vecgo.HNSW[int](dim).SquaredL2().Build()
			if err != nil {
				b.Fatal(err)
			}
			defer db.Close()

			ctx := context.Background()
			b.ResetTimer()

			for i := 0; b.Loop(); i++ {
				vec := randomVector(dim)
				_, err := db.Insert(ctx, vecgo.VectorWithData[int]{
					Vector: vec,
					Data:   i,
				})
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkHNSWBatchInsert benchmarks batch insertion with HNSW
func BenchmarkHNSWBatchInsert(b *testing.B) {
	batchSizes := []int{10, 100, 1000}
	dim := 384

	for _, batchSize := range batchSizes {
		b.Run(formatCount(batchSize), func(b *testing.B) {
			db, err := vecgo.HNSW[int](dim).SquaredL2().Build()
			if err != nil {
				b.Fatal(err)
			}
			defer db.Close()

			ctx := context.Background()
			b.ResetTimer()

			for i := 0; b.Loop(); i++ {
				batch := make([]vecgo.VectorWithData[int], batchSize)
				for j := 0; j < batchSize; j++ {
					batch[j] = vecgo.VectorWithData[int]{
						Vector: randomVector(dim),
						Data:   i*batchSize + j,
					}
				}

				result := db.BatchInsert(ctx, batch)
				for _, err := range result.Errors {
					if err != nil {
						b.Fatal(err)
					}
				}
			}
		})
	}
}

// BenchmarkHNSWSearch benchmarks KNN search with HNSW and measures recall
func BenchmarkHNSWSearch(b *testing.B) {
	sizes := []int{1000, 10000}
	dim := 384
	k := 10

	for _, size := range sizes {
		b.Run(formatCount(size), func(b *testing.B) {
			// Use fresh RNG per sub-benchmark for reproducibility
			localRNG := testutil.NewRNG(42)

			db, vectors := setupHNSWIndexWithVectorsRNG(b, dim, size, localRNG)
			defer db.Close()

			query := localRNG.NormalizedVector(dim)
			ctx := context.Background()

			// Compute ground truth once
			groundTruth := groundTruthSearch(ctx, vectors, query, k)

			// Measure recall before benchmark
			approxResults, _ := db.Search(query).KNN(k).Execute(ctx)
			recall := computeRecall(groundTruth, approxResults)

			b.ResetTimer()

			for i := 0; b.Loop(); i++ {
				_, err := db.Search(query).KNN(k).Execute(ctx)
				if err != nil {
					b.Fatal(err)
				}
			}

			b.ReportMetric(float64(size), "vectors")
			b.ReportMetric(recall*100, "recall%")
		})
	}
}

// BenchmarkHNSWSearchEfTuning benchmarks search with different EF values
func BenchmarkHNSWSearchEfTuning(b *testing.B) {
	efValues := []int{16, 32, 64, 128, 256}
	dim := 384
	size := 10000

	for _, ef := range efValues {
		b.Run(formatCount(ef), func(b *testing.B) {
			db, err := vecgo.HNSW[int](dim).SquaredL2().EF(ef).Build()
			if err != nil {
				b.Fatal(err)
			}
			defer db.Close()

			ctx := context.Background()
			batch := make([]vecgo.VectorWithData[int], size)
			for i := 0; i < size; i++ {
				batch[i] = vecgo.VectorWithData[int]{
					Vector: randomVector(dim),
					Data:   i,
				}
			}
			db.BatchInsert(ctx, batch)

			query := randomVector(dim)
			b.ResetTimer()

			for i := 0; b.Loop(); i++ {
				_, err := db.Search(query).KNN(10).Execute(ctx)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkHNSWSearchMTuning benchmarks search with different M values
func BenchmarkHNSWSearchMTuning(b *testing.B) {
	mValues := []int{8, 16, 32, 48}
	dim := 384
	size := 10000

	for _, m := range mValues {
		b.Run(formatCount(m), func(b *testing.B) {
			db, err := vecgo.HNSW[int](dim).SquaredL2().M(m).Build()
			if err != nil {
				b.Fatal(err)
			}
			defer db.Close()

			ctx := context.Background()
			batch := make([]vecgo.VectorWithData[int], size)
			for i := 0; i < size; i++ {
				batch[i] = vecgo.VectorWithData[int]{
					Vector: randomVector(dim),
					Data:   i,
				}
			}
			db.BatchInsert(ctx, batch)

			query := randomVector(dim)
			b.ResetTimer()

			for i := 0; b.Loop(); i++ {
				_, err := db.Search(query).KNN(10).Execute(ctx)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkHNSWDistanceMetrics benchmarks HNSW with different distance functions
func BenchmarkHNSWDistanceMetrics(b *testing.B) {
	dim := 384
	size := 10000

	metrics := []struct {
		name     string
		distType index.DistanceType
	}{
		{"SquaredL2", index.DistanceTypeSquaredL2},
		{"Cosine", index.DistanceTypeCosine},
		{"DotProduct", index.DistanceTypeDotProduct},
	}

	for _, m := range metrics {
		b.Run(m.name, func(b *testing.B) {
			var builder *vecgo.HNSWBuilder[int]
			switch m.distType {
			case index.DistanceTypeSquaredL2:
				builder = vecgo.HNSW[int](dim).SquaredL2()
			case index.DistanceTypeCosine:
				builder = vecgo.HNSW[int](dim).Cosine()
			case index.DistanceTypeDotProduct:
				builder = vecgo.HNSW[int](dim).DotProduct()
			default:
				b.Fatalf("unsupported distance type: %v", m.distType)
			}
			db, err := builder.Build()
			if err != nil {
				b.Fatal(err)
			}
			defer db.Close()

			ctx := context.Background()
			batch := make([]vecgo.VectorWithData[int], size)
			for i := 0; i < size; i++ {
				batch[i] = vecgo.VectorWithData[int]{
					Vector: randomVector(dim),
					Data:   i,
				}
			}
			db.BatchInsert(ctx, batch)

			query := randomVector(dim)
			b.ResetTimer()

			for i := 0; b.Loop(); i++ {
				_, err := db.Search(query).KNN(10).Execute(ctx)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkHNSWConcurrentSearch benchmarks concurrent search
func BenchmarkHNSWConcurrentSearch(b *testing.B) {
	dim := 384
	size := 10000

	db := setupHNSWIndex(b, dim, size)
	defer db.Close()

	query := randomVector(dim)
	ctx := context.Background()

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			_, err := db.Search(query).KNN(10).Execute(ctx)
			if err != nil {
				b.Fatal(err)
			}
		}
	})
}

// BenchmarkHNSWHybridSearch benchmarks search with metadata filters
func BenchmarkHNSWHybridSearch(b *testing.B) {
	dim := 384
	size := 10000

	db, err := vecgo.HNSW[int](dim).SquaredL2().Build()
	if err != nil {
		b.Fatal(err)
	}
	defer db.Close()

	ctx := context.Background()
	for i := 0; i < size; i++ {
		_, err := db.Insert(ctx, vecgo.VectorWithData[int]{
			Vector: randomVector(dim),
			Data:   i,
			Metadata: metadata.Metadata{
				"category": metadata.String(categories[i%len(categories)]),
				"score":    metadata.Int(int64(i % 100)),
			},
		})
		if err != nil {
			b.Fatal(err)
		}
	}

	query := randomVector(dim)
	filter := &metadata.FilterSet{
		Filters: []metadata.Filter{
			{Key: "category", Operator: metadata.OpEqual, Value: metadata.String("technology")},
		},
	}
	b.ResetTimer()

	for i := 0; b.Loop(); i++ {
		_, err := db.Search(query).
			KNN(10).
			WithMetadata(filter).
			Execute(ctx)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkHNSWUpdate benchmarks vector updates with HNSW
func BenchmarkHNSWUpdate(b *testing.B) {
	dim := 384
	size := 10000

	db := setupHNSWIndex(b, dim, size)
	defer db.Close()

	ctx := context.Background()
	b.ResetTimer()

	for i := 0; b.Loop(); i++ {
		id := uint32(i % size)
		err := db.Update(ctx, id, vecgo.VectorWithData[int]{
			Vector: randomVector(dim),
			Data:   int(id),
		})
		if err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkHNSWDelete benchmarks vector deletion with HNSW
func BenchmarkHNSWDelete(b *testing.B) {
	dim := 384
	size := 10000

	db := setupHNSWIndex(b, dim, size)
	defer db.Close()

	ctx := context.Background()
	b.ResetTimer()

	for i := 0; b.Loop(); i++ {
		id := uint32(i % size)
		err := db.Delete(ctx, id)
		if err != nil && i < size {
			b.Fatal(err)
		}
	}
}

// BenchmarkHNSWStreamingSearch benchmarks streaming search
func BenchmarkHNSWStreamingSearch(b *testing.B) {
	dim := 384
	size := 10000

	db := setupHNSWIndex(b, dim, size)
	defer db.Close()

	query := randomVector(dim)
	ctx := context.Background()
	b.ResetTimer()

	for i := 0; b.Loop(); i++ {
		count := 0
		for _, err := range db.Search(query).KNN(10).Stream(ctx) {
			if err != nil {
				b.Fatal(err)
			}
			count++
			if count >= 10 {
				break
			}
		}
	}
}

// setupHNSWIndex creates a HNSW index with random data for benchmarking
func setupHNSWIndex(b *testing.B, dim, size int) *vecgo.Vecgo[int] {
	b.Helper()
	db, _ := setupHNSWIndexWithVectors(b, dim, size)
	return db
}

// setupHNSWIndexWithVectors creates a HNSW index and returns both db and vectors for recall computation
func setupHNSWIndexWithVectors(b *testing.B, dim, size int) (*vecgo.Vecgo[int], [][]float32) {
	b.Helper()
	return setupHNSWIndexWithVectorsRNG(b, dim, size, rng)
}

// setupHNSWIndexWithVectorsRNG creates a HNSW index with a specific RNG for reproducibility
func setupHNSWIndexWithVectorsRNG(b *testing.B, dim, size int, localRNG *testutil.RNG) (*vecgo.Vecgo[int], [][]float32) {
	b.Helper()

	db, err := vecgo.HNSW[int](dim).SquaredL2().Build()
	if err != nil {
		b.Fatal(err)
	}

	ctx := context.Background()
	vectors := make([][]float32, size)
	batch := make([]vecgo.VectorWithData[int], size)
	for i := 0; i < size; i++ {
		vectors[i] = localRNG.NormalizedVector(dim)
		batch[i] = vecgo.VectorWithData[int]{
			Vector: vectors[i],
			Data:   i,
		}
	}

	result := db.BatchInsert(ctx, batch)
	for _, err := range result.Errors {
		if err != nil {
			b.Fatal(err)
		}
	}

	return db, vectors
}
