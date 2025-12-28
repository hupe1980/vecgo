package vecgo_bench_test

import (
	"context"
	"testing"

	"github.com/hupe1980/vecgo"
	"github.com/hupe1980/vecgo/index"
	"github.com/hupe1980/vecgo/metadata"
)

// BenchmarkFlatInsert benchmarks single vector insertion
func BenchmarkFlatInsert(b *testing.B) {
	dimensions := []int{128, 384, 768, 1536}

	for _, dim := range dimensions {
		b.Run(formatDim(dim), func(b *testing.B) {
			db, err := vecgo.Flat[int](dim).SquaredL2().Build()
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

// BenchmarkFlatBatchInsert benchmarks batch insertion
func BenchmarkFlatBatchInsert(b *testing.B) {
	batchSizes := []int{10, 100, 1000}
	dim := 384

	for _, batchSize := range batchSizes {
		b.Run(formatCount(batchSize), func(b *testing.B) {
			db, err := vecgo.Flat[int](dim).SquaredL2().Build()
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

// BenchmarkFlatSearch benchmarks KNN search
func BenchmarkFlatSearch(b *testing.B) {
	sizes := []int{1000, 10000}
	dim := 384

	for _, size := range sizes {
		b.Run(formatCount(size), func(b *testing.B) {
			db := setupFlatIndex(b, dim, size)
			defer db.Close()

			query := randomVector(dim)
			ctx := context.Background()
			b.ResetTimer()

			for i := 0; b.Loop(); i++ {
				_, err := db.Search(query).KNN(10).Execute(ctx)
				if err != nil {
					b.Fatal(err)
				}
			}

			b.ReportMetric(float64(size), "vectors")
		})
	}
}

// BenchmarkFlatSearchK benchmarks search with different K values
func BenchmarkFlatSearchK(b *testing.B) {
	kValues := []int{1, 10, 50}
	dim := 384
	size := 10000

	db := setupFlatIndex(b, dim, size)
	defer db.Close()

	for _, k := range kValues {
		b.Run(formatCount(k), func(b *testing.B) {
			query := randomVector(dim)
			ctx := context.Background()
			b.ResetTimer()

			for i := 0; b.Loop(); i++ {
				_, err := db.Search(query).KNN(k).Execute(ctx)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkFlatDistanceMetrics benchmarks different distance functions
func BenchmarkFlatDistanceMetrics(b *testing.B) {
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
			var builder *vecgo.FlatBuilder[int]
			switch m.distType {
			case index.DistanceTypeSquaredL2:
				builder = vecgo.Flat[int](dim).SquaredL2()
			case index.DistanceTypeCosine:
				builder = vecgo.Flat[int](dim).Cosine()
			case index.DistanceTypeDotProduct:
				builder = vecgo.Flat[int](dim).DotProduct()
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

// BenchmarkFlatUpdate benchmarks vector updates
func BenchmarkFlatUpdate(b *testing.B) {
	dim := 384
	size := 10000

	db := setupFlatIndex(b, dim, size)
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

// BenchmarkFlatDelete benchmarks vector deletion
func BenchmarkFlatDelete(b *testing.B) {
	dim := 384
	size := 10000

	db := setupFlatIndex(b, dim, size)
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

// BenchmarkFlatHybridSearch benchmarks search with metadata filters
func BenchmarkFlatHybridSearch(b *testing.B) {
	dim := 384
	size := 10000

	db, err := vecgo.Flat[int](dim).SquaredL2().Build()
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

// BenchmarkFlatStreamingSearch benchmarks streaming search
func BenchmarkFlatStreamingSearch(b *testing.B) {
	dim := 384
	size := 10000

	db := setupFlatIndex(b, dim, size)
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

// setupFlatIndex creates a Flat index with random data for benchmarking
func setupFlatIndex(b *testing.B, dim, size int) *vecgo.Vecgo[int] {
	b.Helper()

	db, err := vecgo.Flat[int](dim).SquaredL2().Build()
	if err != nil {
		b.Fatal(err)
	}

	ctx := context.Background()
	batch := make([]vecgo.VectorWithData[int], size)
	for i := 0; i < size; i++ {
		batch[i] = vecgo.VectorWithData[int]{
			Vector: randomVector(dim),
			Data:   i,
		}
	}

	result := db.BatchInsert(ctx, batch)
	for _, err := range result.Errors {
		if err != nil {
			b.Fatal(err)
		}
	}

	return db
}
