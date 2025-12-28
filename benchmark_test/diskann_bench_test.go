package vecgo_bench_test

import (
	"context"
	"testing"

	"github.com/hupe1980/vecgo"
	"github.com/hupe1980/vecgo/index"
	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/testutil"
)

// BenchmarkDiskANNInsert benchmarks single vector insertion with DiskANN
func BenchmarkDiskANNInsert(b *testing.B) {
	dimensions := []int{128, 384, 768}

	for _, dim := range dimensions {
		b.Run(formatDim(dim), func(b *testing.B) {
			tmpDir := b.TempDir()
			db, err := vecgo.DiskANN[int](tmpDir, dim).SquaredL2().Build()
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

// BenchmarkDiskANNBatchInsert benchmarks batch insertion with DiskANN
func BenchmarkDiskANNBatchInsert(b *testing.B) {
	batchSizes := []int{10, 100, 1000}
	dim := 384

	for _, batchSize := range batchSizes {
		b.Run(formatCount(batchSize), func(b *testing.B) {
			tmpDir := b.TempDir()
			db, err := vecgo.DiskANN[int](tmpDir, dim).SquaredL2().Build()
			if err != nil {
				b.Fatal(err)
			}
			defer db.Close()

			ctx := context.Background()
			b.ResetTimer()

			for i := 0; b.Loop(); i++ {
				batch := make([]vecgo.VectorWithData[int], batchSize)
				for j := range batchSize {
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

// BenchmarkDiskANNSearch benchmarks KNN search with DiskANN and measures recall
func BenchmarkDiskANNSearch(b *testing.B) {
	sizes := []int{1000, 10000}
	dim := 384
	k := 10

	for _, size := range sizes {
		b.Run(formatCount(size), func(b *testing.B) {
			// Use fresh RNG per sub-benchmark for reproducibility
			localRNG := testutil.NewRNG(42)

			db, vectors := setupDiskANNIndexWithVectorsRNG(b, dim, size, localRNG)
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

// BenchmarkDiskANNBeamWidth benchmarks search with different beam widths
func BenchmarkDiskANNBeamWidth(b *testing.B) {
	beamWidths := []int{2, 4, 8, 16}
	dim := 384
	size := 5000

	for _, bw := range beamWidths {
		b.Run(formatCount(bw), func(b *testing.B) {
			tmpDir := b.TempDir()
			db, err := vecgo.DiskANN[int](tmpDir, dim).SquaredL2().BeamWidth(bw).Build()
			if err != nil {
				b.Fatal(err)
			}
			defer db.Close()

			ctx := context.Background()
			batch := make([]vecgo.VectorWithData[int], size)
			for i := range size {
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

// BenchmarkDiskANNRTuning benchmarks search with different R values (max degree)
func BenchmarkDiskANNRTuning(b *testing.B) {
	rValues := []int{32, 64, 100}
	dim := 384
	size := 5000

	for _, r := range rValues {
		b.Run(formatCount(r), func(b *testing.B) {
			tmpDir := b.TempDir()
			db, err := vecgo.DiskANN[int](tmpDir, dim).SquaredL2().R(r).Build()
			if err != nil {
				b.Fatal(err)
			}
			defer db.Close()

			ctx := context.Background()
			batch := make([]vecgo.VectorWithData[int], size)
			for i := range size {
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

// BenchmarkDiskANNDistanceMetrics benchmarks DiskANN with different distance functions
func BenchmarkDiskANNDistanceMetrics(b *testing.B) {
	dim := 384
	size := 5000

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
			tmpDir := b.TempDir()
			var db *vecgo.Vecgo[int]
			var err error

			switch m.distType {
			case index.DistanceTypeSquaredL2:
				db, err = vecgo.DiskANN[int](tmpDir, dim).SquaredL2().Build()
			case index.DistanceTypeCosine:
				db, err = vecgo.DiskANN[int](tmpDir, dim).Cosine().Build()
			case index.DistanceTypeDotProduct:
				db, err = vecgo.DiskANN[int](tmpDir, dim).DotProduct().Build()
			}
			if err != nil {
				b.Fatal(err)
			}
			defer db.Close()

			ctx := context.Background()
			batch := make([]vecgo.VectorWithData[int], size)
			for i := range size {
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

// BenchmarkDiskANNUpdate benchmarks vector updates with DiskANN
func BenchmarkDiskANNUpdate(b *testing.B) {
	dim := 384
	size := 5000

	db := setupDiskANNIndex(b, dim, size)
	defer db.Close()

	ctx := context.Background()

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

// BenchmarkDiskANNHybridSearch benchmarks search with metadata filters
func BenchmarkDiskANNHybridSearch(b *testing.B) {
	dim := 384
	size := 5000
	tmpDir := b.TempDir()

	db, err := vecgo.DiskANN[int](tmpDir, dim).SquaredL2().Build()
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

// setupDiskANNIndex creates a DiskANN index with random data for benchmarking
func setupDiskANNIndex(b *testing.B, dim, size int) *vecgo.Vecgo[int] {
	b.Helper()
	db, _ := setupDiskANNIndexWithVectors(b, dim, size)
	return db
}

// setupDiskANNIndexWithVectors creates a DiskANN index and returns both db and vectors for recall computation
func setupDiskANNIndexWithVectors(b *testing.B, dim, size int) (*vecgo.Vecgo[int], [][]float32) {
	b.Helper()
	return setupDiskANNIndexWithVectorsRNG(b, dim, size, rng)
}

// setupDiskANNIndexWithVectorsRNG creates a DiskANN index with a specific RNG for reproducibility
func setupDiskANNIndexWithVectorsRNG(b *testing.B, dim, size int, localRNG *testutil.RNG) (*vecgo.Vecgo[int], [][]float32) {
	b.Helper()

	tmpDir := b.TempDir()
	db, err := vecgo.DiskANN[int](tmpDir, dim).SquaredL2().Build()
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
