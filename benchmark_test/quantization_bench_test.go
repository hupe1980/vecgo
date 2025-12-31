package vecgo_bench_test

import (
	"context"
	"testing"

	"github.com/hupe1980/vecgo"
	"github.com/hupe1980/vecgo/index"
	"github.com/hupe1980/vecgo/quantization"
)

// BenchmarkBinaryQuantization benchmarks binary quantization
func BenchmarkBinaryQuantization(b *testing.B) {
	dimensions := []int{128, 384, 768, 1536}

	for _, dim := range dimensions {
		b.Run(formatDim(dim), func(b *testing.B) {
			bq := quantization.NewBinaryQuantizer(dim)

			// Benchmark encoding
			b.Run("Encode", func(b *testing.B) {
				vec := randomVector(dim)
				b.ResetTimer()

				for b.Loop() {
					_ = bq.EncodeUint64(vec)
				}
			})

			// Benchmark distance
			b.Run("Distance", func(b *testing.B) {
				vec1 := randomVector(dim)
				vec2 := randomVector(dim)
				encoded1 := bq.EncodeUint64(vec1)
				encoded2 := bq.EncodeUint64(vec2)
				b.ResetTimer()

				for b.Loop() {
					_ = quantization.HammingDistance(encoded1, encoded2)
				}
			})
		})
	}
}

// BenchmarkProductQuantization benchmarks PQ operations
func BenchmarkProductQuantization(b *testing.B) {
	dim := 384
	numSubvectors := 8
	numCentroids := 256

	pq, err := quantization.NewProductQuantizer(dim, numSubvectors, numCentroids)
	if err != nil {
		b.Fatal(err)
	}

	// Train
	trainingSize := 10000
	training := make([][]float32, trainingSize)
	for i := range trainingSize {
		training[i] = randomVector(dim)
	}
	pq.Train(training)

	b.Run("Encode", func(b *testing.B) {
		vec := randomVector(dim)
		b.ResetTimer()

		for b.Loop() {
			_ = pq.Encode(vec)
		}
	})

	b.Run("Decode", func(b *testing.B) {
		vec := randomVector(dim)
		codes := pq.Encode(vec)
		b.ResetTimer()

		for b.Loop() {
			_ = pq.Decode(codes)
		}
	})

	b.Run("AsymmetricDistance", func(b *testing.B) {
		query := randomVector(dim)
		vec := randomVector(dim)
		codes := pq.Encode(vec)
		b.ResetTimer()

		for b.Loop() {
			_ = pq.ComputeAsymmetricDistance(query, codes)
		}
	})
}

// BenchmarkOptimizedPQ benchmarks OPQ operations
func BenchmarkOptimizedPQ(b *testing.B) {
	dim := 384
	numSubvectors := 8
	numCentroids := 256
	iterations := 10

	opq, err := quantization.NewOptimizedProductQuantizer(dim, numSubvectors, numCentroids, iterations)
	if err != nil {
		b.Fatal(err)
	}

	// Train
	trainingSize := 10000
	training := make([][]float32, trainingSize)
	for i := 0; i < trainingSize; i++ {
		training[i] = randomVector(dim)
	}

	b.Run("Train", func(b *testing.B) {
		b.ResetTimer()
		for b.Loop() {
			opq.Train(training)
		}
	})

	opq.Train(training)

	b.Run("Encode", func(b *testing.B) {
		vec := randomVector(dim)
		b.ResetTimer()

		for b.Loop() {
			_ = opq.Encode(vec)
		}
	})

	b.Run("AsymmetricDistance", func(b *testing.B) {
		query := randomVector(dim)
		vec := randomVector(dim)
		codes := opq.Encode(vec)
		b.ResetTimer()

		for b.Loop() {
			_ = opq.ComputeAsymmetricDistance(query, codes)
		}
	})
}

// BenchmarkFlatWithPQ benchmarks Flat index with PQ
func BenchmarkFlatWithPQ(b *testing.B) {
	dim := 384
	size := 10000

	db, err := vecgo.Flat[int](dim).SquaredL2().Build()
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

	// Enable PQ
	err = db.EnableProductQuantization(index.ProductQuantizationConfig{
		NumSubvectors: 8,
		NumCentroids:  256,
	})
	if err != nil {
		b.Fatal(err)
	}

	query := randomVector(dim)
	b.ResetTimer()

	for b.Loop() {
		_, err := db.Search(query).KNN(10).Execute(ctx)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkScalarQuantization benchmarks 8-bit scalar quantization
func BenchmarkScalarQuantization(b *testing.B) {
	dimensions := []int{128, 384, 768}

	for _, dim := range dimensions {
		b.Run(formatDim(dim), func(b *testing.B) {
			sq := quantization.NewScalarQuantizer(dim)

			// Train
			trainingSize := 1000
			training := make([][]float32, trainingSize)
			for i := range trainingSize {
				training[i] = randomVector(dim)
			}
			sq.Train(training)

			b.Run("Encode", func(b *testing.B) {
				vec := randomVector(dim)
				b.ResetTimer()

				for b.Loop() {
					_ = sq.Encode(vec)
				}
			})

			b.Run("Decode", func(b *testing.B) {
				vec := randomVector(dim)
				encoded := sq.Encode(vec)
				b.ResetTimer()

				for b.Loop() {
					_ = sq.Decode(encoded)
				}
			})
		})
	}
}
