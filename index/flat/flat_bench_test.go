package flat_test

import (
	"context"
	"sync"
	"testing"

	"github.com/hupe1980/vecgo/index"
	"github.com/hupe1980/vecgo/index/flat"
	"github.com/hupe1980/vecgo/testutil"
)

// Benchmark single-threaded insert
func BenchmarkFlatInsert(b *testing.B) {
	dimensions := []int{128, 384, 768, 1536}

	for _, dim := range dimensions {
		b.Run(string(rune(dim)), func(b *testing.B) {
			f, err := flat.New(func(o *flat.Options) {
				o.Dimension = dim
				o.DistanceType = index.DistanceTypeSquaredL2
			})
			if err != nil {
				b.Fatal(err)
			}
			ctx := context.Background()
			rng := testutil.NewRNG(0)

			b.ResetTimer()
			b.ReportAllocs()

			for i := 0; b.Loop(); i++ {
				v := rng.UniformVectors(1, dim)[0]
				_, err := f.Insert(ctx, v)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// Benchmark batch insert
func BenchmarkFlatBatchInsert(b *testing.B) {
	batchSizes := []int{10, 100, 1000}
	dim := 128

	for _, batchSize := range batchSizes {
		b.Run(string(rune(batchSize)), func(b *testing.B) {
			f, err := flat.New(func(o *flat.Options) {
				o.Dimension = dim
				o.DistanceType = index.DistanceTypeSquaredL2
			})
			if err != nil {
				b.Fatal(err)
			}
			ctx := context.Background()
			rng := testutil.NewRNG(0)

			// Pre-generate vectors
			vectors := rng.UniformVectors(batchSize, dim)

			b.ResetTimer()
			b.ReportAllocs()

			for i := 0; b.Loop(); i++ {
				f.BatchInsert(ctx, vectors)
			}
		})
	}
}

// Benchmark single-threaded search
func BenchmarkFlatSearch(b *testing.B) {
	dimensions := []int{128, 384, 768}
	indexSizes := []int{1000, 10000}
	k := 10

	for _, dim := range dimensions {
		for _, size := range indexSizes {
			b.Run(string(rune(dim))+"_"+string(rune(size)), func(b *testing.B) {
				f, err := flat.New(func(o *flat.Options) {
					o.Dimension = dim
					o.DistanceType = index.DistanceTypeSquaredL2
				})
				if err != nil {
					b.Fatal(err)
				}
				ctx := context.Background()
				rng := testutil.NewRNG(0)

				// Build index
				vectors := rng.UniformVectors(size, dim)
				for i := 0; i < size; i++ {
					f.Insert(ctx, vectors[i])
				}

				query := rng.UniformVectors(1, dim)[0]

				b.ResetTimer()
				b.ReportAllocs()

				for i := 0; b.Loop(); i++ {
					_, err := f.KNNSearch(ctx, query, k, nil)
					if err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}

// Benchmark concurrent inserts
func BenchmarkFlatConcurrentInsert(b *testing.B) {
	goroutines := []int{2, 4, 8, 16}
	dim := 128

	for _, numGoroutines := range goroutines {
		b.Run(string(rune(numGoroutines))+"goroutines", func(b *testing.B) {
			f, err := flat.New(func(o *flat.Options) {
				o.Dimension = dim
				o.DistanceType = index.DistanceTypeSquaredL2
			})
			if err != nil {
				b.Fatal(err)
			}
			ctx := context.Background()

			b.ResetTimer()
			b.ReportAllocs()

			rng := testutil.NewRNG(0)

			var wg sync.WaitGroup
			insertsPerGoroutine := b.N / numGoroutines

			for g := 0; g < numGoroutines; g++ {
				wg.Add(1)
				go func() {
					defer wg.Done()
					for i := 0; i < insertsPerGoroutine; i++ {
						v := rng.UnitVector(dim)
						f.Insert(ctx, v)
					}
				}()
			}

			wg.Wait()
		})
	}
}

// Benchmark concurrent searches
func BenchmarkFlatConcurrentSearch(b *testing.B) {
	goroutines := []int{2, 4, 8, 16}
	dim := 128
	indexSize := 10000
	k := 10

	for _, numGoroutines := range goroutines {
		b.Run(string(rune(numGoroutines))+"goroutines", func(b *testing.B) {
			f, err := flat.New(func(o *flat.Options) {
				o.Dimension = dim
				o.DistanceType = index.DistanceTypeSquaredL2
			})
			if err != nil {
				b.Fatal(err)
			}
			ctx := context.Background()
			rng := testutil.NewRNG(0)

			// Build index
			for i := 0; i < indexSize; i++ {
				v := rng.UnitVector(dim)
				f.Insert(ctx, v)
			}

			b.ResetTimer()
			b.ReportAllocs()

			var wg sync.WaitGroup
			searchesPerGoroutine := b.N / numGoroutines

			for g := 0; g < numGoroutines; g++ {
				wg.Add(1)
				go func() {
					defer wg.Done()
					query := rng.UnitVector(dim)
					for i := 0; i < searchesPerGoroutine; i++ {
						f.KNNSearch(ctx, query, k, nil)
					}
				}()
			}

			wg.Wait()
		})
	}
}

// Benchmark mixed read/write workload
func BenchmarkFlatMixedWorkload(b *testing.B) {
	readWriteRatios := []struct {
		name     string
		readPct  int
		writePct int
	}{
		{"90read_10write", 90, 10},
		{"50read_50write", 50, 50},
		{"10read_90write", 10, 90},
	}

	dim := 128
	indexSize := 5000
	k := 10
	numGoroutines := 8

	for _, ratio := range readWriteRatios {
		b.Run(ratio.name, func(b *testing.B) {
			f, err := flat.New(func(o *flat.Options) {
				o.Dimension = dim
				o.DistanceType = index.DistanceTypeSquaredL2
			})
			if err != nil {
				b.Fatal(err)
			}
			ctx := context.Background()
			rng := testutil.NewRNG(0)

			// Build initial index
			for i := 0; i < indexSize; i++ {
				v := rng.UnitVector(dim)
				f.Insert(ctx, v)
			}

			b.ResetTimer()
			b.ReportAllocs()

			var wg sync.WaitGroup
			opsPerGoroutine := b.N / numGoroutines

			for g := 0; g < numGoroutines; g++ {
				wg.Add(1)
				go func() {
					defer wg.Done()
					for i := 0; i < opsPerGoroutine; i++ {
						if rng.Intn(100) < ratio.readPct {
							// Read operation
							query := rng.UnitVector(dim)
							f.KNNSearch(ctx, query, k, nil)
						} else {
							// Write operation
							v := rng.UnitVector(dim)
							f.Insert(ctx, v)
						}
					}
				}()
			}

			wg.Wait()
		})
	}
}

// Benchmark update operations
func BenchmarkFlatUpdate(b *testing.B) {
	dim := 128
	indexSize := 10000
	f, err := flat.New(func(o *flat.Options) {
		o.Dimension = dim
		o.DistanceType = index.DistanceTypeSquaredL2
	})
	if err != nil {
		b.Fatal(err)
	}
	ctx := context.Background()
	rng := testutil.NewRNG(0)

	// Build index
	ids := make([]uint64, indexSize)
	for i := 0; i < indexSize; i++ {
		v := rng.UnitVector(dim)
		id, _ := f.Insert(ctx, v)
		ids[i] = id
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; b.Loop(); i++ {
		id := ids[i%len(ids)]
		v := rng.UnitVector(dim)
		f.Update(ctx, id, v)
	}
}

// Benchmark delete operations
func BenchmarkFlatDelete(b *testing.B) {
	dim := 128

	b.ResetTimer()
	b.ReportAllocs()

	rng := testutil.NewRNG(0)

	for i := 0; b.Loop(); i++ {
		b.StopTimer()
		f, err := flat.New(func(o *flat.Options) {
			o.Dimension = dim
			o.DistanceType = index.DistanceTypeSquaredL2
		})
		if err != nil {
			b.Fatal(err)
		}
		ctx := context.Background()

		// Insert vectors to delete
		ids := make([]uint64, 1000)
		for j := 0; j < 1000; j++ {
			v := rng.UnitVector(dim)
			id, _ := f.Insert(ctx, v)
			ids[j] = id
		}
		b.StartTimer()

		// Delete all
		for _, id := range ids {
			f.Delete(ctx, id)
		}
	}
}
