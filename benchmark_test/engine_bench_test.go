package vecgo_bench_test

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/hupe1980/vecgo"
	"github.com/hupe1980/vecgo/testutil"
)

// BenchmarkEngineInsert benchmarks insertion with and without MemTable (SyncWrite).
func BenchmarkEngineInsert(b *testing.B) {
	dim := 128

	scenarios := []struct {
		name      string
		syncWrite bool
	}{
		{"Async(MemTable)", false},
		{"Sync(NoMemTable)", true},
	}

	for _, sc := range scenarios {
		b.Run(sc.name, func(b *testing.B) {
			db, err := vecgo.HNSW[int](dim).
				SquaredL2().
				WithSyncWrite(sc.syncWrite).
				Build()
			if err != nil {
				b.Fatal(err)
			}

			rng := testutil.NewRNG(42)
			ctx := context.Background()

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				vec := rng.UnitVector(dim)
				_, err := db.Insert(ctx, vecgo.VectorWithData[int]{
					Vector: vec,
					Data:   i,
				})
				if err != nil {
					b.Fatal(err)
				}
			}
			b.StopTimer()
			db.Close()
		})
	}
}

// BenchmarkEngineSearch benchmarks search with and without MemTable usage during inserts.
// Note: This benchmark measures search performance while inserts might be happening or after.
// For pure search performance, the index structure is the same, but MemTable might add overhead if not empty.
func BenchmarkEngineSearch(b *testing.B) {
	dim := 128
	size := 10000
	k := 10

	scenarios := []struct {
		name      string
		syncWrite bool
	}{
		{"Async(MemTable)", false},
		{"Sync(NoMemTable)", true},
	}

	for _, sc := range scenarios {
		b.Run(sc.name, func(b *testing.B) {
			db, err := vecgo.HNSW[int](dim).
				SquaredL2().
				WithSyncWrite(sc.syncWrite).
				Build()
			if err != nil {
				b.Fatal(err)
			}

			rng := testutil.NewRNG(42)
			ctx := context.Background()

			// Pre-fill
			vectors := make([][]float32, size)
			for i := 0; i < size; i++ {
				vec := rng.UnitVector(dim)
				vectors[i] = vec
				_, err := db.Insert(ctx, vecgo.VectorWithData[int]{
					Vector: vec,
					Data:   i,
				})
				if err != nil {
					b.Fatal(err)
				}
			}

			query := rng.UnitVector(dim)
			groundTruth := testutil.BruteForceSearch(vectors, query, k)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				res, err := db.Search(query).KNN(k).Execute(ctx)
				if err != nil {
					b.Fatal(err)
				}

				if i == 0 {
					approx := make([]testutil.SearchResult, len(res))
					for j, r := range res {
						approx[j] = testutil.SearchResult{
							ID:       r.ID,
							Distance: r.Distance,
						}
					}
					recall := testutil.ComputeRecall(groundTruth, approx)
					b.ReportMetric(recall, "recall")
				}
			}
			b.StopTimer()
			db.Close()
		})
	}
}

// BenchmarkEngineConcurrency benchmarks mixed read/write throughput.
func BenchmarkEngineConcurrency(b *testing.B) {
	dim := 128
	initialSize := 10000
	k := 10

	scenarios := []struct {
		name      string
		shards    int
		readers   int
		writers   int
		syncWrite bool
	}{
		{"SingleShard/1W/1R", 1, 1, 1, false},
		{"SingleShard/1W/10R", 1, 10, 1, false},
		{"SingleShard/10W/10R", 1, 10, 10, false},
		{"Sharded4/1W/1R", 4, 1, 1, false},
		{"Sharded4/1W/10R", 4, 10, 1, false},
		{"Sharded4/10W/10R", 4, 10, 10, false},
	}

	for _, sc := range scenarios {
		b.Run(sc.name, func(b *testing.B) {
			var db *vecgo.Vecgo[int]
			var err error

			if sc.shards > 1 {
				db, err = vecgo.HNSW[int](dim).
					SquaredL2().
					WithSyncWrite(sc.syncWrite).
					Shards(sc.shards).
					Build()
			} else {
				db, err = vecgo.HNSW[int](dim).
					SquaredL2().
					WithSyncWrite(sc.syncWrite).
					Build()
			}
			if err != nil {
				b.Fatal(err)
			}
			defer db.Close()

			rng := testutil.NewRNG(42)
			ctx := context.Background()

			// Pre-fill
			for i := 0; i < initialSize; i++ {
				vec := rng.UnitVector(dim)
				_, err := db.Insert(ctx, vecgo.VectorWithData[int]{
					Vector: vec,
					Data:   i,
				})
				if err != nil {
					b.Fatal(err)
				}
			}

			b.ResetTimer()

			var wg sync.WaitGroup
			start := time.Now()

			// Readers
			for i := 0; i < sc.readers; i++ {
				wg.Add(1)
				go func() {
					defer wg.Done()
					query := rng.UnitVector(dim)
					for i := 0; i < b.N; i++ {
						_, _ = db.Search(query).KNN(k).Execute(ctx)
					}
				}()
			}

			// Writers
			for i := 0; i < sc.writers; i++ {
				wg.Add(1)
				go func() {
					defer wg.Done()
					vec := rng.UnitVector(dim)
					for i := 0; i < b.N; i++ {
						_, _ = db.Insert(ctx, vecgo.VectorWithData[int]{
							Vector: vec,
							Data:   i,
						})
					}
				}()
			}

			wg.Wait()
			elapsed := time.Since(start)
			ops := float64(b.N) * float64(sc.readers+sc.writers)
			b.ReportMetric(ops/elapsed.Seconds(), "ops/sec")
		})
	}
}
