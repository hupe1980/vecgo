package hnsw

import (
	"context"
	"sync"
	"testing"

	"github.com/hupe1980/vecgo/index"
	"github.com/hupe1980/vecgo/testutil"
)

// BenchmarkInsertSequential measures sequential insert performance
func BenchmarkInsertSequential(b *testing.B) {
	ctx := context.Background()
	vectors := testutil.NewRNG(42).UniformVectors(1000, 128)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; b.Loop(); i++ {
		h, err := New(func(o *Options) {
			o.Dimension = 128
			o.M = 16
			o.EF = 200
			o.DistanceType = index.DistanceTypeSquaredL2
		})
		if err != nil {
			b.Fatal(err)
		}

		for _, v := range vectors {
			_, err := h.Insert(ctx, v)
			if err != nil {
				b.Fatalf("Insert failed: %v", err)
			}
		}
	}
}

// BenchmarkInsertParallel measures parallel insert performance with fine-grained locking
func BenchmarkInsertParallel(b *testing.B) {
	ctx := context.Background()
	vectors := testutil.NewRNG(42).UniformVectors(1000, 128)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; b.Loop(); i++ {
		h, err := New(func(o *Options) {
			o.Dimension = 128
			o.M = 16
			o.EF = 200
			o.DistanceType = index.DistanceTypeSquaredL2
		})
		if err != nil {
			b.Fatal(err)
		}

		var wg sync.WaitGroup
		numWorkers := 8
		batchSize := len(vectors) / numWorkers

		for w := 0; w < numWorkers; w++ {
			wg.Add(1)
			go func(workerID int) {
				defer wg.Done()
				start := workerID * batchSize
				end := start + batchSize
				if workerID == numWorkers-1 {
					end = len(vectors)
				}

				for j := start; j < end; j++ {
					_, err := h.Insert(ctx, vectors[j])
					if err != nil {
						b.Errorf("Insert failed: %v", err)
					}
				}
			}(w)
		}

		wg.Wait()
	}
}

// BenchmarkBatchInsert measures batch insert performance
func BenchmarkBatchInsert(b *testing.B) {
	ctx := context.Background()
	vectors := testutil.NewRNG(42).UniformVectors(1000, 128)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; b.Loop(); i++ {
		h, err := New(func(o *Options) {
			o.Dimension = 128
			o.M = 16
			o.EF = 200
			o.DistanceType = index.DistanceTypeSquaredL2
		})
		if err != nil {
			b.Fatal(err)
		}

		_ = h.BatchInsert(ctx, vectors)
	}
}

// BenchmarkKNNSearch measures search performance (should be lock-free)
func BenchmarkKNNSearch(b *testing.B) {
	ctx := context.Background()
	dim := 128
	vectors := testutil.NewRNG(42).UniformVectors(10000, dim)
	query := testutil.NewRNG(42).UniformVectors(1, dim)[0]

	h, err := New(func(o *Options) {
		o.Dimension = dim
		o.M = 16
		o.EF = 200
		o.DistanceType = index.DistanceTypeSquaredL2
	})
	if err != nil {
		b.Fatal(err)
	}

	// Build index
	for _, v := range vectors {
		_, err := h.Insert(ctx, v)
		if err != nil {
			b.Fatalf("Insert failed: %v", err)
		}
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; b.Loop(); i++ {
		_, err := h.KNNSearch(ctx, query, 10, &index.SearchOptions{
			EFSearch: 50,
			Filter:   func(id uint64) bool { return true },
		})
		if err != nil {
			b.Fatalf("KNNSearch failed: %v", err)
		}
	}
}

// BenchmarkConcurrentSearchAndInsert measures concurrent search and insert performance
func BenchmarkConcurrentSearchAndInsert(b *testing.B) {
	ctx := context.Background()
	dim := 128
	vectors := testutil.NewRNG(42).UniformVectors(5000, dim)
	queries := testutil.NewRNG(42).UniformVectors(100, dim)

	h, err := New(func(o *Options) {
		o.Dimension = dim
		o.M = 16
		o.EF = 200
		o.DistanceType = index.DistanceTypeSquaredL2
	})
	if err != nil {
		b.Fatal(err)
	}

	// Build initial index
	for i := 0; i < 1000; i++ {
		_, err := h.Insert(ctx, vectors[i])
		if err != nil {
			b.Fatalf("Insert failed: %v", err)
		}
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; b.Loop(); i++ {
		var wg sync.WaitGroup

		// Launch search workers
		for w := 0; w < 4; w++ {
			wg.Add(1)
			go func(workerID int) {
				defer wg.Done()
				for j := 0; j < 25; j++ {
					queryIdx := (workerID*25 + j) % len(queries)
					_, err := h.KNNSearch(ctx, queries[queryIdx], 10, &index.SearchOptions{
						EFSearch: 50,
						Filter:   func(id uint64) bool { return true },
					})
					if err != nil {
						b.Errorf("KNNSearch failed: %v", err)
					}
				}
			}(w)
		}

		// Launch insert workers
		for w := 0; w < 2; w++ {
			wg.Add(1)
			go func(workerID int) {
				defer wg.Done()
				for j := 0; j < 50; j++ {
					vecIdx := (1000 + workerID*50 + j + i*100) % len(vectors)
					_, err := h.Insert(ctx, vectors[vecIdx])
					if err != nil {
						b.Errorf("Insert failed: %v", err)
					}
				}
			}(w)
		}

		wg.Wait()
	}
}

// BenchmarkIDReuse measures performance of ID reuse after deletions
func BenchmarkIDReuse(b *testing.B) {
	ctx := context.Background()
	dim := 128
	vectors := testutil.NewRNG(42).UniformVectors(2000, dim)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; b.Loop(); i++ {
		h, err := New(func(o *Options) {
			o.Dimension = dim
			o.M = 16
			o.EF = 200
			o.DistanceType = index.DistanceTypeSquaredL2
		})
		if err != nil {
			b.Fatal(err)
		}

		// Insert 1000 vectors
		ids := make([]uint64, 1000)
		for j := 0; j < 1000; j++ {
			id, err := h.Insert(ctx, vectors[j])
			if err != nil {
				b.Fatalf("Insert failed: %v", err)
			}
			ids[j] = id
		}

		// Delete half of them
		for j := 0; j < 500; j++ {
			err := h.Delete(ctx, ids[j])
			if err != nil {
				b.Fatalf("Delete failed: %v", err)
			}
		}

		// Insert 500 more - these should reuse IDs
		for j := 1000; j < 1500; j++ {
			_, err := h.Insert(ctx, vectors[j])
			if err != nil {
				b.Fatalf("Insert failed: %v", err)
			}
		}

		// Verify free list is being used
		if len(h.freeList) > 0 {
			b.Errorf("Expected free list to be empty after reuse, got %d entries", len(h.freeList))
		}
	}
}
