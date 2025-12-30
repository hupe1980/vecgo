package engine

import (
	"context"
	"fmt"
	"math/rand"
	"sync/atomic"
	"testing"

	"github.com/hupe1980/vecgo/codec"
	"github.com/hupe1980/vecgo/index"
	"github.com/hupe1980/vecgo/index/hnsw"
	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/wal"
)

func BenchmarkEngineConcurrency(b *testing.B) {
	// Setup
	dim := 128
	idx, err := hnsw.New(func(o *hnsw.Options) {
		o.Dimension = dim
		o.DistanceType = index.DistanceTypeSquaredL2
	})
	if err != nil {
		b.Fatal(err)
	}

	dataStore := NewMapStore[string]()
	metaStore := metadata.NewUnifiedIndex()

	walLog, err := wal.New(func(o *wal.Options) {
		o.Path = b.TempDir()
		o.DurabilityMode = wal.DurabilityAsync // Use Async for better throughput in benchmark
	})
	if err != nil {
		b.Fatal(err)
	}
	defer walLog.Close()

	coord, err := New(idx, dataStore, metaStore, walLog, codec.Default, WithDimension(dim))
	if err != nil {
		b.Fatal(err)
	}
	defer coord.Close()

	// Pre-populate with some data
	ctx := context.Background()
	initialSize := 10000
	for i := 0; i < initialSize; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = rand.Float32()
		}
		id := fmt.Sprintf("init-%d", i)
		if _, err := coord.Insert(ctx, vec, id, nil); err != nil {
			b.Fatal(err)
		}
	}

	// Helper to generate random vector
	randomVec := func() []float32 {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = rand.Float32()
		}
		return vec
	}

	b.ResetTimer()

	b.Run("ReadOnly", func(b *testing.B) {
		b.RunParallel(func(pb *testing.PB) {
			for pb.Next() {
				query := randomVec()
				_, err := coord.KNNSearch(ctx, query, 10, nil)
				if err != nil {
					b.Error(err)
				}
			}
		})
	})

	b.Run("WriteOnly", func(b *testing.B) {
		var counter int64
		b.RunParallel(func(pb *testing.PB) {
			for pb.Next() {
				id := atomic.AddInt64(&counter, 1)
				vec := randomVec()
				_, err := coord.Insert(ctx, vec, fmt.Sprintf("write-%d", id), nil)
				if err != nil {
					b.Error(err)
				}
			}
		})
	})

	b.Run("Mixed_80Read_20Write", func(b *testing.B) {
		var counter int64
		b.RunParallel(func(pb *testing.PB) {
			for pb.Next() {
				if rand.Float32() < 0.2 {
					// Write
					id := atomic.AddInt64(&counter, 1)
					vec := randomVec()
					_, err := coord.Insert(ctx, vec, fmt.Sprintf("mixed-%d", id), nil)
					if err != nil {
						b.Error(err)
					}
				} else {
					// Read
					query := randomVec()
					_, err := coord.KNNSearch(ctx, query, 10, nil)
					if err != nil {
						b.Error(err)
					}
				}
			}
		})
	})
}

func BenchmarkShardedEngineConcurrency(b *testing.B) {
	// Setup
	dim := 128
	numShards := 4
	indexes := make([]index.Index, numShards)
	dataStores := make([]Store[string], numShards)
	metaStores := make([]*metadata.UnifiedIndex, numShards)
	durabilities := make([]Durability, numShards)

	tmpDir := b.TempDir()

	for i := 0; i < numShards; i++ {
		var err error
		indexes[i], err = hnsw.New(func(o *hnsw.Options) {
			o.Dimension = dim
			o.DistanceType = index.DistanceTypeSquaredL2
		})
		if err != nil {
			b.Fatal(err)
		}
		dataStores[i] = NewMapStore[string]()
		metaStores[i] = metadata.NewUnifiedIndex()

		walLog, err := wal.New(func(o *wal.Options) {
			o.Path = fmt.Sprintf("%s/shard-%d", tmpDir, i)
			o.DurabilityMode = wal.DurabilityAsync
		})
		if err != nil {
			b.Fatal(err)
		}
		defer walLog.Close()
		durabilities[i] = walLog
	}

	coord, err := NewSharded(indexes, dataStores, metaStores, durabilities, codec.Default, WithDimension(dim))
	if err != nil {
		b.Fatal(err)
	}
	defer coord.Close()

	// Pre-populate with some data
	ctx := context.Background()
	initialSize := 10000
	for i := 0; i < initialSize; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = rand.Float32()
		}
		id := fmt.Sprintf("init-%d", i)
		if _, err := coord.Insert(ctx, vec, id, nil); err != nil {
			b.Fatal(err)
		}
	}

	// Helper to generate random vector
	randomVec := func() []float32 {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = rand.Float32()
		}
		return vec
	}

	b.ResetTimer()

	b.Run("ReadOnly", func(b *testing.B) {
		b.RunParallel(func(pb *testing.PB) {
			for pb.Next() {
				query := randomVec()
				_, err := coord.KNNSearch(ctx, query, 10, nil)
				if err != nil {
					b.Error(err)
				}
			}
		})
	})

	b.Run("WriteOnly", func(b *testing.B) {
		var counter int64
		b.RunParallel(func(pb *testing.PB) {
			for pb.Next() {
				id := atomic.AddInt64(&counter, 1)
				vec := randomVec()
				_, err := coord.Insert(ctx, vec, fmt.Sprintf("write-%d", id), nil)
				if err != nil {
					b.Error(err)
				}
			}
		})
	})

	b.Run("Mixed_80Read_20Write", func(b *testing.B) {
		var counter int64
		b.RunParallel(func(pb *testing.PB) {
			for pb.Next() {
				if rand.Float32() < 0.2 {
					// Write
					id := atomic.AddInt64(&counter, 1)
					vec := randomVec()
					_, err := coord.Insert(ctx, vec, fmt.Sprintf("mixed-%d", id), nil)
					if err != nil {
						b.Error(err)
					}
				} else {
					// Read
					query := randomVec()
					_, err := coord.KNNSearch(ctx, query, 10, nil)
					if err != nil {
						b.Error(err)
					}
				}
			}
		})
	})
}
