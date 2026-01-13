package cache

import (
	"context"
	"sync"
	"testing"

	"github.com/hupe1980/vecgo/model"
)

func TestShardedLRUBlockCache_BasicOperations(t *testing.T) {
	cache := NewShardedLRUBlockCache(1024*1024, nil) // 1MB

	ctx := context.Background()
	key := CacheKey{SegmentID: 1, Offset: 0}
	data := []byte("test data")

	// Test Set and Get
	cache.Set(ctx, key, data)
	got, ok := cache.Get(ctx, key)
	if !ok {
		t.Fatal("expected cache hit")
	}
	if string(got) != string(data) {
		t.Errorf("got %q, want %q", got, data)
	}

	// Test miss
	missKey := CacheKey{SegmentID: 999, Offset: 0}
	_, ok = cache.Get(ctx, missKey)
	if ok {
		t.Fatal("expected cache miss")
	}
}

func TestShardedLRUBlockCache_ShardDistribution(t *testing.T) {
	cache := NewShardedLRUBlockCache(64*1024*1024, nil) // 64MB

	ctx := context.Background()
	data := make([]byte, 1024) // 1KB

	// Insert 1000 items
	for i := range 1000 {
		key := CacheKey{SegmentID: model.SegmentID(i % 100), Offset: uint64(i * 4096)}
		cache.Set(ctx, key, data)
	}

	// Check that items are distributed across shards
	stats := cache.ShardStats()
	nonEmptyShards := 0
	for _, s := range stats {
		if s.Size > 0 {
			nonEmptyShards++
		}
	}

	// With 1000 items across 64 shards, we expect most shards to have items
	if nonEmptyShards < 30 {
		t.Errorf("poor shard distribution: only %d shards have items", nonEmptyShards)
	}
}

func TestShardedLRUBlockCache_Concurrent(t *testing.T) {
	cache := NewShardedLRUBlockCache(64*1024*1024, nil) // 64MB

	ctx := context.Background()
	data := make([]byte, 1024)

	const numGoroutines = 100
	const numOpsPerGoroutine = 1000

	var wg sync.WaitGroup
	wg.Add(numGoroutines)

	for g := range numGoroutines {
		go func(goroutineID int) {
			defer wg.Done()
			for i := range numOpsPerGoroutine {
				key := CacheKey{
					SegmentID: model.SegmentID(goroutineID),
					Offset:    uint64(i * 4096),
				}
				cache.Set(ctx, key, data)
				cache.Get(ctx, key)
			}
		}(g)
	}

	wg.Wait()

	hits, misses := cache.Stats()
	total := hits + misses
	if total != numGoroutines*numOpsPerGoroutine {
		t.Errorf("stats mismatch: got %d total, want %d", total, numGoroutines*numOpsPerGoroutine)
	}
}

func TestShardedLRUBlockCache_Invalidate(t *testing.T) {
	cache := NewShardedLRUBlockCache(64*1024*1024, nil)

	ctx := context.Background()
	data := []byte("test")

	// Insert items for segment 1 and 2
	for i := range 100 {
		cache.Set(ctx, CacheKey{SegmentID: 1, Offset: uint64(i * 4096)}, data)
		cache.Set(ctx, CacheKey{SegmentID: 2, Offset: uint64(i * 4096)}, data)
	}

	// Invalidate segment 1
	cache.Invalidate(func(key CacheKey) bool {
		return key.SegmentID == 1
	})

	// Check segment 1 is gone
	_, ok := cache.Get(ctx, CacheKey{SegmentID: 1, Offset: 0})
	if ok {
		t.Error("expected segment 1 to be invalidated")
	}

	// Check segment 2 is still there
	_, ok = cache.Get(ctx, CacheKey{SegmentID: 2, Offset: 0})
	if !ok {
		t.Error("expected segment 2 to still be cached")
	}
}

func BenchmarkLRUBlockCache_Get(b *testing.B) {
	cache := NewLRUBlockCache(64*1024*1024, nil)
	ctx := context.Background()
	key := CacheKey{SegmentID: 1, Offset: 0}
	cache.Set(ctx, key, make([]byte, 4096))

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			cache.Get(ctx, key)
		}
	})
}

func BenchmarkShardedLRUBlockCache_Get(b *testing.B) {
	cache := NewShardedLRUBlockCache(64*1024*1024, nil)
	ctx := context.Background()
	key := CacheKey{SegmentID: 1, Offset: 0}
	cache.Set(ctx, key, make([]byte, 4096))

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			cache.Get(ctx, key)
		}
	})
}

func BenchmarkLRUBlockCache_GetMixed(b *testing.B) {
	cache := NewLRUBlockCache(64*1024*1024, nil)
	ctx := context.Background()
	data := make([]byte, 4096)

	// Pre-populate
	for i := range 1000 {
		cache.Set(ctx, CacheKey{SegmentID: model.SegmentID(i % 10), Offset: uint64(i * 4096)}, data)
	}

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			key := CacheKey{SegmentID: model.SegmentID(i % 10), Offset: uint64(i * 4096)}
			cache.Get(ctx, key)
			i++
		}
	})
}

func BenchmarkShardedLRUBlockCache_GetMixed(b *testing.B) {
	cache := NewShardedLRUBlockCache(64*1024*1024, nil)
	ctx := context.Background()
	data := make([]byte, 4096)

	// Pre-populate
	for i := range 1000 {
		cache.Set(ctx, CacheKey{SegmentID: model.SegmentID(i % 10), Offset: uint64(i * 4096)}, data)
	}

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			key := CacheKey{SegmentID: model.SegmentID(i % 10), Offset: uint64(i * 4096)}
			cache.Get(ctx, key)
			i++
		}
	})
}
