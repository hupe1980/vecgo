package cache

import (
	"context"
	"sync"
	"sync/atomic"

	"github.com/hupe1980/vecgo/internal/resource"
)

const numShards = 64

// ShardedLRUBlockCache is a sharded LRU cache for high-concurrency workloads.
// It distributes entries across 64 shards to reduce lock contention.
type ShardedLRUBlockCache struct {
	shards [numShards]*LRUBlockCache
}

// NewShardedLRUBlockCache creates a new sharded LRU cache.
// The capacity is divided evenly across all shards.
func NewShardedLRUBlockCache(capacity int64, rc *resource.Controller) *ShardedLRUBlockCache {
	shardCapacity := capacity / numShards
	if shardCapacity < 1 {
		shardCapacity = 1
	}

	s := &ShardedLRUBlockCache{}

	for i := range numShards {
		s.shards[i] = NewLRUBlockCache(shardCapacity, rc)
	}

	return s
}

// shard returns the shard for a given key using fast XOR hash.
// Uses splitmix64 finalizer for good distribution.
func (s *ShardedLRUBlockCache) shard(key CacheKey) *LRUBlockCache {
	// Combine SegmentID and Offset with XOR and splitmix64 finalizer
	h := uint64(key.SegmentID) ^ key.Offset ^ key.ManifestID
	// splitmix64 finalizer for excellent distribution
	h = (h ^ (h >> 30)) * 0xbf58476d1ce4e5b9
	h = (h ^ (h >> 27)) * 0x94d049bb133111eb
	h ^= h >> 31
	return s.shards[h&(numShards-1)] // Faster than modulo for power-of-2
}

// Get returns a cached block.
func (s *ShardedLRUBlockCache) Get(ctx context.Context, key CacheKey) ([]byte, bool) {
	return s.shard(key).Get(ctx, key)
}

// Set caches a block.
func (s *ShardedLRUBlockCache) Set(ctx context.Context, key CacheKey, b []byte) {
	s.shard(key).Set(ctx, key, b)
}

// Invalidate removes entries matching the predicate.
// This iterates all shards, which is expensive but rare.
func (s *ShardedLRUBlockCache) Invalidate(predicate func(key CacheKey) bool) {
	var wg sync.WaitGroup
	wg.Add(numShards)

	for i := range numShards {
		go func(shard *LRUBlockCache) {
			defer wg.Done()
			shard.Invalidate(predicate)
		}(s.shards[i])
	}

	wg.Wait()
}

// Close closes all shards.
func (s *ShardedLRUBlockCache) Close() error {
	for i := range numShards {
		if err := s.shards[i].Close(); err != nil {
			return err
		}
	}
	return nil
}

// Stats returns aggregated hit/miss statistics.
func (s *ShardedLRUBlockCache) Stats() (hits, misses int64) {
	for i := range numShards {
		h, m := s.shards[i].Stats()
		hits += h
		misses += m
	}
	return hits, misses
}

// Size returns the total size across all shards.
func (s *ShardedLRUBlockCache) Size() int64 {
	var total int64
	for i := range numShards {
		total += s.shards[i].Size()
	}
	return total
}

// shardedCacheStats provides per-shard statistics for debugging.
type shardedCacheStats struct {
	ShardID int
	Size    int64
	Hits    int64
	Misses  int64
}

// ShardStats returns per-shard statistics.
func (s *ShardedLRUBlockCache) ShardStats() []shardedCacheStats {
	stats := make([]shardedCacheStats, numShards)
	for i := range numShards {
		h, m := s.shards[i].Stats()
		stats[i] = shardedCacheStats{
			ShardID: i,
			Size:    s.shards[i].Size(),
			Hits:    h,
			Misses:  m,
		}
	}
	return stats
}

// ShardedLRUBlockCacheWithCounter wraps ShardedLRUBlockCache with atomic counters
// for fast path statistics without per-shard aggregation.
type ShardedLRUBlockCacheWithCounter struct {
	*ShardedLRUBlockCache
	totalHits   atomic.Int64
	totalMisses atomic.Int64
}

// NewShardedLRUBlockCacheWithCounter creates a sharded cache with global counters.
func NewShardedLRUBlockCacheWithCounter(capacity int64, rc *resource.Controller) *ShardedLRUBlockCacheWithCounter {
	return &ShardedLRUBlockCacheWithCounter{
		ShardedLRUBlockCache: NewShardedLRUBlockCache(capacity, rc),
	}
}

// Get returns a cached block with global counter tracking.
func (s *ShardedLRUBlockCacheWithCounter) Get(ctx context.Context, key CacheKey) ([]byte, bool) {
	val, ok := s.ShardedLRUBlockCache.Get(ctx, key)
	if ok {
		s.totalHits.Add(1)
	} else {
		s.totalMisses.Add(1)
	}
	return val, ok
}

// FastStats returns the global hit/miss counters (faster than aggregating shards).
func (s *ShardedLRUBlockCacheWithCounter) FastStats() (hits, misses int64) {
	return s.totalHits.Load(), s.totalMisses.Load()
}
