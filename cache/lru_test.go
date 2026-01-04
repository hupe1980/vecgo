package cache

import (
	"context"
	"testing"

	"github.com/hupe1980/vecgo/resource"
	"github.com/stretchr/testify/assert"
)

func TestLRUBlockCache(t *testing.T) {
	rc := resource.NewController(resource.Config{MemoryLimitBytes: 100})
	c := NewLRUBlockCache(50, rc) // Cache limit 50, Global limit 100
	ctx := context.Background()

	k1 := CacheKey{SegmentID: 1, Offset: 1}
	v1 := make([]byte, 20)

	k2 := CacheKey{SegmentID: 1, Offset: 2}
	v2 := make([]byte, 20)

	k3 := CacheKey{SegmentID: 1, Offset: 3}
	v3 := make([]byte, 20)

	// 1. Set k1 (20 bytes)
	c.Set(ctx, k1, v1)
	assert.Equal(t, int64(20), c.Size())
	assert.Equal(t, int64(20), rc.MemoryUsage())

	// 2. Set k2 (20 bytes) -> Total 40
	c.Set(ctx, k2, v2)
	assert.Equal(t, int64(40), c.Size())
	assert.Equal(t, int64(40), rc.MemoryUsage())

	// 3. Set k3 (20 bytes) -> Total 60 > 50. Should evict k1 (LRU).
	c.Set(ctx, k3, v3)
	assert.Equal(t, int64(40), c.Size()) // 40 because 60-20=40
	assert.Equal(t, int64(40), rc.MemoryUsage())

	_, ok := c.Get(ctx, k1)
	assert.False(t, ok, "k1 should be evicted")

	_, ok = c.Get(ctx, k2)
	assert.True(t, ok, "k2 should be present")

	_, ok = c.Get(ctx, k3)
	assert.True(t, ok, "k3 should be present")
}

func TestLRUBlockCache_GlobalLimit(t *testing.T) {
	// Global limit smaller than cache limit
	rc := resource.NewController(resource.Config{MemoryLimitBytes: 30})
	c := NewLRUBlockCache(100, rc)
	ctx := context.Background()

	k1 := CacheKey{SegmentID: 1, Offset: 1}
	v1 := make([]byte, 20)

	k2 := CacheKey{SegmentID: 1, Offset: 2}
	v2 := make([]byte, 20)

	// 1. Set k1 (20 bytes)
	c.Set(ctx, k1, v1)
	assert.Equal(t, int64(20), c.Size())

	// 2. Set k2 (20 bytes) -> Total 40 > Global 30. Should fail to acquire and not cache.
	c.Set(ctx, k2, v2)
	assert.Equal(t, int64(20), c.Size())

	_, ok := c.Get(ctx, k2)
	assert.False(t, ok, "k2 should not be cached due to global limit")
}
