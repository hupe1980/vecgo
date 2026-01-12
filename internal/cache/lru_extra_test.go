package cache

import (
	"context"
	"testing"

	"github.com/hupe1980/vecgo/internal/resource"
	"github.com/stretchr/testify/assert"
)

func TestLRU_EdgeCases(t *testing.T) {
	rc := resource.NewController(resource.Config{MemoryLimitBytes: 100})
	c := NewLRUBlockCache(50, rc) // Cache cap 50
	ctx := context.Background()
	k := CacheKey{SegmentID: 1, Offset: 1}

	// 1. Item larger than capacity
	big := make([]byte, 60)
	c.Set(ctx, k, big)
	_, ok := c.Get(ctx, k)
	assert.False(t, ok, "Item > capacity should not be cached")

	// 2. Update existing item
	v1 := make([]byte, 10)
	c.Set(ctx, k, v1)
	assert.Equal(t, int64(10), c.Size())

	// Reset RC usage to known state? RC has 10 usage now.

	// Update with larger (20 bytes) -> +10 bytes
	v2 := make([]byte, 20)
	c.Set(ctx, k, v2)
	assert.Equal(t, int64(20), c.Size())

	// Update with smaller (5 bytes) -> -15 bytes
	v3 := make([]byte, 5)
	c.Set(ctx, k, v3)
	assert.Equal(t, int64(5), c.Size())

	// 3. Update fails due to RC limit
	// Current usage: 5 in cache. RC usage matches cache.
	// Force RC limit.
	// Make a new full RC logic.
	rc2 := resource.NewController(resource.Config{MemoryLimitBytes: 10}) // Limit 10
	c2 := NewLRUBlockCache(50, rc2)
	c2.Set(ctx, k, make([]byte, 8)) // RC usages 8

	// Try update to 12 bytes -> would need +4 bytes. RC limit 10. 8+4=12 > 10.
	// Should fail update.
	c2.Set(ctx, k, make([]byte, 12))

	val, ok := c2.Get(ctx, k)
	assert.True(t, ok)
	assert.Len(t, val, 8, "Update should have been rejected by RC")
}

func TestLRU_Stats_Coverage(t *testing.T) {
	c := NewLRUBlockCache(100, nil)
	ctx := context.Background()
	k := CacheKey{SegmentID: 1, Offset: 1}
	c.Set(ctx, k, []byte{1})
	c.Get(ctx, k)                                 // Hit
	c.Get(ctx, CacheKey{SegmentID: 2, Offset: 2}) // Miss

	hits, misses := c.Stats()
	assert.Equal(t, int64(1), hits)
	assert.Equal(t, int64(1), misses)
}

func TestLRU_Invalidate(t *testing.T) {
	c := NewLRUBlockCache(100, nil)
	ctx := context.Background()
	c.Set(ctx, CacheKey{SegmentID: 1, Offset: 1}, []byte("a"))
	c.Set(ctx, CacheKey{SegmentID: 1, Offset: 2}, []byte("b"))
	c.Set(ctx, CacheKey{SegmentID: 2, Offset: 1}, []byte("c"))

	// Invalidate segment 1
	c.Invalidate(func(k CacheKey) bool {
		return k.SegmentID == 1
	})

	_, ok := c.Get(ctx, CacheKey{SegmentID: 1, Offset: 1})
	assert.False(t, ok)
	_, ok = c.Get(ctx, CacheKey{SegmentID: 2, Offset: 1})
	assert.True(t, ok)
}
