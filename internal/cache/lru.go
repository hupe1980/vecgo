package cache

import (
	"container/list"
	"context"
	"sync"
	"sync/atomic"

	"github.com/hupe1980/vecgo/internal/resource"
)

// LRUBlockCache implements a simple LRU BlockCache.
type LRUBlockCache struct {
	mu        sync.Mutex
	capacity  int64
	size      int64
	items     map[CacheKey]*list.Element
	evictList *list.List
	rc        *resource.Controller

	hits   atomic.Int64
	misses atomic.Int64
}

type entry struct {
	key   CacheKey
	value []byte
}

// NewLRUBlockCache creates a new LRU cache with the given capacity in bytes.
// If rc is provided, it will be used to track memory usage.
func NewLRUBlockCache(capacity int64, rc *resource.Controller) *LRUBlockCache {
	return &LRUBlockCache{
		capacity:  capacity,
		items:     make(map[CacheKey]*list.Element),
		evictList: list.New(),
		rc:        rc,
	}
}

// Get returns a cached block.
func (c *LRUBlockCache) Get(ctx context.Context, key CacheKey) ([]byte, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if ent, ok := c.items[key]; ok {
		c.hits.Add(1)
		c.evictList.MoveToFront(ent)
		return ent.Value.(*entry).value, true
	}
	c.misses.Add(1)
	return nil, false
}

// Set caches a block.
func (c *LRUBlockCache) Set(ctx context.Context, key CacheKey, b []byte) {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Check if already exists
	if ent, ok := c.items[key]; ok {
		c.evictList.MoveToFront(ent)
		// Update value
		oldSize := int64(len(ent.Value.(*entry).value))
		newSize := int64(len(b))
		if c.rc != nil && newSize > oldSize {
			// If the global ResourceController denies the growth, keep the old value.
			if !c.rc.TryAcquireMemory(newSize - oldSize) {
				return
			}
		}

		// Update size tracking
		c.size += newSize - oldSize
		if c.rc != nil {
			if newSize < oldSize {
				c.rc.ReleaseMemory(oldSize - newSize)
			}
		}

		ent.Value.(*entry).value = b
		c.evict()
		return
	}

	// Add new item
	itemSize := int64(len(b))

	// Check capacity before adding
	// If item is larger than capacity, don't cache
	if itemSize > c.capacity {
		return
	}

	// Evict to make space in local capacity first
	// This helps releasing memory to RC before we try to acquire it back
	for c.size+itemSize > c.capacity {
		ent := c.evictList.Back()
		if ent == nil {
			break
		}
		c.removeElement(ent)
	}

	// Acquire memory from RC
	if c.rc != nil {
		// We use TryAcquire here to avoid blocking on cache set
		if !c.rc.TryAcquireMemory(itemSize) {
			// If we can't acquire, we might need to evict to make space in RC?
			// Or just don't cache.
			// But RC tracks GLOBAL memory. If we are within cache capacity but global limit is hit,
			// we should probably respect global limit.
			// However, if we evict from cache, we release memory to RC.
			// So we can try to evict until we can acquire.

			// For now, simple strategy: if RC says no, don't cache.
			return
		}
	}

	ent := &entry{key, b}
	element := c.evictList.PushFront(ent)
	c.items[key] = element
	c.size += itemSize
}

// Invalidate removes entries matching the predicate.
func (c *LRUBlockCache) Invalidate(predicate func(key CacheKey) bool) {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Capture keys to remove to avoid modifying map while iterating (if map semantics require it)
	// Go maps are safe to delete during iteration, but modifying list in loop requires care.
	// However, removeElement modifies the list.
	// So we should collect elements first.
	var toRemove []*list.Element

	for key, element := range c.items {
		if predicate(key) {
			toRemove = append(toRemove, element)
		}
	}

	for _, e := range toRemove {
		c.removeElement(e)
	}
}

func (c *LRUBlockCache) evict() {
	for c.size > c.capacity {
		if c.evictList.Len() == 0 {
			break
		}
		element := c.evictList.Back()
		if element != nil {
			c.removeElement(element)
		}
	}
}

func (c *LRUBlockCache) Close() error {
	return nil
}

func (c *LRUBlockCache) Stats() (hits, misses int64) {
	return c.hits.Load(), c.misses.Load()
}

func (c *LRUBlockCache) removeElement(e *list.Element) {
	c.evictList.Remove(e)
	kv := e.Value.(*entry)
	delete(c.items, kv.key)
	itemSize := int64(len(kv.value))
	c.size -= itemSize
	if c.rc != nil {
		c.rc.ReleaseMemory(itemSize)
	}
}

// Size returns the current size of the cache in bytes.
func (c *LRUBlockCache) Size() int64 {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.size
}
