// Package engine provides the core database engine.
package engine

import (
	"hash/maphash"
	"sync"
	"sync/atomic"
	"time"

	"github.com/hupe1980/vecgo/metadata"
)

// NegativeCache caches filter signatures that produced empty results.
// This enables instant empty returns for repeated bad queries without
// any segment I/O or bitmap computation.
//
// Use cases:
//   - Dashboard queries with stale filters
//   - Exploratory queries hitting empty partitions
//   - Repeated queries with impossible predicates
//
// The cache is LRU-based with TTL expiration to handle data changes.
type NegativeCache struct {
	mu      sync.RWMutex
	entries map[uint64]*negativeCacheEntry
	order   []uint64 // LRU order (oldest first)
	maxSize int
	ttl     time.Duration
	seed    maphash.Seed

	// Stats
	hits   atomic.Uint64
	misses atomic.Uint64
}

type negativeCacheEntry struct {
	signature  uint64
	expireAt   time.Time
	emptySegs  []uint64 // Segment IDs that were empty for this filter
	queryCount atomic.Uint64
}

// NegativeCacheConfig configures the negative cache.
type NegativeCacheConfig struct {
	// MaxSize is the maximum number of cached filter signatures.
	// Default: 1000
	MaxSize int

	// TTL is how long to cache negative results before re-checking.
	// Default: 5 minutes
	TTL time.Duration

	// Enabled controls whether negative caching is active.
	// Default: true
	Enabled bool
}

// DefaultNegativeCacheConfig returns the default configuration.
func DefaultNegativeCacheConfig() NegativeCacheConfig {
	return NegativeCacheConfig{
		MaxSize: 1000,
		TTL:     5 * time.Minute,
		Enabled: true,
	}
}

// NewNegativeCache creates a new negative result cache.
func NewNegativeCache(cfg NegativeCacheConfig) *NegativeCache {
	if cfg.MaxSize <= 0 {
		cfg.MaxSize = 1000
	}
	if cfg.TTL <= 0 {
		cfg.TTL = 5 * time.Minute
	}
	return &NegativeCache{
		entries: make(map[uint64]*negativeCacheEntry, cfg.MaxSize),
		order:   make([]uint64, 0, cfg.MaxSize),
		maxSize: cfg.MaxSize,
		ttl:     cfg.TTL,
		seed:    maphash.MakeSeed(),
	}
}

// FilterSignature computes a hash signature for a filter.
// Same filter predicates produce the same signature.
func (nc *NegativeCache) FilterSignature(filters []metadata.Filter) uint64 {
	if len(filters) == 0 {
		return 0
	}

	var h maphash.Hash
	h.SetSeed(nc.seed)

	for _, f := range filters {
		// Write key
		h.WriteString(f.Key)
		// Write operator (as string since Operator is string type)
		h.WriteString(string(f.Operator))
		// Write value type and content
		h.WriteByte(byte(f.Value.Kind))
		switch f.Value.Kind {
		case metadata.KindInt:
			// Write int64 as bytes
			v := f.Value.I64
			for i := 0; i < 8; i++ {
				h.WriteByte(byte(v >> (i * 8)))
			}
		case metadata.KindFloat:
			// Write float64 bits as bytes using Key() for stability
			h.WriteString(f.Value.Key())
		case metadata.KindString:
			h.WriteString(f.Value.StringValue())
		case metadata.KindBool:
			if f.Value.B {
				h.WriteByte(1)
			} else {
				h.WriteByte(0)
			}
		default:
			h.WriteString(f.Value.Key())
		}
	}

	return h.Sum64()
}

// CheckEmpty returns true if this filter signature is cached as empty.
// Also returns the list of segment IDs known to be empty for this filter.
func (nc *NegativeCache) CheckEmpty(sig uint64) (isEmpty bool, emptySegs []uint64) {
	if sig == 0 {
		nc.misses.Add(1)
		return false, nil
	}

	nc.mu.RLock()
	entry, ok := nc.entries[sig]
	nc.mu.RUnlock()

	if !ok {
		nc.misses.Add(1)
		return false, nil
	}

	// Check TTL
	if time.Now().After(entry.expireAt) {
		nc.misses.Add(1)
		// Don't remove here to avoid write lock in hot path
		// Cleanup happens during Put
		return false, nil
	}

	nc.hits.Add(1)
	entry.queryCount.Add(1)
	return true, entry.emptySegs
}

// RecordEmpty records that a filter signature produced empty results.
// emptySegs is the list of segment IDs that were empty for this filter.
func (nc *NegativeCache) RecordEmpty(sig uint64, emptySegs []uint64) {
	if sig == 0 {
		return
	}

	nc.mu.Lock()
	defer nc.mu.Unlock()

	// Check if already exists
	if entry, ok := nc.entries[sig]; ok {
		// Update expiration and segments
		entry.expireAt = time.Now().Add(nc.ttl)
		entry.emptySegs = emptySegs
		// Move to end of LRU
		nc.moveToEnd(sig)
		return
	}

	// Evict if at capacity
	nc.evictExpiredLocked()
	if len(nc.entries) >= nc.maxSize {
		nc.evictOldestLocked()
	}

	// Add new entry
	nc.entries[sig] = &negativeCacheEntry{
		signature: sig,
		expireAt:  time.Now().Add(nc.ttl),
		emptySegs: emptySegs,
	}
	nc.order = append(nc.order, sig)
}

// Invalidate removes entries that might be affected by data changes.
// Call this after inserts/deletes that could make previously-empty
// queries return results.
func (nc *NegativeCache) Invalidate(segmentIDs []uint64) {
	if len(segmentIDs) == 0 {
		return
	}

	nc.mu.Lock()
	defer nc.mu.Unlock()

	// Build set of affected segment IDs
	affected := make(map[uint64]struct{}, len(segmentIDs))
	for _, id := range segmentIDs {
		affected[id] = struct{}{}
	}

	// Remove entries that reference affected segments
	toRemove := make([]uint64, 0)
	for sig, entry := range nc.entries {
		for _, segID := range entry.emptySegs {
			if _, ok := affected[segID]; ok {
				toRemove = append(toRemove, sig)
				break
			}
		}
	}

	for _, sig := range toRemove {
		delete(nc.entries, sig)
		nc.removeFromOrder(sig)
	}
}

// InvalidateAll clears the entire cache.
// Call this after bulk operations or schema changes.
func (nc *NegativeCache) InvalidateAll() {
	nc.mu.Lock()
	defer nc.mu.Unlock()

	nc.entries = make(map[uint64]*negativeCacheEntry, nc.maxSize)
	nc.order = nc.order[:0]
}

// Stats returns cache statistics.
func (nc *NegativeCache) Stats() NegativeCacheStats {
	nc.mu.RLock()
	size := len(nc.entries)
	nc.mu.RUnlock()

	return NegativeCacheStats{
		Size:   size,
		Hits:   nc.hits.Load(),
		Misses: nc.misses.Load(),
	}
}

// NegativeCacheStats contains cache statistics.
type NegativeCacheStats struct {
	Size   int
	Hits   uint64
	Misses uint64
}

// HitRate returns the cache hit rate (0.0-1.0).
func (s NegativeCacheStats) HitRate() float64 {
	total := s.Hits + s.Misses
	if total == 0 {
		return 0
	}
	return float64(s.Hits) / float64(total)
}

// evictExpiredLocked removes expired entries. Must hold write lock.
func (nc *NegativeCache) evictExpiredLocked() {
	now := time.Now()
	toRemove := make([]uint64, 0)

	for sig, entry := range nc.entries {
		if now.After(entry.expireAt) {
			toRemove = append(toRemove, sig)
		}
	}

	for _, sig := range toRemove {
		delete(nc.entries, sig)
		nc.removeFromOrder(sig)
	}
}

// evictOldestLocked removes the oldest entry. Must hold write lock.
func (nc *NegativeCache) evictOldestLocked() {
	if len(nc.order) == 0 {
		return
	}

	oldest := nc.order[0]
	nc.order = nc.order[1:]
	delete(nc.entries, oldest)
}

// moveToEnd moves a signature to the end of the LRU order. Must hold write lock.
func (nc *NegativeCache) moveToEnd(sig uint64) {
	nc.removeFromOrder(sig)
	nc.order = append(nc.order, sig)
}

// removeFromOrder removes a signature from the LRU order. Must hold write lock.
func (nc *NegativeCache) removeFromOrder(sig uint64) {
	for i, s := range nc.order {
		if s == sig {
			nc.order = append(nc.order[:i], nc.order[i+1:]...)
			return
		}
	}
}
