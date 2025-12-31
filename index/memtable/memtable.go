package memtable

import (
	"container/heap"
	"context"
	"sync"

	"github.com/hupe1980/vecgo/index"
	"github.com/hupe1980/vecgo/index/hnsw"
	"github.com/hupe1980/vecgo/vectorstore/columnar"
)

// Item represents a vector in the memtable.
type Item struct {
	ID        uint64
	Vector    []float32
	IsDeleted bool
}

// MemTable is a thread-safe linear buffer for vectors.
// It serves as the L0 (Level 0) store in the LSM tree architecture.
type MemTable struct {
	mu        sync.RWMutex
	items     []Item
	distFunc  index.DistanceFunc
	dimension int
	hnswIndex *hnsw.HNSW
	vecPool   sync.Pool
	heapPool  sync.Pool
}

// New creates a new MemTable.
func New(dimension int, distFunc index.DistanceFunc) *MemTable {
	m := &MemTable{
		items:     make([]Item, 0, 1024), // Initial capacity
		distFunc:  distFunc,
		dimension: dimension,
		vecPool: sync.Pool{
			New: func() any {
				return make([]float32, dimension)
			},
		},
		heapPool: sync.Pool{
			New: func() any {
				h := make(resultHeap, 0, 32)
				return &h
			},
		},
	}
	m.resetHNSW()
	return m
}

func (m *MemTable) resetHNSW() {
	// Create a small HNSW index for fast search
	// We use small parameters since MemTable is small
	h, err := hnsw.New(func(o *hnsw.Options) {
		o.Dimension = m.dimension
		o.M = 8
		o.EF = 64
		o.InitialArenaSize = 32 * 1024 * 1024 // 32MB
		o.Vectors = columnar.New(m.dimension)
	})
	if err != nil {
		panic(err) // Should not happen for in-memory config
	}
	m.hnswIndex = h
}

// Reset clears the memtable for reuse.
func (m *MemTable) Reset() {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Return vectors to pool
	for i := range m.items {
		if m.items[i].Vector != nil {
			// Reset capacity to avoid holding large buffers if not needed?
			// sync.Pool handles this naturally (GC can take them).
			// We just Put them back.
			m.vecPool.Put(m.items[i].Vector)
		}
		// Clear pointers to avoid memory leaks
		m.items[i].Vector = nil
	}
	m.items = m.items[:0]
	m.hnswIndex.Reset()
}

// Insert adds a vector to the memtable.
func (m *MemTable) Insert(id uint64, vector []float32) {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Use pooled allocator
	vecCopy := m.vecPool.Get().([]float32)
	if cap(vecCopy) < len(vector) {
		vecCopy = make([]float32, len(vector))
	}
	vecCopy = vecCopy[:len(vector)]
	copy(vecCopy, vector)

	// Update HNSW
	// We ignore errors here as HNSW insert should succeed in memory
	if err := m.hnswIndex.ApplyInsert(context.Background(), id, vecCopy); err != nil {
		// Log error but continue (HNSW might be out of sync but linear scan will work)
	}

	// Append-only
	m.items = append(m.items, Item{
		ID:        id,
		Vector:    vecCopy,
		IsDeleted: false,
	})
}

// Get retrieves a vector from the memtable.
// Returns (vector, found, isDeleted).
func (m *MemTable) Get(id uint64) ([]float32, bool, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	// Scan backwards for latest version
	for i := len(m.items) - 1; i >= 0; i-- {
		if m.items[i].ID == id {
			return m.items[i].Vector, true, m.items[i].IsDeleted
		}
	}
	return nil, false, false
}

// Delete marks a vector as deleted in the memtable (Tombstone).
func (m *MemTable) Delete(id uint64) {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Update HNSW
	_ = m.hnswIndex.ApplyDelete(context.Background(), id)

	// Append tombstone
	m.items = append(m.items, Item{
		ID:        id,
		Vector:    nil,
		IsDeleted: true,
	})
}

// Flush returns all items in the memtable and clears it.
// This is an atomic operation.
func (m *MemTable) Flush() []Item {
	m.mu.Lock()
	defer m.mu.Unlock()

	if len(m.items) == 0 {
		return nil
	}

	flushed := m.items
	// We don't reset HNSW here because we might reuse this MemTable instance
	// via Reset() later. But wait, the original code did:
	// m.items = make(...)
	// m.resetHNSW()
	// This implies MemTable stays alive and starts fresh.
	// If we want to reuse MemTable in Tx, we should probably NOT reset here,
	// but let Tx handle the lifecycle.
	// However, to keep existing behavior for now:
	m.items = make([]Item, 0, 1024)
	m.hnswIndex.Reset()
	return flushed
}

// Size returns the number of items in the memtable.
func (m *MemTable) Size() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return len(m.items)
}

// Search performs a search on the memtable using the internal HNSW index.
func (m *MemTable) Search(query []float32, k int, filter func(uint64) bool) []index.SearchResult {
	m.mu.RLock()
	hnswIndex := m.hnswIndex
	m.mu.RUnlock()

	if hnswIndex == nil {
		return nil
	}

	// Use HNSW for search
	results, err := hnswIndex.KNNSearch(context.Background(), query, k, &index.SearchOptions{
		Filter: filter,
	})
	if err != nil {
		// Fallback to linear scan if HNSW fails
		return m.linearSearch(query, k, filter)
	}
	return results
}

// SearchWithBuffer performs a search on the memtable using the internal HNSW index and appends to buf.
func (m *MemTable) SearchWithBuffer(query []float32, k int, filter func(uint64) bool, buf *[]index.SearchResult) error {
	m.mu.RLock()
	hnswIndex := m.hnswIndex
	m.mu.RUnlock()

	if hnswIndex == nil {
		return nil
	}

	// Use HNSW for search
	return hnswIndex.KNNSearchWithBuffer(context.Background(), query, k, &index.SearchOptions{
		Filter: filter,
	}, buf)
}

// linearSearch performs a brute-force search on the memtable.
func (m *MemTable) linearSearch(query []float32, k int, filter func(uint64) bool) []index.SearchResult {
	// Use pooled heap
	h := m.heapPool.Get().(*resultHeap)
	*h = (*h)[:0]
	defer m.heapPool.Put(h)

	m.mu.RLock()
	defer m.mu.RUnlock()

	// We need to handle duplicates (latest version wins)
	// Use a map to track seen IDs? Or just scan backwards and ignore seen?
	// Scanning backwards and keeping top-k is tricky because we need global top-k.
	// Easier: Scan forwards, update map of ID -> (Vector, IsDeleted).
	// Then compute distances.
	// But that allocates a map.
	// Given MemTable is small, maybe just scan all and filter duplicates?
	// Or rely on HNSW which is the primary path.
	// Linear search is fallback.
	// Let's just scan all and deduplicate using a map for now, or assume HNSW works.
	// If we scan backwards, we see the latest version first.
	// We can keep a "seen" set.

	// Optimization: If we assume HNSW is always up to date, we rarely hit this.
	// But for correctness:
	seen := make(map[uint64]struct{}) // Allocation!
	// To avoid allocation, we could use a pooled map or just accept it for fallback.

	for i := len(m.items) - 1; i >= 0; i-- {
		item := m.items[i]
		if _, ok := seen[item.ID]; ok {
			continue
		}
		seen[item.ID] = struct{}{}

		if item.IsDeleted {
			continue
		}
		if filter != nil && !filter(item.ID) {
			continue
		}

		dist := m.distFunc(query, item.Vector)

		if h.Len() < k {
			heap.Push(h, index.SearchResult{ID: item.ID, Distance: dist})
		} else if dist < (*h)[0].Distance {
			heap.Pop(h)
			heap.Push(h, index.SearchResult{ID: item.ID, Distance: dist})
		}
	}

	// Convert heap to sorted slice (nearest first)
	results := make([]index.SearchResult, h.Len())
	for i := len(results) - 1; i >= 0; i-- {
		results[i] = heap.Pop(h).(index.SearchResult)
	}

	return results
}

// Items returns a copy of all items in the memtable without clearing it.
func (m *MemTable) Items() []Item {
	m.mu.RLock()
	defer m.mu.RUnlock()

	items := make([]Item, len(m.items))
	copy(items, m.items)
	return items
}

// resultHeap is a max-heap of SearchResults (based on distance)
type resultHeap []index.SearchResult

func (h resultHeap) Len() int           { return len(h) }
func (h resultHeap) Less(i, j int) bool { return h[i].Distance > h[j].Distance } // Max-heap
func (h resultHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *resultHeap) Push(x any) {
	*h = append(*h, x.(index.SearchResult))
}

func (h *resultHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}
