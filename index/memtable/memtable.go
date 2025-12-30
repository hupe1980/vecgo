package memtable

import (
	"container/heap"
	"context"
	"sync"

	"github.com/hupe1980/vecgo/index"
	"github.com/hupe1980/vecgo/index/hnsw"
	"github.com/hupe1980/vecgo/internal/mem"
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
	idToIdx   map[uint64]int
	distFunc  index.DistanceFunc
	dimension int
	hnswIndex *hnsw.HNSW
}

// New creates a new MemTable.
func New(dimension int, distFunc index.DistanceFunc) *MemTable {
	m := &MemTable{
		items:     make([]Item, 0, 1024), // Initial capacity
		idToIdx:   make(map[uint64]int),
		distFunc:  distFunc,
		dimension: dimension,
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

// Insert adds a vector to the memtable.
func (m *MemTable) Insert(id uint64, vector []float32) {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Use aligned allocator for SIMD efficiency during brute-force search
	// We need to copy the vector because the source might be reused
	vecCopy := mem.AllocAlignedFloat32(len(vector))
	copy(vecCopy, vector)

	// Update HNSW
	// We ignore errors here as HNSW insert should succeed in memory
	if err := m.hnswIndex.ApplyInsert(context.Background(), id, vecCopy); err != nil {
		println("MemTable HNSW Insert failed:", err.Error())
	}

	if idx, ok := m.idToIdx[id]; ok {
		// Update existing
		m.items[idx].Vector = vecCopy
		m.items[idx].IsDeleted = false
		return
	}

	m.items = append(m.items, Item{
		ID:        id,
		Vector:    vecCopy,
		IsDeleted: false,
	})
	m.idToIdx[id] = len(m.items) - 1
}

// Get retrieves a vector from the memtable.
// Returns (vector, found, isDeleted).
func (m *MemTable) Get(id uint64) ([]float32, bool, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	idx, ok := m.idToIdx[id]
	if !ok {
		return nil, false, false
	}
	item := m.items[idx]
	return item.Vector, true, item.IsDeleted
}

// Delete marks a vector as deleted in the memtable (Tombstone).
func (m *MemTable) Delete(id uint64) {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Update HNSW
	_ = m.hnswIndex.ApplyDelete(context.Background(), id)

	if idx, ok := m.idToIdx[id]; ok {
		// Update existing to tombstone
		m.items[idx].Vector = nil
		m.items[idx].IsDeleted = true
		return
	}

	// Add new tombstone
	m.items = append(m.items, Item{
		ID:        id,
		Vector:    nil,
		IsDeleted: true,
	})
	m.idToIdx[id] = len(m.items) - 1
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
	m.items = make([]Item, 0, 1024) // Reset with fresh buffer
	m.idToIdx = make(map[uint64]int)
	m.resetHNSW() // Reset HNSW
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
	defer m.mu.RUnlock()

	if len(m.items) == 0 {
		return nil
	}

	// Use HNSW for search
	// We need to wrap the filter to check for deleted items in MemTable
	// (HNSW handles its own deletions, but we might have tombstones in MemTable that override HNSW?)
	// Actually, we update HNSW on Delete, so HNSW should be consistent.
	// But we still need to respect the external filter.

	results, err := m.hnswIndex.KNNSearch(context.Background(), query, k, &index.SearchOptions{
		Filter: filter,
	})
	if err != nil {
		println("MemTable HNSW Search failed:", err.Error())
		// Fallback to linear scan if HNSW fails (should not happen)
		return m.linearSearch(query, k, filter)
	}
	return results
}

// linearSearch performs a brute-force search on the memtable.
func (m *MemTable) linearSearch(query []float32, k int, filter func(uint64) bool) []index.SearchResult {
	// Use a max-heap to keep track of the top-k results
	h := &resultHeap{}
	heap.Init(h)

	for _, item := range m.items {
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
