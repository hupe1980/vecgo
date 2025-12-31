package memtable

import (
	"container/heap"
	"sync"

	"github.com/hupe1980/vecgo/index"
)

// Item represents a vector in the memtable.
type Item struct {
	ID        uint64
	Vector    []float32
	IsDeleted bool
}

// MemTable is a thread-safe linear buffer for vectors.
// It serves as the L0 (Level 0) store in the LSM tree architecture.
// It uses a Struct of Arrays (SoA) layout for better cache locality.
type MemTable struct {
	mu        sync.RWMutex
	ids       []uint64
	vectors   []float32 // Flattened vectors
	deleted   []bool
	distFunc  index.DistanceFunc
	dimension int
	heapPool  sync.Pool
	mapPool   sync.Pool
}

// New creates a new MemTable.
func New(dimension int, distFunc index.DistanceFunc) *MemTable {
	m := &MemTable{
		ids:       make([]uint64, 0, 1024),
		vectors:   make([]float32, 0, 1024*dimension),
		deleted:   make([]bool, 0, 1024),
		distFunc:  distFunc,
		dimension: dimension,
		heapPool: sync.Pool{
			New: func() any {
				h := make(resultHeap, 0, 32)
				return &h
			},
		},
		mapPool: sync.Pool{
			New: func() any {
				return make(map[uint64]struct{}, 1024)
			},
		},
	}
	return m
}

// Reset clears the memtable for reuse.
func (m *MemTable) Reset() {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.ids = m.ids[:0]
	m.vectors = m.vectors[:0]
	m.deleted = m.deleted[:0]
}

// Insert adds a vector to the memtable.
func (m *MemTable) Insert(id uint64, vector []float32) {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Append-only (SoA)
	m.ids = append(m.ids, id)
	m.vectors = append(m.vectors, vector...)
	m.deleted = append(m.deleted, false)
}

// Get retrieves a vector from the memtable.
// Returns (vector, found, isDeleted).
func (m *MemTable) Get(id uint64) ([]float32, bool, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	// Scan backwards for latest version
	// SoA scan is faster due to cache locality of ids slice
	for i := len(m.ids) - 1; i >= 0; i-- {
		if m.ids[i] == id {
			if m.deleted[i] {
				return nil, true, true
			}
			// Extract vector from flattened slice
			start := i * m.dimension
			end := start + m.dimension
			// Return a copy to be safe
			vec := make([]float32, m.dimension)
			copy(vec, m.vectors[start:end])
			return vec, true, false
		}
	}
	return nil, false, false
}

// Contains checks if an ID exists in the memtable and returns (found, isDeleted).
// This avoids allocating the vector copy.
func (m *MemTable) Contains(id uint64) (bool, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	for i := len(m.ids) - 1; i >= 0; i-- {
		if m.ids[i] == id {
			return true, m.deleted[i]
		}
	}
	return false, false
}

// Delete marks a vector as deleted in the memtable (Tombstone).
func (m *MemTable) Delete(id uint64) {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Append tombstone
	m.ids = append(m.ids, id)
	// We need to append dummy vector data to keep alignment
	zeros := make([]float32, m.dimension)
	m.vectors = append(m.vectors, zeros...)
	m.deleted = append(m.deleted, true)
}

// Flush returns all items in the memtable and clears it.
// This is an atomic operation.
func (m *MemTable) Flush() []Item {
	m.mu.Lock()
	defer m.mu.Unlock()

	if len(m.ids) == 0 {
		return nil
	}

	// Reconstruct Items from SoA
	items := make([]Item, len(m.ids))
	for i := range m.ids {
		items[i] = Item{
			ID:        m.ids[i],
			IsDeleted: m.deleted[i],
		}
		if !m.deleted[i] {
			start := i * m.dimension
			end := start + m.dimension
			vec := make([]float32, m.dimension)
			copy(vec, m.vectors[start:end])
			items[i].Vector = vec
		}
	}

	// Reset
	m.ids = make([]uint64, 0, 1024)
	m.vectors = make([]float32, 0, 1024*m.dimension)
	m.deleted = make([]bool, 0, 1024)

	return items
}

// Size returns the number of items in the memtable.
func (m *MemTable) Size() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return len(m.ids)
}

// Search performs a search on the memtable using linear scan.
func (m *MemTable) Search(query []float32, k int, filter func(uint64) bool) []index.SearchResult {
	var res []index.SearchResult
	_ = m.SearchWithBuffer(query, k, filter, &res)
	return res
}

// SearchWithBuffer performs a search on the memtable using linear scan and appends to buf.
func (m *MemTable) SearchWithBuffer(query []float32, k int, filter func(uint64) bool, buf *[]index.SearchResult) error {
	m.mu.RLock()
	defer m.mu.RUnlock()

	// Use pooled heap
	h := m.heapPool.Get().(*resultHeap)
	*h = (*h)[:0]
	defer m.heapPool.Put(h)

	// Linear scan
	// We iterate backwards to find the latest version of each ID
	seen := m.mapPool.Get().(map[uint64]struct{})
	clear(seen)
	defer m.mapPool.Put(seen)

	for i := len(m.ids) - 1; i >= 0; i-- {
		id := m.ids[i]
		if _, ok := seen[id]; ok {
			continue
		}
		seen[id] = struct{}{}

		if m.deleted[i] {
			continue
		}

		if filter != nil && !filter(id) {
			continue
		}

		// Compute distance
		start := i * m.dimension
		end := start + m.dimension
		vec := m.vectors[start:end]
		dist := m.distFunc(query, vec)

		// Push to heap
		if h.Len() < k {
			heap.Push(h, index.SearchResult{ID: id, Distance: dist})
		} else if dist < (*h)[0].Distance {
			heap.Pop(h)
			heap.Push(h, index.SearchResult{ID: id, Distance: dist})
		}
	}

	// Extract from heap (reverse order)
	startLen := len(*buf)
	for h.Len() > 0 {
		*buf = append(*buf, heap.Pop(h).(index.SearchResult))
	}

	// Reverse the appended segment to get nearest first
	res := *buf
	for i, j := startLen, len(res)-1; i < j; i, j = i+1, j-1 {
		res[i], res[j] = res[j], res[i]
	}

	return nil
}

// Items returns a copy of all items in the memtable without clearing it.
func (m *MemTable) Items() []Item {
	m.mu.RLock()
	defer m.mu.RUnlock()

	items := make([]Item, len(m.ids))
	for i := range m.ids {
		items[i] = Item{
			ID:        m.ids[i],
			IsDeleted: m.deleted[i],
		}
		if !m.deleted[i] {
			start := i * m.dimension
			end := start + m.dimension
			vec := make([]float32, m.dimension)
			copy(vec, m.vectors[start:end])
			items[i].Vector = vec
		}
	}
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
