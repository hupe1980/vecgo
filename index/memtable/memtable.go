package memtable

import (
	"sync"

	"github.com/hupe1980/vecgo/core"
	"github.com/hupe1980/vecgo/index"
	"github.com/hupe1980/vecgo/internal/mem"
	"github.com/hupe1980/vecgo/searcher"
)

// Item represents a vector in the memtable.
type Item struct {
	ID        core.LocalID
	Vector    []float32
	IsDeleted bool
}

// MemTable is a thread-safe linear buffer for vectors.
// It serves as the L0 (Level 0) store in the LSM tree architecture.
// It uses a Struct of Arrays (SoA) layout for better cache locality.
type MemTable struct {
	mu        sync.RWMutex
	ids       []core.LocalID
	vectors   []float32 // Flattened vectors
	deleted   []bool
	idToIndex map[core.LocalID]int // Map for O(1) lookups
	distFunc  index.DistanceFunc
	dimension int
	zeroVec   []float32 // Pre-allocated zero vector for deletes
	heapPool  sync.Pool
}

// New creates a new MemTable.
func New(dimension int, distFunc index.DistanceFunc) *MemTable {
	m := &MemTable{
		ids:       make([]core.LocalID, 0, 1024),
		vectors:   mem.AllocAlignedFloat32(1024 * dimension)[:0],
		deleted:   make([]bool, 0, 1024),
		idToIndex: make(map[core.LocalID]int, 1024),
		distFunc:  distFunc,
		dimension: dimension,
		zeroVec:   make([]float32, dimension),
		heapPool: sync.Pool{
			New: func() any {
				return searcher.NewPriorityQueue(true)
			},
		},
	}
	return m
}

// Insert adds or updates a vector in the memtable.
func (m *MemTable) Insert(id core.LocalID, vector []float32) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if idx, ok := m.idToIndex[id]; ok {
		// Update existing
		copy(m.vectors[idx*m.dimension:], vector)
		m.deleted[idx] = false
		return
	}

	// Append new
	idx := len(m.ids)
	m.ids = append(m.ids, id)
	m.vectors = append(m.vectors, vector...)
	m.deleted = append(m.deleted, false)
	m.idToIndex[id] = idx
}

// Delete marks a vector as deleted (tombstone).
func (m *MemTable) Delete(id core.LocalID) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if idx, ok := m.idToIndex[id]; ok {
		m.deleted[idx] = true
		// Clear vector data to avoid holding references/memory if needed
		copy(m.vectors[idx*m.dimension:], m.zeroVec)
		return
	}

	// If not found, append a tombstone
	idx := len(m.ids)
	m.ids = append(m.ids, id)
	m.vectors = append(m.vectors, m.zeroVec...)
	m.deleted = append(m.deleted, true)
	m.idToIndex[id] = idx
}

// Get retrieves a vector from the memtable.
// Returns (vector, found, isDeleted).
func (m *MemTable) Get(id core.LocalID) ([]float32, bool, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if idx, ok := m.idToIndex[id]; ok {
		if m.deleted[idx] {
			return nil, true, true
		}
		// Return a copy to ensure safety
		vec := make([]float32, m.dimension)
		copy(vec, m.vectors[idx*m.dimension:(idx+1)*m.dimension])
		return vec, true, false
	}
	return nil, false, false
}

// Contains checks if an ID exists in the memtable.
// Returns (found, isDeleted).
func (m *MemTable) Contains(id core.LocalID) (bool, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if idx, ok := m.idToIndex[id]; ok {
		return true, m.deleted[idx]
	}
	return false, false
}

// Size returns the number of items in the memtable.
func (m *MemTable) Size() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return len(m.ids)
}

// Items returns all items in the memtable.
func (m *MemTable) Items() []Item {
	m.mu.RLock()
	defer m.mu.RUnlock()

	items := make([]Item, len(m.ids))
	for i := range m.ids {
		vec := make([]float32, m.dimension)
		copy(vec, m.vectors[i*m.dimension:(i+1)*m.dimension])
		items[i] = Item{
			ID:        m.ids[i],
			Vector:    vec,
			IsDeleted: m.deleted[i],
		}
	}
	return items
}

// Flush returns all items and resets the memtable.
func (m *MemTable) Flush() []Item {
	m.mu.Lock()
	defer m.mu.Unlock()

	items := make([]Item, len(m.ids))
	for i := range m.ids {
		var vec []float32
		if !m.deleted[i] {
			vec = make([]float32, m.dimension)
			copy(vec, m.vectors[i*m.dimension:(i+1)*m.dimension])
		}
		items[i] = Item{
			ID:        m.ids[i],
			Vector:    vec,
			IsDeleted: m.deleted[i],
		}
	}

	// Reset
	m.ids = m.ids[:0]
	m.vectors = m.vectors[:0]
	m.deleted = m.deleted[:0]
	// Re-allocate map to avoid memory leak if it was huge, or just clear it?
	// clear(m.idToIndex) is Go 1.21+.
	// For now, let's make a new one to be safe and release old memory.
	m.idToIndex = make(map[core.LocalID]int, 1024)

	return items
}

// Search performs a brute-force search on the memtable.
func (m *MemTable) Search(query []float32, k int, filter func(core.LocalID) bool) []index.SearchResult {
	var results []index.SearchResult
	m.SearchWithBuffer(query, k, filter, &results)
	return results
}

// SearchWithBuffer performs a brute-force search and appends to buffer.
func (m *MemTable) SearchWithBuffer(query []float32, k int, filter func(core.LocalID) bool, buf *[]index.SearchResult) error {
	m.mu.RLock()
	defer m.mu.RUnlock()

	pq := m.heapPool.Get().(*searcher.PriorityQueue)
	pq.Reset()
	defer m.heapPool.Put(pq)

	// Scan all items
	for i, id := range m.ids {
		if m.deleted[i] {
			continue
		}
		if filter != nil && !filter(id) {
			continue
		}

		vec := m.vectors[i*m.dimension : (i+1)*m.dimension]
		dist := m.distFunc(query, vec)

		pq.PushItemBounded(searcher.PriorityQueueItem{
			Node:     id,
			Distance: dist,
		}, k)
	}

	// Extract results
	n := pq.Len()
	startIdx := len(*buf)
	// Ensure capacity
	if cap(*buf) < startIdx+n {
		newBuf := make([]index.SearchResult, len(*buf), startIdx+n)
		copy(newBuf, *buf)
		*buf = newBuf
	}
	*buf = (*buf)[:startIdx+n]

	// Pop in reverse order (worst to best)
	for i := n - 1; i >= 0; i-- {
		item, _ := pq.PopItem()
		(*buf)[startIdx+i] = index.SearchResult{
			ID:       uint32(item.Node),
			Distance: item.Distance,
		}
	}

	return nil
}
