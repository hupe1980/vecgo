package engine

import (
	"maps"
	"sync"
)

// MapStore is an in-memory implementation of Store using a Go map.
// It's suitable for datasets that fit in memory and provides fast O(1) access.
type MapStore[T any] struct {
	mu   sync.RWMutex
	data map[uint32]T
}

// NewMapStore creates a new in-memory map-based store.
func NewMapStore[T any]() *MapStore[T] {
	return &MapStore[T]{
		data: make(map[uint32]T),
	}
}

// Get retrieves the data associated with the given ID.
func (m *MapStore[T]) Get(id uint32) (T, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	v, ok := m.data[id]
	return v, ok
}

// Set stores data associated with the given ID.
func (m *MapStore[T]) Set(id uint32, data T) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.data[id] = data
	return nil
}

// Delete removes the data associated with the given ID.
func (m *MapStore[T]) Delete(id uint32) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if _, ok := m.data[id]; !ok {
		return ErrNotFound
	}

	delete(m.data, id)
	return nil
}

// BatchGet retrieves data for multiple IDs in a single operation.
func (m *MapStore[T]) BatchGet(ids []uint32) (map[uint32]T, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	result := make(map[uint32]T, len(ids))
	for _, id := range ids {
		if data, ok := m.data[id]; ok {
			result[id] = data
		}
	}

	return result, nil
}

// BatchSet stores multiple id -> data pairs in a single operation.
func (m *MapStore[T]) BatchSet(items map[uint32]T) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	for id, data := range items {
		m.data[id] = data
	}

	return nil
}

// BatchDelete removes data for multiple IDs in a single operation.
func (m *MapStore[T]) BatchDelete(ids []uint32) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	for _, id := range ids {
		delete(m.data, id)
	}

	return nil
}

// Len returns the number of items currently stored.
func (m *MapStore[T]) Len() int {
	m.mu.RLock()
	defer m.mu.RUnlock()

	return len(m.data)
}

// Clear removes all items from the store.
func (m *MapStore[T]) Clear() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.data = make(map[uint32]T)
	return nil
}

// ToMap returns a copy of all data as a map (for serialization).
func (m *MapStore[T]) ToMap() map[uint32]T {
	m.mu.RLock()
	defer m.mu.RUnlock()

	result := make(map[uint32]T, len(m.data))
	maps.Copy(result, m.data)

	return result
}
