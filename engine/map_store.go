package engine

import (
	"iter"
	"sync"

	"github.com/hupe1980/vecgo/core"
)

// MapStore is a simple in-memory implementation of the Store interface using a Go map.
// It is thread-safe.
type MapStore[T any] struct {
	mu   sync.RWMutex
	data map[core.LocalID]T
}

// NewMapStore creates a new MapStore.
func NewMapStore[T any](capacity ...int) *MapStore[T] {
	initialCap := 0
	if len(capacity) > 0 {
		initialCap = capacity[0]
	}
	return &MapStore[T]{
		data: make(map[core.LocalID]T, initialCap),
	}
}

// Get retrieves the data associated with the given ID.
func (s *MapStore[T]) Get(id core.LocalID) (T, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	val, ok := s.data[id]
	return val, ok
}

// Set stores data associated with the given ID.
func (s *MapStore[T]) Set(id core.LocalID, data T) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.data[id] = data
	return nil
}

// Delete removes the data associated with the given ID.
func (s *MapStore[T]) Delete(id core.LocalID) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.data[id]; !ok {
		return ErrNotFound
	}
	delete(s.data, id)
	return nil
}

// BatchGet retrieves data for multiple IDs.
func (s *MapStore[T]) BatchGet(ids []core.LocalID) (map[core.LocalID]T, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	result := make(map[core.LocalID]T, len(ids))
	for _, id := range ids {
		if val, ok := s.data[id]; ok {
			result[id] = val
		}
	}
	return result, nil
}

// BatchSet stores multiple id -> data pairs.
func (s *MapStore[T]) BatchSet(items map[core.LocalID]T) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	for id, data := range items {
		s.data[id] = data
	}
	return nil
}

// BatchDelete removes data for multiple IDs.
func (s *MapStore[T]) BatchDelete(ids []core.LocalID) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, id := range ids {
		delete(s.data, id)
	}
	return nil
}

// Len returns the number of items currently stored.
func (s *MapStore[T]) Len() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.data)
}

// Clear removes all items from the store.
func (s *MapStore[T]) Clear() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.data = make(map[core.LocalID]T)
	return nil
}

// ToMap returns a copy of all data as a map.
func (s *MapStore[T]) ToMap() map[core.LocalID]T {
	s.mu.RLock()
	defer s.mu.RUnlock()
	result := make(map[core.LocalID]T, len(s.data))
	for k, v := range s.data {
		result[k] = v
	}
	return result
}

// All returns an iterator over all items in the store.
func (s *MapStore[T]) All() iter.Seq2[core.LocalID, T] {
	return func(yield func(core.LocalID, T) bool) {
		s.mu.RLock()
		defer s.mu.RUnlock()
		for k, v := range s.data {
			if !yield(k, v) {
				return
			}
		}
	}
}
