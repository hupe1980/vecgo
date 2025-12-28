// Package zerocopy provides a vector store for zero-copy slice references.
//
// This store is designed for mmap loading where vectors are slices that
// point directly into memory-mapped file regions. It does NOT copy vectors
// on get/set operations - callers must treat returned slices as immutable.
//
// For regular in-memory usage, prefer vectorstore/columnar which provides
// better cache locality and SIMD-friendly layout.
package zerocopy

import (
	"sync"

	"github.com/hupe1980/vecgo/vectorstore"
)

// Store holds zero-copy vector slice references.
//
// Vectors stored here may alias external memory (e.g., mmap'd regions).
// The store does not copy vectors on SetVector/GetVector - callers must
// ensure vectors are not mutated after SetVector.
//
// Thread-safe for concurrent reads; writes require external synchronization.
type Store struct {
	dim int

	mu sync.RWMutex
	m  map[uint32][]float32
}

// New creates a new zero-copy vector store.
func New(dim int) *Store {
	if dim < 0 {
		dim = 0
	}
	return &Store{dim: dim, m: make(map[uint32][]float32)}
}

// Dimension returns the vector dimensionality.
func (s *Store) Dimension() int { return s.dim }

// GetVector returns the vector at the given ID.
// The returned slice may alias internal or external memory; do not modify.
func (s *Store) GetVector(id uint32) ([]float32, bool) {
	s.mu.RLock()
	v, ok := s.m[id]
	s.mu.RUnlock()
	return v, ok
}

// SetVector stores a vector reference (no copy).
// The caller must ensure v is not mutated after this call.
func (s *Store) SetVector(id uint32, v []float32) error {
	if s.dim != 0 && len(v) != s.dim {
		return vectorstore.ErrWrongDimension
	}
	s.mu.Lock()
	s.m[id] = v
	s.mu.Unlock()
	return nil
}

// DeleteVector removes a vector from the store.
func (s *Store) DeleteVector(id uint32) error {
	s.mu.Lock()
	delete(s.m, id)
	s.mu.Unlock()
	return nil
}

var _ vectorstore.Store = (*Store)(nil)
