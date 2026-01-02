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
	"errors"
	"sync/atomic"

	"github.com/hupe1980/vecgo/core"
	"github.com/hupe1980/vecgo/vectorstore"
)

// Store is a zero-copy vector store backed by a single contiguous slice.
// It is designed for mmap usage where the entire vector data is mapped as a single block.
type Store struct {
	dim  int
	data atomic.Pointer[[]float32] // Contiguous storage: data[id*dim : (id+1)*dim]
	size int                       // Number of vectors (capacity)
}

// New creates a new zero-copy store.
// Initially empty. Use SetData to provide the backing slice.
func New(dim int) *Store {
	if dim < 0 {
		dim = 0
	}
	s := &Store{dim: dim}
	empty := []float32{}
	s.data.Store(&empty)
	return s
}

// SetData sets the backing slice for the store.
// The slice must be a multiple of dimension.
func (s *Store) SetData(data []float32) {
	if s.dim > 0 && len(data)%s.dim != 0 {
		panic("zerocopy: data length must be multiple of dimension")
	}
	s.data.Store(&data)
	if s.dim > 0 {
		s.size = len(data) / s.dim
	}
}

// Dimension returns the dimension of the vectors.
func (s *Store) Dimension() int { return s.dim }

// GetVector retrieves a vector by ID.
func (s *Store) GetVector(id core.LocalID) ([]float32, bool) {
	data := *s.data.Load()
	idx := int(id) * s.dim
	if idx < 0 || idx+s.dim > len(data) {
		return nil, false
	}
	return data[idx : idx+s.dim], true
}

// SetVector sets a vector by ID.
func (s *Store) SetVector(id core.LocalID, v []float32) error {
	if len(v) != s.dim {
		return vectorstore.ErrWrongDimension
	}
	data := *s.data.Load()
	idx := int(id) * s.dim
	if idx < 0 || idx+s.dim > len(data) {
		return errors.New("zerocopy: id out of bounds")
	}
	// Copy data into the slice
	copy(data[idx:idx+s.dim], v)
	return nil
}

// DeleteVector deletes a vector by ID.
func (s *Store) DeleteVector(id core.LocalID) error {
	// Zero out the vector? Or just ignore?
	// For mmap read-only, we can't really delete.
	// But this store might be used for mutable buffers too.
	// For now, no-op or zeroing.
	return nil
}

var _ vectorstore.Store = (*Store)(nil)
