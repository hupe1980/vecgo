// Package vectorstore defines a canonical vector storage interface.
//
// The Flat and HNSW indexes use this interface as their internal (canonical)
// vector memory owner, including snapshot/mmap load paths.
package vectorstore

import "errors"

var (
	// ErrWrongDimension is returned when a vector doesn't match the store dimension.
	ErrWrongDimension = errors.New("wrong vector dimension")
)

// Store is the canonical storage for vectors.
//
// Implementations must treat the configured dimension as authoritative.
// Callers should assume returned slices may alias internal memory unless the
// implementation documents otherwise.
type Store interface {
	Dimension() int
	GetVector(id uint32) ([]float32, bool)
	SetVector(id uint32, v []float32) error
	DeleteVector(id uint32) error
}
