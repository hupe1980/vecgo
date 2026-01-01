// Package vectorstore defines a canonical vector storage interface.
//
// The Flat and HNSW indexes use this interface as their internal (canonical)
// vector memory owner, including snapshot/mmap load paths.
package vectorstore

import (
	"errors"

	"github.com/hupe1980/vecgo/core"
)

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
	GetVector(id core.LocalID) ([]float32, bool)
	SetVector(id core.LocalID, v []float32) error
	DeleteVector(id core.LocalID) error
}
