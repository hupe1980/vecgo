// Package vectorstore defines a canonical vector storage interface.
//
// The Flat and HNSW indexes use this interface as their internal (canonical)
// vector memory owner, including snapshot/mmap load paths.
package vectorstore

import (
	"errors"

	"github.com/hupe1980/vecgo/distance"
	"github.com/hupe1980/vecgo/model"
)

var (
	// ErrWrongDimension is returned when a vector doesn't match the store dimension.
	ErrWrongDimension = errors.New("wrong vector dimension")
)

// VectorStore is the canonical storage interface for vectors.
//
// Implementations must treat the configured dimension as authoritative.
// Callers should assume returned slices may alias internal memory unless the
// implementation documents otherwise.
type VectorStore interface {
	Dimension() int
	GetVector(id model.RowID) ([]float32, bool)
	ComputeDistance(id model.RowID, query []float32, metric distance.Metric) (float32, bool)
	SetVector(id model.RowID, v []float32) error
	DeleteVector(id model.RowID) error
}
