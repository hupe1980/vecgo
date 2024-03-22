// Package index provides interfaces and types for vector search indexes.
package index

import (
	"encoding/gob"
	"fmt"

	"github.com/hupe1980/vecgo/metric"
)

// ErrDimensionMismatch is a named error type for dimension mismatch
type ErrDimensionMismatch struct {
	Expected int // Expected dimensions
	Actual   int // Actual dimensions
}

// Error returns the error message for dimension mismatch
func (e *ErrDimensionMismatch) Error() string {
	return fmt.Sprintf("dimension mismatch: expected %d, got %d", e.Expected, e.Actual)
}

// DistanceFunc represents a function for calculating the distance between two vectors
type DistanceFunc func(v1, v2 []float32) (float32, error)

// DistanceType represents the type of distance function used for calculating distances between vectors.
type DistanceType int

// Constants representing different types of distance functions.
const (
	DistanceTypeSquaredL2 DistanceType = iota
	DistanceTypeCosineSimilarity
)

// NewDistanceFunc returns a distance function based on the specified distance type.
func NewDistanceFunc(distanceType DistanceType) DistanceFunc {
	switch distanceType {
	case DistanceTypeSquaredL2:
		return metric.SquaredL2
	case DistanceTypeCosineSimilarity:
		return metric.CosineSimilarity
	default:
		return nil
	}
}

// String returns a string representation of the DistanceType.
func (dt DistanceType) String() string {
	switch dt {
	case DistanceTypeSquaredL2:
		return "SquaredL2"
	case DistanceTypeCosineSimilarity:
		return "CosineSimilarity"
	default:
		return "Unknown"
	}
}

// SearchResult represents a search result.
type SearchResult struct {
	// ID is the identifier of the search result.
	ID uint32

	// Distance is the distance between the query vector and the result vector.
	Distance float32
}

// Index represents an index for vector search
type Index interface {
	gob.GobEncoder
	gob.GobDecoder

	// Insert adds a vector to the index
	Insert(v []float32) (uint32, error)

	// KNNSearch performs a K-nearest neighbor search
	KNNSearch(q []float32, k int, efSearch int, filter func(id uint32) bool) ([]SearchResult, error)

	// BruteSearch performs a brute-force search
	BruteSearch(query []float32, k int, filter func(id uint32) bool) ([]SearchResult, error)

	// Stats prints statistics about the index
	Stats()
}
