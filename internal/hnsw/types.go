package hnsw

import (
	"errors"
	"fmt"

	"github.com/hupe1980/vecgo/internal/segment"
	"github.com/hupe1980/vecgo/model"
)

var (
	ErrEmptyVector         = errors.New("vector cannot be empty")
	ErrInvalidK            = errors.New("k must be positive")
	ErrEntryPointDeleted   = errors.New("entry point has been deleted")
	ErrInsufficientVectors = errors.New("not enough vectors for operation")
)

type ErrInvalidDimension struct {
	Dimension int
}

func (e *ErrInvalidDimension) Error() string {
	return fmt.Sprintf("invalid dimension: %d", e.Dimension)
}

type ErrDimensionMismatch struct {
	Expected int
	Actual   int
}

func (e *ErrDimensionMismatch) Error() string {
	return fmt.Sprintf("dimension mismatch: expected %d, got %d", e.Expected, e.Actual)
}

type ErrNodeNotFound struct {
	ID model.RowID
}

func (e *ErrNodeNotFound) Error() string {
	return fmt.Sprintf("node %d not found", e.ID)
}

type SearchResult struct {
	ID       uint32
	Distance float32
}

type SearchOptions struct {
	EFSearch int
	Filter   segment.Filter

	// Selectivity is the estimated filter pass rate (0.0-1.0).
	// Used to choose between predicate-aware (low selectivity) and
	// unfiltered+post-filter (high selectivity) traversal.
	// 0 means unknown/not provided (defaults to predicate-aware).
	Selectivity float64
}

type BatchInsertResult struct {
	IDs    []model.RowID
	Errors []error
}

type LevelStats struct {
	Level          int
	Nodes          int
	Connections    int
	AvgConnections int
}

type Stats struct {
	Options    map[string]string
	Parameters map[string]string
	Storage    map[string]string
	Levels     []LevelStats
}
