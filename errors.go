package vecgo

import (
	"errors"
	"fmt"

	"github.com/hupe1980/vecgo/engine"
	"github.com/hupe1980/vecgo/index"
)

var (
	// ErrInvalidK is returned when k is not positive.
	ErrInvalidK = errors.New("k must be positive")
)

// ErrDimensionMismatch indicates a vector/query dimensionality mismatch.
//
// The original underlying error (if any) can be accessed via errors.Unwrap.
type ErrDimensionMismatch struct {
	Expected int
	Actual   int
	cause    error
}

func (e *ErrDimensionMismatch) Error() string {
	return fmt.Sprintf("dimension mismatch: expected %d, got %d", e.Expected, e.Actual)
}

func (e *ErrDimensionMismatch) Unwrap() error { return e.cause }

// ErrInvalidDimension indicates an invalid configured dimension.
//
// The original underlying error (if any) can be accessed via errors.Unwrap.
type ErrInvalidDimension struct {
	Dimension int
	cause     error
}

func (e *ErrInvalidDimension) Error() string {
	return fmt.Sprintf("invalid dimension: %d", e.Dimension)
}

func (e *ErrInvalidDimension) Unwrap() error { return e.cause }

// ErrInvalidDistanceType indicates an unsupported distance type.
//
// The original underlying error (if any) can be accessed via errors.Unwrap.
type ErrInvalidDistanceType struct {
	DistanceType index.DistanceType
	cause        error
}

func (e *ErrInvalidDistanceType) Error() string {
	return fmt.Sprintf("invalid distance type: %d", e.DistanceType)
}

func (e *ErrInvalidDistanceType) Unwrap() error { return e.cause }

func translateError(err error) error {
	if err == nil {
		return nil
	}

	// Not found unification.
	if errors.Is(err, engine.ErrNotFound) {
		return fmt.Errorf("%w: %w", ErrNotFound, err)
	}
	var enf *index.ErrNodeNotFound
	if errors.As(err, &enf) {
		return fmt.Errorf("%w: %w", ErrNotFound, err)
	}
	var end *index.ErrNodeDeleted
	if errors.As(err, &end) {
		return fmt.Errorf("%w: %w", ErrNotFound, err)
	}

	// Dimension and argument normalization.
	var dm *index.ErrDimensionMismatch
	if errors.As(err, &dm) {
		return &ErrDimensionMismatch{Expected: dm.Expected, Actual: dm.Actual, cause: err}
	}
	var id *index.ErrInvalidDimension
	if errors.As(err, &id) {
		return &ErrInvalidDimension{Dimension: id.Dimension, cause: err}
	}
	var dt *index.ErrInvalidDistanceType
	if errors.As(err, &dt) {
		return &ErrInvalidDistanceType{DistanceType: dt.DistanceType, cause: err}
	}
	if errors.Is(err, index.ErrInvalidK) {
		return fmt.Errorf("%w: %w", ErrInvalidK, err)
	}

	return err
}
