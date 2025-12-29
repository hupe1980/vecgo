package engine

import (
	"context"
	"fmt"
	"io"
	"iter"
	"math"
	"sync/atomic"

	"github.com/hupe1980/vecgo/index"
	"github.com/hupe1980/vecgo/metadata"
)

// ValidationLimits defines bounds for input validation.
// These prevent crashes from malformed input and DoS attacks via resource exhaustion.
type ValidationLimits struct {
	MaxDimension     int // Max vector dimension (default: 65536)
	MaxVectors       int // Max total vectors (default: 100M)
	MaxK             int // Max search results (default: 10000)
	MaxMetadataBytes int // Max metadata size per vector (default: 64KB)
	MaxBatchSize     int // Max items per batch (default: 10000)
}

// DefaultLimits returns safe production defaults.
func DefaultLimits() ValidationLimits {
	return ValidationLimits{
		MaxDimension:     65536,
		MaxVectors:       100_000_000,
		MaxK:             10000,
		MaxMetadataBytes: 64 * 1024,
		MaxBatchSize:     10000,
	}
}

// ValidatedCoordinator wraps a Coordinator with input validation.
// This prevents crashes from nil vectors, NaN values, dimension mismatches,
// and DoS attacks via oversized metadata or excessive k values.
type ValidatedCoordinator[T any] struct {
	inner     Coordinator[T]
	dimension int
	limits    ValidationLimits
	count     atomic.Int64 // track vector count for limit enforcement
}

// Compile-time interface check
var _ Coordinator[any] = (*ValidatedCoordinator[any])(nil)

// WithValidation wraps a coordinator with input validation.
// This should ALWAYS be applied in production.
//
// Example:
//
//	coord, _ := engine.New(idx, store, meta, nil, nil)
//	validated := engine.WithValidation(coord, 128, engine.DefaultLimits())
func WithValidation[T any](c Coordinator[T], dimension int, limits ValidationLimits) Coordinator[T] {
	return &ValidatedCoordinator[T]{
		inner:     c,
		dimension: dimension,
		limits:    limits,
	}
}

// Insert validates input and delegates to the inner coordinator.
func (v *ValidatedCoordinator[T]) Insert(ctx context.Context, vector []float32, data T, meta metadata.Metadata) (uint32, error) {
	if err := v.validateVector(vector); err != nil {
		return 0, err
	}
	if err := v.validateMetadata(meta); err != nil {
		return 0, err
	}
	if v.count.Load() >= int64(v.limits.MaxVectors) {
		return 0, fmt.Errorf("vector limit exceeded: %d >= %d", v.count.Load(), v.limits.MaxVectors)
	}

	id, err := v.inner.Insert(ctx, vector, data, meta)
	if err == nil {
		v.count.Add(1)
	}
	return id, err
}

// BatchInsert validates all inputs and delegates to the inner coordinator.
func (v *ValidatedCoordinator[T]) BatchInsert(ctx context.Context, vectors [][]float32, data []T, meta []metadata.Metadata) ([]uint32, error) {
	if len(vectors) > v.limits.MaxBatchSize {
		return nil, fmt.Errorf("batch size %d exceeds limit %d", len(vectors), v.limits.MaxBatchSize)
	}
	if len(vectors) != len(data) {
		return nil, fmt.Errorf("vectors length %d != data length %d", len(vectors), len(data))
	}
	if len(vectors) != len(meta) {
		return nil, fmt.Errorf("vectors length %d != metadata length %d", len(vectors), len(meta))
	}
	if v.count.Load()+int64(len(vectors)) > int64(v.limits.MaxVectors) {
		return nil, fmt.Errorf("batch would exceed vector limit: %d + %d > %d", v.count.Load(), len(vectors), v.limits.MaxVectors)
	}

	for i, vec := range vectors {
		if err := v.validateVector(vec); err != nil {
			return nil, fmt.Errorf("vector[%d]: %w", i, err)
		}
	}
	for i, m := range meta {
		if err := v.validateMetadata(m); err != nil {
			return nil, fmt.Errorf("metadata[%d]: %w", i, err)
		}
	}

	ids, err := v.inner.BatchInsert(ctx, vectors, data, meta)
	if err == nil {
		v.count.Add(int64(len(ids)))
	}
	return ids, err
}

// Update validates input and delegates to the inner coordinator.
func (v *ValidatedCoordinator[T]) Update(ctx context.Context, id uint32, vector []float32, data T, meta metadata.Metadata) error {
	if err := v.validateVector(vector); err != nil {
		return err
	}
	if err := v.validateMetadata(meta); err != nil {
		return err
	}
	return v.inner.Update(ctx, id, vector, data, meta)
}

// Delete delegates to the inner coordinator and decrements count on success.
func (v *ValidatedCoordinator[T]) Delete(ctx context.Context, id uint32) error {
	err := v.inner.Delete(ctx, id)
	if err == nil {
		v.count.Add(-1)
	}
	return err
}

// KNNSearch validates query and k, then delegates to the inner coordinator.
func (v *ValidatedCoordinator[T]) KNNSearch(ctx context.Context, query []float32, k int, opts *index.SearchOptions) ([]index.SearchResult, error) {
	if err := v.validateVector(query); err != nil {
		return nil, err
	}
	if k <= 0 {
		return nil, index.ErrInvalidK
	}
	if k > v.limits.MaxK {
		return nil, fmt.Errorf("k=%d exceeds limit %d", k, v.limits.MaxK)
	}
	return v.inner.KNNSearch(ctx, query, k, opts)
}

// BruteSearch validates query and k, then delegates to the inner coordinator.
func (v *ValidatedCoordinator[T]) BruteSearch(ctx context.Context, query []float32, k int, filter func(id uint32) bool) ([]index.SearchResult, error) {
	if err := v.validateVector(query); err != nil {
		return nil, err
	}
	if k <= 0 {
		return nil, index.ErrInvalidK
	}
	if k > v.limits.MaxK {
		return nil, fmt.Errorf("k=%d exceeds limit %d", k, v.limits.MaxK)
	}
	return v.inner.BruteSearch(ctx, query, k, filter)
}

// Get delegates to the inner coordinator.
func (v *ValidatedCoordinator[T]) Get(id uint32) (T, bool) {
	return v.inner.Get(id)
}

// GetMetadata delegates to the inner coordinator.
func (v *ValidatedCoordinator[T]) GetMetadata(id uint32) (metadata.Metadata, bool) {
	return v.inner.GetMetadata(id)
}

// validateVector checks for nil, dimension mismatch, and invalid values (NaN/Inf).
func (v *ValidatedCoordinator[T]) validateVector(vec []float32) error {
	if vec == nil {
		return fmt.Errorf("vector is nil")
	}
	if len(vec) != v.dimension {
		return &index.ErrDimensionMismatch{Expected: v.dimension, Actual: len(vec)}
	}
	if len(vec) > v.limits.MaxDimension {
		return fmt.Errorf("dimension %d exceeds limit %d", len(vec), v.limits.MaxDimension)
	}
	for i, val := range vec {
		if math.IsNaN(float64(val)) {
			return fmt.Errorf("vector[%d] contains NaN", i)
		}
		if math.IsInf(float64(val), 0) {
			return fmt.Errorf("vector[%d] contains Inf", i)
		}
	}
	return nil
}

// validateMetadata checks size limits.
func (v *ValidatedCoordinator[T]) validateMetadata(meta metadata.Metadata) error {
	if meta == nil {
		return nil // nil metadata is valid
	}
	// Estimate size (rough but fast)
	size := 0
	for k, val := range meta {
		size += len(k)
		switch val.Kind {
		case metadata.KindString:
			size += len(val.StringValue())
		case metadata.KindArray:
			// Rough estimate for arrays
			size += len(val.A) * 16
		default:
			size += 8 // assume 8 bytes for numbers, bool, null
		}
	}
	if size > v.limits.MaxMetadataBytes {
		return fmt.Errorf("metadata size ~%d exceeds limit %d", size, v.limits.MaxMetadataBytes)
	}
	return nil
}

// Count returns the current number of vectors tracked by the validator.
// This is an approximation that may drift if errors occur during insert/delete.
func (v *ValidatedCoordinator[T]) Count() int64 {
	return v.count.Load()
}

// SetCount sets the vector count, useful when restoring from snapshot.
func (v *ValidatedCoordinator[T]) SetCount(n int64) {
	v.count.Store(n)
}

// HybridSearch validates input and delegates to the inner coordinator.
func (v *ValidatedCoordinator[T]) HybridSearch(ctx context.Context, query []float32, k int, opts *HybridSearchOptions) ([]index.SearchResult, error) {
	if err := v.validateVector(query); err != nil {
		return nil, err
	}
	if k <= 0 {
		return nil, index.ErrInvalidK
	}
	if k > v.limits.MaxK {
		return nil, fmt.Errorf("k %d exceeds limit %d", k, v.limits.MaxK)
	}
	return v.inner.HybridSearch(ctx, query, k, opts)
}

// KNNSearchStream validates input and delegates to the inner coordinator.
func (v *ValidatedCoordinator[T]) KNNSearchStream(ctx context.Context, query []float32, k int, opts *index.SearchOptions) iter.Seq2[index.SearchResult, error] {
	// We can't easily return an error from the iterator setup, so we return an iterator that yields an error immediately if validation fails.
	if err := v.validateVector(query); err != nil {
		return func(yield func(index.SearchResult, error) bool) {
			yield(index.SearchResult{}, err)
		}
	}
	if k <= 0 {
		return func(yield func(index.SearchResult, error) bool) {
			yield(index.SearchResult{}, index.ErrInvalidK)
		}
	}
	if k > v.limits.MaxK {
		return func(yield func(index.SearchResult, error) bool) {
			yield(index.SearchResult{}, fmt.Errorf("k %d exceeds limit %d", k, v.limits.MaxK))
		}
	}
	return v.inner.KNNSearchStream(ctx, query, k, opts)
}

// EnableProductQuantization delegates to the inner coordinator.
func (v *ValidatedCoordinator[T]) EnableProductQuantization(cfg index.ProductQuantizationConfig) error {
	return v.inner.EnableProductQuantization(cfg)
}

// DisableProductQuantization delegates to the inner coordinator.
func (v *ValidatedCoordinator[T]) DisableProductQuantization() {
	v.inner.DisableProductQuantization()
}

// SaveToWriter delegates to the inner coordinator.
func (v *ValidatedCoordinator[T]) SaveToWriter(w io.Writer) error {
	return v.inner.SaveToWriter(w)
}

// SaveToFile delegates to the inner coordinator.
func (v *ValidatedCoordinator[T]) SaveToFile(path string) error {
	return v.inner.SaveToFile(path)
}

// RecoverFromWAL delegates to the inner coordinator.
func (v *ValidatedCoordinator[T]) RecoverFromWAL(ctx context.Context) error {
	return v.inner.RecoverFromWAL(ctx)
}

// Stats delegates to the inner coordinator.
func (v *ValidatedCoordinator[T]) Stats() index.Stats {
	return v.inner.Stats()
}

// Checkpoint delegates to the inner coordinator.
func (v *ValidatedCoordinator[T]) Checkpoint() error {
	return v.inner.Checkpoint()
}

// Close closes the wrapped coordinator.
func (v *ValidatedCoordinator[T]) Close() error {
	return v.inner.Close()
}
