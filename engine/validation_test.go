package engine

import (
	"context"
	"math"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/hupe1980/vecgo/index"
	"github.com/hupe1980/vecgo/metadata"
)

// mockCoordinator implements Coordinator for testing validation layer
type mockCoordinator[T any] struct {
	insertCount int
	deleteCount int
}

func (m *mockCoordinator[T]) Insert(ctx context.Context, vector []float32, data T, meta metadata.Metadata) (uint32, error) {
	m.insertCount++
	return uint32(m.insertCount), nil
}

func (m *mockCoordinator[T]) BatchInsert(ctx context.Context, vectors [][]float32, data []T, meta []metadata.Metadata) ([]uint32, error) {
	ids := make([]uint32, len(vectors))
	for i := range vectors {
		m.insertCount++
		ids[i] = uint32(m.insertCount)
	}
	return ids, nil
}

func (m *mockCoordinator[T]) Update(ctx context.Context, id uint32, vector []float32, data T, meta metadata.Metadata) error {
	return nil
}

func (m *mockCoordinator[T]) Delete(ctx context.Context, id uint32) error {
	m.deleteCount++
	return nil
}

func (m *mockCoordinator[T]) Get(id uint32) (T, bool) {
	var zero T
	return zero, true
}

func (m *mockCoordinator[T]) GetMetadata(id uint32) (metadata.Metadata, bool) {
	return nil, false
}

func (m *mockCoordinator[T]) KNNSearch(ctx context.Context, query []float32, k int, opts *index.SearchOptions) ([]index.SearchResult, error) {
	return []index.SearchResult{{ID: 1, Distance: 0.1}}, nil
}

func (m *mockCoordinator[T]) BruteSearch(ctx context.Context, query []float32, k int, filter func(id uint32) bool) ([]index.SearchResult, error) {
	return []index.SearchResult{{ID: 1, Distance: 0.1}}, nil
}

func TestValidation_NilVector(t *testing.T) {
	mock := &mockCoordinator[string]{}
	coord := WithValidation(mock, 128, DefaultLimits())

	ctx := context.Background()
	_, err := coord.Insert(ctx, nil, "data", nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "vector is nil")
}

func TestValidation_DimensionMismatch(t *testing.T) {
	mock := &mockCoordinator[string]{}
	coord := WithValidation(mock, 128, DefaultLimits())

	ctx := context.Background()
	_, err := coord.Insert(ctx, make([]float32, 64), "data", nil)
	require.Error(t, err)
	var dimErr *index.ErrDimensionMismatch
	assert.ErrorAs(t, err, &dimErr)
	assert.Equal(t, 128, dimErr.Expected)
	assert.Equal(t, 64, dimErr.Actual)
}

func TestValidation_NaNValue(t *testing.T) {
	mock := &mockCoordinator[string]{}
	coord := WithValidation(mock, 4, DefaultLimits())

	ctx := context.Background()
	vec := []float32{1.0, 2.0, float32(math.NaN()), 4.0}
	_, err := coord.Insert(ctx, vec, "data", nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "vector[2] contains NaN")
}

func TestValidation_InfValue(t *testing.T) {
	mock := &mockCoordinator[string]{}
	coord := WithValidation(mock, 4, DefaultLimits())

	ctx := context.Background()
	vec := []float32{1.0, float32(math.Inf(1)), 3.0, 4.0}
	_, err := coord.Insert(ctx, vec, "data", nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "vector[1] contains Inf")
}

func TestValidation_ValidVector(t *testing.T) {
	mock := &mockCoordinator[string]{}
	coord := WithValidation(mock, 4, DefaultLimits())

	ctx := context.Background()
	vec := []float32{1.0, 2.0, 3.0, 4.0}
	id, err := coord.Insert(ctx, vec, "data", nil)
	require.NoError(t, err)
	assert.Equal(t, uint32(1), id)
}

func TestValidation_MaxK(t *testing.T) {
	mock := &mockCoordinator[string]{}
	limits := DefaultLimits()
	limits.MaxK = 100
	coord := WithValidation(mock, 4, limits)

	ctx := context.Background()
	query := []float32{1.0, 2.0, 3.0, 4.0}

	// Valid k
	_, err := coord.KNNSearch(ctx, query, 50, nil)
	require.NoError(t, err)

	// k exceeds limit
	_, err = coord.KNNSearch(ctx, query, 200, nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "k=200 exceeds limit 100")
}

func TestValidation_MaxBatchSize(t *testing.T) {
	mock := &mockCoordinator[string]{}
	limits := DefaultLimits()
	limits.MaxBatchSize = 10
	coord := WithValidation(mock, 4, limits)

	ctx := context.Background()
	vectors := make([][]float32, 20)
	data := make([]string, 20)
	meta := make([]metadata.Metadata, 20)
	for i := range vectors {
		vectors[i] = []float32{1, 2, 3, 4}
		data[i] = "data"
	}

	_, err := coord.BatchInsert(ctx, vectors, data, meta)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "batch size 20 exceeds limit 10")
}

func TestValidation_MaxVectors(t *testing.T) {
	mock := &mockCoordinator[string]{}
	limits := DefaultLimits()
	limits.MaxVectors = 5
	coord := WithValidation(mock, 4, limits)

	ctx := context.Background()
	vec := []float32{1, 2, 3, 4}

	// Insert up to limit
	for i := 0; i < 5; i++ {
		_, err := coord.Insert(ctx, vec, "data", nil)
		require.NoError(t, err)
	}

	// Exceed limit
	_, err := coord.Insert(ctx, vec, "data", nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "vector limit exceeded")
}

func TestValidation_MetadataSize(t *testing.T) {
	mock := &mockCoordinator[string]{}
	limits := DefaultLimits()
	limits.MaxMetadataBytes = 100
	coord := WithValidation(mock, 4, limits)

	ctx := context.Background()
	vec := []float32{1, 2, 3, 4}

	// Small metadata - OK
	_, err := coord.Insert(ctx, vec, "data", metadata.Metadata{"key": metadata.String("value")})
	require.NoError(t, err)

	// Large metadata - error (create a string large enough to exceed 100 bytes)
	largeString := strings.Repeat("x", 200)
	_, err = coord.Insert(ctx, vec, "data", metadata.Metadata{"key": metadata.String(largeString)})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "metadata size")
}

func TestValidation_BatchVectorError(t *testing.T) {
	mock := &mockCoordinator[string]{}
	coord := WithValidation(mock, 4, DefaultLimits())

	ctx := context.Background()
	vectors := [][]float32{
		{1, 2, 3, 4},
		{1, 2, float32(math.NaN()), 4}, // NaN in second vector
		{1, 2, 3, 4},
	}
	data := []string{"a", "b", "c"}
	meta := []metadata.Metadata{nil, nil, nil}

	_, err := coord.BatchInsert(ctx, vectors, data, meta)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "vector[1]")
	assert.Contains(t, err.Error(), "NaN")
}

func TestValidation_CountTracking(t *testing.T) {
	mock := &mockCoordinator[string]{}
	validated := WithValidation(mock, 4, DefaultLimits())
	coord := validated.(*ValidatedCoordinator[string])

	ctx := context.Background()
	vec := []float32{1, 2, 3, 4}

	// Insert increments count
	_, _ = coord.Insert(ctx, vec, "data", nil)
	assert.Equal(t, int64(1), coord.Count())

	// Batch insert increments count
	vectors := [][]float32{vec, vec, vec}
	data := []string{"a", "b", "c"}
	meta := []metadata.Metadata{nil, nil, nil}
	_, _ = coord.BatchInsert(ctx, vectors, data, meta)
	assert.Equal(t, int64(4), coord.Count())

	// Delete decrements count
	_ = coord.Delete(ctx, 1)
	assert.Equal(t, int64(3), coord.Count())
}

func BenchmarkValidation_Vector(b *testing.B) {
	mock := &mockCoordinator[string]{}
	coord := WithValidation(mock, 128, DefaultLimits())

	ctx := context.Background()
	vec := make([]float32, 128)
	for i := range vec {
		vec[i] = float32(i)
	}

	b.ResetTimer()
	for b.Loop() {
		_, _ = coord.Insert(ctx, vec, "data", nil)
	}
}
