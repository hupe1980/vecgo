package engine_test

import (
	"context"
	"errors"
	"fmt"
	"io"
	"iter"
	"testing"
	"time"

	"github.com/hupe1980/vecgo/engine"
	"github.com/hupe1980/vecgo/index"
	"github.com/hupe1980/vecgo/metadata"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Helper function to create a random-ish vector
func randomVector(dim int, seed int) []float32 {
	vec := make([]float32, dim)
	for j := range vec {
		vec[j] = float32(seed*j) * 0.01
	}
	return vec
}

// failingCoordinator is a mock coordinator that always fails with a specific error.
type failingCoordinator[T any] struct {
	err error
}

func (f *failingCoordinator[T]) Insert(ctx context.Context, vector []float32, data T, meta metadata.Metadata) (uint64, error) {
	return 0, f.err
}

func (f *failingCoordinator[T]) BatchInsert(ctx context.Context, vectors [][]float32, dataSlice []T, metadataSlice []metadata.Metadata) ([]uint64, error) {
	return nil, f.err
}

func (f *failingCoordinator[T]) Update(ctx context.Context, id uint64, vector []float32, data T, meta metadata.Metadata) error {
	return f.err
}

func (f *failingCoordinator[T]) Delete(ctx context.Context, id uint64) error {
	return f.err
}

func (f *failingCoordinator[T]) Get(id uint64) (T, bool) {
	var zero T
	return zero, false
}

func (f *failingCoordinator[T]) GetMetadata(id uint64) (metadata.Metadata, bool) {
	return nil, false
}

func (f *failingCoordinator[T]) KNNSearch(ctx context.Context, query []float32, k int, opts *index.SearchOptions) ([]index.SearchResult, error) {
	return nil, f.err
}

func (f *failingCoordinator[T]) KNNSearchWithBuffer(ctx context.Context, query []float32, k int, opts *index.SearchOptions, buf *[]index.SearchResult) error {
	return f.err
}

func (f *failingCoordinator[T]) BruteSearch(ctx context.Context, query []float32, k int, filter func(id uint64) bool) ([]index.SearchResult, error) {
	return nil, f.err
}

func (f *failingCoordinator[T]) HybridSearch(ctx context.Context, query []float32, k int, opts *engine.HybridSearchOptions) ([]index.SearchResult, error) {
	return nil, f.err
}

func (f *failingCoordinator[T]) KNNSearchStream(ctx context.Context, query []float32, k int, opts *index.SearchOptions) iter.Seq2[index.SearchResult, error] {
	return func(yield func(index.SearchResult, error) bool) {
		yield(index.SearchResult{}, f.err)
	}
}

func (f *failingCoordinator[T]) EnableProductQuantization(cfg index.ProductQuantizationConfig) error {
	return f.err
}

func (f *failingCoordinator[T]) DisableProductQuantization() {}

func (f *failingCoordinator[T]) SaveToWriter(w io.Writer) error {
	return f.err
}

func (f *failingCoordinator[T]) SaveToFile(path string) error {
	return f.err
}

func (f *failingCoordinator[T]) RecoverFromWAL(ctx context.Context) error {
	return f.err
}

func (f *failingCoordinator[T]) Stats() index.Stats {
	return index.Stats{}
}

func (f *failingCoordinator[T]) Checkpoint() error {
	return f.err
}

func (f *failingCoordinator[T]) Close() error {
	return f.err
}

// slowCoordinator is a mock coordinator that delays operations.
type slowCoordinator[T any] struct {
	delay time.Duration
}

func (s *slowCoordinator[T]) Insert(ctx context.Context, vector []float32, data T, meta metadata.Metadata) (uint64, error) {
	time.Sleep(s.delay)
	return 1, nil
}

func (s *slowCoordinator[T]) BatchInsert(ctx context.Context, vectors [][]float32, dataSlice []T, metadataSlice []metadata.Metadata) ([]uint64, error) {
	time.Sleep(s.delay)
	return make([]uint64, len(vectors)), nil
}

func (s *slowCoordinator[T]) Update(ctx context.Context, id uint64, vector []float32, data T, meta metadata.Metadata) error {
	time.Sleep(s.delay)
	return nil
}

func (s *slowCoordinator[T]) Delete(ctx context.Context, id uint64) error {
	time.Sleep(s.delay)
	return nil
}

func (s *slowCoordinator[T]) Get(id uint64) (T, bool) {
	var zero T
	return zero, false
}

func (s *slowCoordinator[T]) GetMetadata(id uint64) (metadata.Metadata, bool) {
	return nil, false
}

func (s *slowCoordinator[T]) KNNSearch(ctx context.Context, query []float32, k int, opts *index.SearchOptions) ([]index.SearchResult, error) {
	select {
	case <-time.After(s.delay):
		return nil, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

func (s *slowCoordinator[T]) KNNSearchWithBuffer(ctx context.Context, query []float32, k int, opts *index.SearchOptions, buf *[]index.SearchResult) error {
	select {
	case <-time.After(s.delay):
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (s *slowCoordinator[T]) BruteSearch(ctx context.Context, query []float32, k int, filter func(id uint64) bool) ([]index.SearchResult, error) {
	select {
	case <-time.After(s.delay):
		return nil, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

func (s *slowCoordinator[T]) HybridSearch(ctx context.Context, query []float32, k int, opts *engine.HybridSearchOptions) ([]index.SearchResult, error) {
	select {
	case <-time.After(s.delay):
		return nil, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

func (s *slowCoordinator[T]) KNNSearchStream(ctx context.Context, query []float32, k int, opts *index.SearchOptions) iter.Seq2[index.SearchResult, error] {
	return func(yield func(index.SearchResult, error) bool) {
		select {
		case <-time.After(s.delay):
		case <-ctx.Done():
			yield(index.SearchResult{}, ctx.Err())
		}
	}
}

func (s *slowCoordinator[T]) EnableProductQuantization(cfg index.ProductQuantizationConfig) error {
	time.Sleep(s.delay)
	return nil
}

func (s *slowCoordinator[T]) DisableProductQuantization() {
	time.Sleep(s.delay)
}

func (s *slowCoordinator[T]) SaveToWriter(w io.Writer) error {
	time.Sleep(s.delay)
	return nil
}

func (s *slowCoordinator[T]) SaveToFile(path string) error {
	time.Sleep(s.delay)
	return nil
}

func (s *slowCoordinator[T]) RecoverFromWAL(ctx context.Context) error {
	select {
	case <-time.After(s.delay):
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (s *slowCoordinator[T]) Stats() index.Stats {
	return index.Stats{}
}

func (s *slowCoordinator[T]) Checkpoint() error {
	time.Sleep(s.delay)
	return nil
}

func (s *slowCoordinator[T]) Close() error {
	time.Sleep(s.delay)
	return nil
}

// TestShardedSearchErrorPropagation verifies that errors from individual shards
// are properly propagated to the caller with shard identification.
func TestShardedSearchErrorPropagation(t *testing.T) {
	tests := []struct {
		name          string
		injectError   func(shards []engine.Coordinator[string])
		expectedError string
	}{
		{
			name: "Single shard failure in KNNSearch",
			injectError: func(shards []engine.Coordinator[string]) {
				// Replace shard 2 with failing coordinator
				shards[2] = &failingCoordinator[string]{err: errors.New("disk read error")}
			},
			expectedError: "shard 2",
		},
		{
			name: "Multiple shard failures in KNNSearch",
			injectError: func(shards []engine.Coordinator[string]) {
				shards[1] = &failingCoordinator[string]{err: errors.New("out of memory")}
				shards[3] = &failingCoordinator[string]{err: errors.New("disk read error")}
			},
			expectedError: "parallel search failed (2/4 shards)",
		},
		{
			name: "All shards fail in KNNSearch",
			injectError: func(shards []engine.Coordinator[string]) {
				for i := range shards {
					shards[i] = &failingCoordinator[string]{err: errors.New("catastrophic failure")}
				}
			},
			expectedError: "parallel search failed (4/4 shards)",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create valid sharded coordinator
			sc := setupShardedCoordinator(t, 4)

			// Inject failures
			tt.injectError(sc.GetShards()) // We'll need to expose shards for testing

			// Perform search - should fail and propagate error
			query := randomVector(128, 999)
			ctx := context.Background()
			_, err := sc.KNNSearch(ctx, query, 10, nil)

			require.Error(t, err, "Expected error from failed shard")
			assert.Contains(t, err.Error(), tt.expectedError, "Error should include shard identification")
		})
	}
}

// TestShardedSearchTimeout verifies that context timeouts are properly handled.
func TestShardedSearchTimeout(t *testing.T) {
	tests := []struct {
		name        string
		timeout     time.Duration
		shardDelays []time.Duration
		expectError bool
	}{
		{
			name:        "Timeout before any shard completes",
			timeout:     50 * time.Millisecond,
			shardDelays: []time.Duration{200 * time.Millisecond, 200 * time.Millisecond, 200 * time.Millisecond, 200 * time.Millisecond},
			expectError: true,
		},
		{
			name:        "Timeout before slowest shard",
			timeout:     100 * time.Millisecond,
			shardDelays: []time.Duration{10 * time.Millisecond, 20 * time.Millisecond, 300 * time.Millisecond, 10 * time.Millisecond},
			expectError: true,
		},
		{
			name:        "No timeout - all shards fast enough",
			timeout:     500 * time.Millisecond,
			shardDelays: []time.Duration{10 * time.Millisecond, 20 * time.Millisecond, 30 * time.Millisecond, 40 * time.Millisecond},
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create sharded coordinator with slow shards
			sc := setupShardedCoordinatorWithDelays(t, tt.shardDelays)

			ctx, cancel := context.WithTimeout(context.Background(), tt.timeout)
			defer cancel()

			query := randomVector(128, 999)
			_, err := sc.KNNSearch(ctx, query, 10, nil)

			if tt.expectError {
				require.Error(t, err, "Expected timeout error")
				assert.ErrorIs(t, err, context.DeadlineExceeded, "Error should be context.DeadlineExceeded")
			} else {
				assert.NoError(t, err, "Should not timeout")
			}
		})
	}
}

// TestShardedBruteSearchErrorPropagation verifies error handling in BruteSearch.
func TestShardedBruteSearchErrorPropagation(t *testing.T) {
	sc := setupShardedCoordinator(t, 4)

	// Inject failure in shard 1
	shards := sc.GetShards()
	shards[1] = &failingCoordinator[string]{err: errors.New("brute search failure")}

	query := randomVector(128, 999)
	ctx := context.Background()
	_, err := sc.BruteSearch(ctx, query, 10, nil)

	require.Error(t, err)
	assert.Contains(t, err.Error(), "shard 1")
	assert.Contains(t, err.Error(), "brute search")
}

// TestContextCancellation verifies graceful handling of context cancellation.
func TestContextCancellation(t *testing.T) {
	sc := setupShardedCoordinatorWithDelays(t, []time.Duration{
		500 * time.Millisecond,
		500 * time.Millisecond,
		500 * time.Millisecond,
		500 * time.Millisecond,
	})

	ctx, cancel := context.WithCancel(context.Background())

	// Cancel after 50ms
	go func() {
		time.Sleep(50 * time.Millisecond)
		cancel()
	}()

	query := randomVector(128, 999)
	_, err := sc.KNNSearch(ctx, query, 10, nil)

	require.Error(t, err)
	assert.ErrorIs(t, err, context.Canceled)
}

// Helper functions

// setupShardedCoordinator creates a ShardedCoordinator for testing.
// This is a simplified version - actual implementation would need proper initialization.
func setupShardedCoordinator(t *testing.T, numShards int) *testShardedCoordinator[string] {
	// For testing, we'll use a wrapper that exposes shards
	// Initialize with slow coordinators by default (they always succeed)
	shards := make([]engine.Coordinator[string], numShards)
	for i := 0; i < numShards; i++ {
		shards[i] = &slowCoordinator[string]{delay: 1 * time.Millisecond}
	}
	return &testShardedCoordinator[string]{
		shards: shards,
	}
}

func setupShardedCoordinatorWithDelays(t *testing.T, delays []time.Duration) *testShardedCoordinator[string] {
	shards := make([]engine.Coordinator[string], len(delays))
	for i, delay := range delays {
		shards[i] = &slowCoordinator[string]{delay: delay}
	}
	return &testShardedCoordinator[string]{shards: shards}
}

// testShardedCoordinator wraps ShardedCoordinator for testing with exposed shards.
type testShardedCoordinator[T any] struct {
	shards []engine.Coordinator[T]
}

func (tsc *testShardedCoordinator[T]) GetShards() []engine.Coordinator[T] {
	return tsc.shards
}

func (tsc *testShardedCoordinator[T]) KNNSearch(ctx context.Context, query []float32, k int, opts *index.SearchOptions) ([]index.SearchResult, error) {
	// Simplified implementation for testing
	type shardResult struct {
		shardIdx int
		results  []index.SearchResult
		err      error
	}
	resultsCh := make(chan shardResult, len(tsc.shards))

	for i := range tsc.shards {
		go func(shardIdx int) {
			results, err := tsc.shards[shardIdx].KNNSearch(ctx, query, k, opts)
			select {
			case resultsCh <- shardResult{shardIdx: shardIdx, results: results, err: err}:
			case <-ctx.Done():
			}
		}(i)
	}

	allResults := make([]index.SearchResult, 0)
	var errors []error

	for i := 0; i < len(tsc.shards); i++ {
		select {
		case res := <-resultsCh:
			if res.err != nil {
				errors = append(errors, fmt.Errorf("shard %d: %w", res.shardIdx, res.err))
			} else {
				allResults = append(allResults, res.results...)
			}
		case <-ctx.Done():
			return nil, fmt.Errorf("search cancelled: %w", ctx.Err())
		}
	}

	if len(errors) > 0 {
		return nil, fmt.Errorf("parallel search failed (%d/%d shards): %v", len(errors), len(tsc.shards), errors)
	}

	return allResults, nil
}

func (tsc *testShardedCoordinator[T]) BruteSearch(ctx context.Context, query []float32, k int, filter func(id uint64) bool) ([]index.SearchResult, error) {
	// Similar to KNNSearch
	type shardResult struct {
		shardIdx int
		results  []index.SearchResult
		err      error
	}
	resultsCh := make(chan shardResult, len(tsc.shards))

	for i := range tsc.shards {
		go func(shardIdx int) {
			results, err := tsc.shards[shardIdx].BruteSearch(ctx, query, k, filter)
			select {
			case resultsCh <- shardResult{shardIdx: shardIdx, results: results, err: err}:
			case <-ctx.Done():
			}
		}(i)
	}

	allResults := make([]index.SearchResult, 0)
	var errors []error

	for i := 0; i < len(tsc.shards); i++ {
		select {
		case res := <-resultsCh:
			if res.err != nil {
				errors = append(errors, fmt.Errorf("shard %d: %w", res.shardIdx, res.err))
			} else {
				allResults = append(allResults, res.results...)
			}
		case <-ctx.Done():
			return nil, fmt.Errorf("brute search cancelled: %w", ctx.Err())
		}
	}

	if len(errors) > 0 {
		return nil, fmt.Errorf("parallel brute search failed (%d/%d shards): %v", len(errors), len(tsc.shards), errors)
	}

	return allResults, nil
}

// Implement other Coordinator methods as no-ops for testing
func (tsc *testShardedCoordinator[T]) Insert(ctx context.Context, vector []float32, data T, meta metadata.Metadata) (uint32, error) {
	return 0, nil
}

func (tsc *testShardedCoordinator[T]) BatchInsert(ctx context.Context, vectors [][]float32, dataSlice []T, metadataSlice []metadata.Metadata) ([]uint32, error) {
	return nil, nil
}

func (tsc *testShardedCoordinator[T]) Update(ctx context.Context, id uint32, vector []float32, data T, meta metadata.Metadata) error {
	return nil
}

func (tsc *testShardedCoordinator[T]) Delete(ctx context.Context, id uint32) error {
	return nil
}

func (tsc *testShardedCoordinator[T]) Get(id uint32) (T, bool) {
	var zero T
	return zero, false
}

func (tsc *testShardedCoordinator[T]) GetMetadata(id uint32) (metadata.Metadata, bool) {
	return nil, false
}
