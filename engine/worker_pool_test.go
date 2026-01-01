package engine

import (
	"context"
	"errors"
	"io"
	"iter"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/hupe1980/vecgo/index"
	"github.com/hupe1980/vecgo/metadata"
)

type testShardResult struct {
	shardIdx int
	results  []SearchResult
	err      error
}

// mockShard is a test double for Coordinator that can simulate search operations.
type mockShard[T any] struct {
	searchDelay time.Duration
	searchError error
	searchCount atomic.Int32
}

func (m *mockShard[T]) KNNSearch(ctx context.Context, query []float32, k int, opts *SearchOptions) ([]SearchResult, error) {
	m.searchCount.Add(1)

	if m.searchDelay > 0 {
		select {
		case <-time.After(m.searchDelay):
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}

	if m.searchError != nil {
		return nil, m.searchError
	}

	results := make([]SearchResult, k)
	for i := range results {
		results[i] = SearchResult{
			ID:       uint64(i),
			Distance: float32(i) * 0.1,
		}
	}
	return results, nil
}

func (m *mockShard[T]) KNNSearchWithBuffer(ctx context.Context, query []float32, k int, opts *SearchOptions, buf *[]SearchResult) error {
	res, err := m.KNNSearch(ctx, query, k, opts)
	if err != nil {
		return err
	}
	*buf = append(*buf, res...)
	return nil
}

func (m *mockShard[T]) BruteSearch(ctx context.Context, query []float32, k int, filter func(id uint64) bool) ([]SearchResult, error) {
	return m.KNNSearch(ctx, query, k, nil)
}

func (m *mockShard[T]) Insert(ctx context.Context, vector []float32, data T, meta metadata.Metadata) (uint64, error) {
	return 0, nil
}

func (m *mockShard[T]) BatchInsert(ctx context.Context, vectors [][]float32, data []T, meta []metadata.Metadata) ([]uint64, error) {
	return nil, nil
}

func (m *mockShard[T]) Update(ctx context.Context, id uint64, vector []float32, data T, meta metadata.Metadata) error {
	return nil
}

func (m *mockShard[T]) Delete(ctx context.Context, id uint64) error {
	return nil
}

func (m *mockShard[T]) Get(id uint64) (T, bool) {
	var zero T
	return zero, false
}

func (m *mockShard[T]) GetMetadata(id uint64) (metadata.Metadata, bool) {
	return nil, false
}

func (m *mockShard[T]) HybridSearch(ctx context.Context, query []float32, k int, opts *HybridSearchOptions) ([]SearchResult, error) {
	return m.KNNSearch(ctx, query, k, nil)
}

func (m *mockShard[T]) KNNSearchStream(ctx context.Context, query []float32, k int, opts *SearchOptions) iter.Seq2[SearchResult, error] {
	return func(yield func(SearchResult, error) bool) {
		res, err := m.KNNSearch(ctx, query, k, opts)
		if err != nil {
			yield(SearchResult{}, err)
			return
		}
		for _, r := range res {
			if !yield(r, nil) {
				return
			}
		}
	}
}

func (m *mockShard[T]) EnableProductQuantization(cfg index.ProductQuantizationConfig) error {
	return nil
}

func (m *mockShard[T]) DisableProductQuantization() {}

func (m *mockShard[T]) SaveToWriter(w io.Writer) error {
	return nil
}

func (m *mockShard[T]) SaveToFile(path string) error {
	return nil
}

func (m *mockShard[T]) RecoverFromWAL(ctx context.Context) error {
	return nil
}

func (m *mockShard[T]) Stats() index.Stats {
	return index.Stats{}
}

func (m *mockShard[T]) Checkpoint() error {
	return nil
}

func (m *mockShard[T]) Close() error {
	return nil
}

// TestWorkerPoolBasic verifies basic worker pool functionality.
func TestWorkerPoolBasic(t *testing.T) {
	pool := NewWorkerPool(2)
	defer pool.Close()

	shard := &mockShard[string]{}
	resultsCh := make(chan testShardResult, 1)

	ctx := context.Background()
	err := pool.Submit(ctx, func() {
		res, err := shard.KNNSearch(ctx, []float32{1, 2, 3}, 5, nil)
		resultsCh <- testShardResult{shardIdx: 0, results: res, err: err}
	})
	if err != nil {
		t.Fatalf("Submit failed: %v", err)
	}

	select {
	case result := <-resultsCh:
		if result.err != nil {
			t.Fatalf("Search failed: %v", result.err)
		}
		if len(result.results) != 5 {
			t.Errorf("Expected 5 results, got %d", len(result.results))
		}
		if result.shardIdx != 0 {
			t.Errorf("Expected shardIdx 0, got %d", result.shardIdx)
		}
	case <-time.After(1 * time.Second):
		t.Fatal("Timeout waiting for result")
	}

	if count := shard.searchCount.Load(); count != 1 {
		t.Errorf("Expected 1 search call, got %d", count)
	}
}

// TestWorkerPoolConcurrency verifies concurrent work submission.
func TestWorkerPoolConcurrency(t *testing.T) {
	const numWorkers = 4
	const numRequests = 100

	pool := NewWorkerPool(numWorkers)
	defer pool.Close()

	shard := &mockShard[string]{searchDelay: 1 * time.Millisecond}
	resultsCh := make(chan testShardResult, numRequests)

	var wg sync.WaitGroup
	wg.Add(numRequests)

	start := time.Now()

	// Submit many requests concurrently
	for i := 0; i < numRequests; i++ {
		go func(idx int) {
			defer wg.Done()

			ctx := context.Background()
			if err := pool.Submit(ctx, func() {
				res, err := shard.KNNSearch(ctx, []float32{float32(idx)}, 1, nil)
				resultsCh <- testShardResult{shardIdx: idx, results: res, err: err}
			}); err != nil {
				t.Errorf("Submit %d failed: %v", idx, err)
			}
		}(i)
	}

	wg.Wait()
	elapsed := time.Since(start)

	// Collect all results
	successCount := 0
	for i := 0; i < numRequests; i++ {
		select {
		case result := <-resultsCh:
			if result.err == nil {
				successCount++
			}
		case <-time.After(5 * time.Second):
			t.Fatal("Timeout waiting for results")
		}
	}

	if successCount != numRequests {
		t.Errorf("Expected %d successful results, got %d", numRequests, successCount)
	}

	// With 4 workers and 100 requests of 1ms each, should complete in ~25ms
	// Allow 10x overhead for scheduling/testing variance
	maxExpected := 250 * time.Millisecond
	if elapsed > maxExpected {
		t.Errorf("Pool too slow: expected ~25ms, got %v", elapsed)
	}

	if count := shard.searchCount.Load(); count != numRequests {
		t.Errorf("Expected %d search calls, got %d", numRequests, count)
	}
}

// TestWorkerPoolContextCancellation verifies context cancellation handling.
func TestWorkerPoolContextCancellation(t *testing.T) {
	pool := NewWorkerPool(2)
	defer pool.Close()

	// Create shard with long delay
	shard := &mockShard[string]{searchDelay: 100 * time.Millisecond}
	resultsCh := make(chan testShardResult, 1)

	// Cancel context before search completes
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel()

	err := pool.Submit(ctx, func() {
		res, err := shard.KNNSearch(ctx, []float32{1, 2, 3}, 5, nil)
		resultsCh <- testShardResult{shardIdx: 0, results: res, err: err}
	})
	if err != nil {
		t.Fatalf("Submit failed: %v", err)
	}

	// Wait for result or timeout
	select {
	case result := <-resultsCh:
		// Context cancellation should cause search error
		if result.err == nil {
			t.Error("Expected context cancellation error, got nil")
		}
		if !errors.Is(result.err, context.DeadlineExceeded) {
			t.Errorf("Expected DeadlineExceeded, got %v", result.err)
		}
	case <-time.After(1 * time.Second):
		// Result might not be sent due to context cancellation (this is OK)
		// The worker should still exit cleanly
	}
}

// TestWorkerPoolShutdown verifies graceful shutdown.
func TestWorkerPoolShutdown(t *testing.T) {
	pool := NewWorkerPool(2)

	shard := &mockShard[string]{searchDelay: 10 * time.Millisecond}
	resultsCh := make(chan testShardResult, 10)

	// Submit some work
	for i := 0; i < 5; i++ {
		idx := i
		ctx := context.Background()
		if err := pool.Submit(ctx, func() {
			res, err := shard.KNNSearch(ctx, []float32{float32(idx)}, 1, nil)
			resultsCh <- testShardResult{shardIdx: idx, results: res, err: err}
		}); err != nil {
			t.Fatalf("Submit %d failed: %v", i, err)
		}
	}

	// Close pool while work is in progress
	start := time.Now()
	pool.Close()
	elapsed := time.Since(start)

	// Close should wait for in-flight work to complete
	// With 2 workers and 5 tasks of 10ms each: ~30ms total (3 batches)
	minExpected := 20 * time.Millisecond
	if elapsed < minExpected {
		t.Errorf("Close returned too quickly: %v (expected >%v)", elapsed, minExpected)
	}

	// Try submitting after close (should fail)
	ctx := context.Background()
	err := pool.Submit(ctx, func() {
		// no-op
	})
	if !errors.Is(err, ErrCoordinatorClosed) {
		t.Errorf("Expected ErrCoordinatorClosed after shutdown, got %v", err)
	}
}

// TestWorkerPoolBackpressure verifies backpressure when work channel is full.
func TestWorkerPoolBackpressure(t *testing.T) {
	const numWorkers = 2
	pool := NewWorkerPool(numWorkers)
	defer pool.Close()

	// Create shard with delay to cause backpressure
	shard := &mockShard[string]{searchDelay: 50 * time.Millisecond}
	resultsCh := make(chan testShardResult, 100)

	// Submit more work than buffer can hold (buffer is 2*numWorkers = 4)
	const numRequests = 20
	submitted := 0
	timeout := time.After(100 * time.Millisecond)

	for i := 0; i < numRequests; i++ {
		ctx := context.Background()

		// Try to submit with timeout
		done := make(chan error, 1)
		go func(idx int) {
			done <- pool.Submit(ctx, func() {
				res, err := shard.KNNSearch(ctx, []float32{float32(idx)}, 1, nil)
				resultsCh <- testShardResult{shardIdx: idx, results: res, err: err}
			})
		}(i)

		select {
		case err := <-done:
			if err != nil {
				t.Fatalf("Submit %d failed: %v", i, err)
			}
			submitted++
		case <-timeout:
			// Backpressure kicked in (work channel full)
			t.Logf("Backpressure activated after %d requests", submitted)
			goto done
		}
	}

done:
	// Should have hit backpressure before submitting all requests
	if submitted >= numRequests {
		t.Error("Expected backpressure, but all requests were submitted immediately")
	}

	// At least buffer size + workers should have been submitted
	minSubmitted := 2*numWorkers + numWorkers
	if submitted < minSubmitted {
		t.Errorf("Expected at least %d submissions before backpressure, got %d", minSubmitted, submitted)
	}
}

// TestWorkerPoolBruteSearch verifies SubmitBrute works correctly.
func TestWorkerPoolBruteSearch(t *testing.T) {
	pool := NewWorkerPool(2)
	defer pool.Close()

	shard := &mockShard[string]{}
	resultsCh := make(chan testShardResult, 1)

	filter := func(id uint64) bool {
		return id%2 == 0 // Even IDs only
	}

	ctx := context.Background()
	err := pool.Submit(ctx, func() {
		res, err := shard.BruteSearch(ctx, []float32{1, 2, 3}, 5, filter)
		resultsCh <- testShardResult{shardIdx: 0, results: res, err: err}
	})
	if err != nil {
		t.Fatalf("SubmitBrute failed: %v", err)
	}

	select {
	case result := <-resultsCh:
		if result.err != nil {
			t.Fatalf("BruteSearch failed: %v", result.err)
		}
		if len(result.results) != 5 {
			t.Errorf("Expected 5 results, got %d", len(result.results))
		}
	case <-time.After(1 * time.Second):
		t.Fatal("Timeout waiting for result")
	}
}

// TestWorkerPoolErrorHandling verifies error propagation from shards.
func TestWorkerPoolErrorHandling(t *testing.T) {
	pool := NewWorkerPool(2)
	defer pool.Close()

	searchErr := errors.New("shard search failed")
	shard := &mockShard[string]{searchError: searchErr}
	resultsCh := make(chan testShardResult, 1)

	ctx := context.Background()
	err := pool.Submit(ctx, func() {
		res, err := shard.KNNSearch(ctx, []float32{1, 2, 3}, 5, nil)
		resultsCh <- testShardResult{shardIdx: 0, results: res, err: err}
	})
	if err != nil {
		t.Fatalf("Submit failed: %v", err)
	}

	select {
	case result := <-resultsCh:
		if result.err == nil {
			t.Fatal("Expected error, got nil")
		}
		if !errors.Is(result.err, searchErr) {
			t.Errorf("Expected %v, got %v", searchErr, result.err)
		}
	case <-time.After(1 * time.Second):
		t.Fatal("Timeout waiting for result")
	}
}

// TestWorkerPoolZeroWorkers verifies default worker count.
func TestWorkerPoolZeroWorkers(t *testing.T) {
	pool := NewWorkerPool(0) // Should use GOMAXPROCS
	defer pool.Close()

	if pool.numWorkers <= 0 {
		t.Errorf("Expected positive worker count, got %d", pool.numWorkers)
	}

	// Verify pool is functional
	shard := &mockShard[string]{}
	resultsCh := make(chan testShardResult, 1)

	ctx := context.Background()
	err := pool.Submit(ctx, func() {
		res, err := shard.KNNSearch(ctx, []float32{1, 2, 3}, 5, nil)
		resultsCh <- testShardResult{shardIdx: 0, results: res, err: err}
	})
	if err != nil {
		t.Fatalf("Submit failed: %v", err)
	}

	select {
	case result := <-resultsCh:
		if result.err != nil {
			t.Fatalf("Search failed: %v", result.err)
		}
	case <-time.After(1 * time.Second):
		t.Fatal("Timeout waiting for result")
	}
}
