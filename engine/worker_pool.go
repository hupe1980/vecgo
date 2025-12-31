package engine

import (
	"context"
	"runtime"
	"sync"
	"sync/atomic"
)

// WorkerPool manages a fixed pool of goroutines for parallel tasks.
// This eliminates the performance overhead of spawning thousands of goroutines
// per second under high query load.
type WorkerPool struct {
	numWorkers int
	workCh     chan func() // Channel carries work closures
	stopCh     chan struct{}
	wg         sync.WaitGroup
	closed     atomic.Bool // Tracks if pool is closed
	submitMu   sync.RWMutex
}

// NewWorkerPool creates a worker pool with numWorkers goroutines.
//
// Recommended sizing:
//   - numWorkers = numShards (one worker per shard for optimal affinity)
//   - For CPU-bound searches: runtime.GOMAXPROCS(0)
//   - For I/O-bound searches: 2-4x GOMAXPROCS
func NewWorkerPool(numWorkers int) *WorkerPool {
	if numWorkers <= 0 {
		numWorkers = runtime.GOMAXPROCS(0)
	}

	wp := &WorkerPool{
		numWorkers: numWorkers,
		workCh:     make(chan func(), numWorkers*2), // 2x buffer for pipelining
		stopCh:     make(chan struct{}),
	}

	// Start worker goroutines
	wp.wg.Add(numWorkers)
	for i := 0; i < numWorkers; i++ {
		go wp.worker()
	}

	return wp
}

// worker processes work closures from the work channel.
func (wp *WorkerPool) worker() {
	defer wp.wg.Done()

	for {
		select {
		case <-wp.stopCh:
			// Drain remaining work before exiting
			for {
				select {
				case workFunc, ok := <-wp.workCh:
					if !ok {
						return
					}
					workFunc()
				default:
					return
				}
			}
		case workFunc, ok := <-wp.workCh:
			if !ok {
				return
			}
			workFunc()
		}
	}
}

// Submit submits a task to the worker pool.
//
// The function returns immediately after enqueueing the work.
//
// Error conditions:
//   - Returns error if pool is closed
//   - Returns error if context is cancelled before enqueueing
func (wp *WorkerPool) Submit(ctx context.Context, task func()) error {
	wp.submitMu.RLock()
	defer wp.submitMu.RUnlock()

	// Check if closed first
	if wp.closed.Load() {
		return ErrCoordinatorClosed
	}

	// Enqueue work (with backpressure)
	select {
	case wp.workCh <- task:
		return nil
	case <-wp.stopCh:
		return ErrCoordinatorClosed
	case <-ctx.Done():
		return ctx.Err()
	}
}

// Close shuts down the worker pool gracefully.
func (wp *WorkerPool) Close() {
	// Mark as closed (atomic, idempotent)
	if !wp.closed.CompareAndSwap(false, true) {
		return
	}

	wp.submitMu.Lock()
	close(wp.stopCh)
	close(wp.workCh)
	wp.submitMu.Unlock()

	wp.wg.Wait()
}
