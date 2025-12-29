package engine

import (
	"context"
	"fmt"

	"github.com/hupe1980/vecgo/codec"
	"github.com/hupe1980/vecgo/index"
	"github.com/hupe1980/vecgo/metadata"
)

// ShardedCoordinator wraps multiple independent coordinators (shards) to enable
// parallel write throughput. This eliminates the global lock bottleneck by
// partitioning vectors across shards.
//
// # ID Encoding
//
// ShardedCoordinator uses GlobalID encoding: [ShardID:8 bits][LocalID:24 bits]
// This enables O(1) shard routing for Update/Delete operations without external mapping.
//
// # Design
//
//   - Each shard is a complete Tx[T] with its own lock
//   - Write operations route to a single shard (shard-level lock only)
//   - Search operations fan out to all shards in parallel and merge results
//   - IDs are globally unique and self-describing (contain shard routing info)
//
// # Performance
//
//   - Target: 3-8x write speedup on multi-core systems
//   - Search: Parallel fan-out via worker pool (zero goroutine creation)
//   - Update/Delete: O(1) shard lookup from ID
//   - Worker pool: Constant goroutine count, reduced GC pressure
type ShardedCoordinator[T any] struct {
	shards     []*Tx[T]
	numShards  int
	nextShard  int            // round-robin counter for insert distribution
	workerPool *WorkerPool[T] // Fixed-size pool for parallel searches
}

// NewSharded creates a new sharded coordinator with independent shards.
//
// Parameters:
//   - indexes: One index per shard (must implement index.TransactionalIndex)
//   - dataStores: One data store per shard
//   - metaStores: One metadata store per shard
//   - durabilities: One durability layer per shard (or nil for all)
//   - codec: Shared codec for serialization (can be nil for default)
//
// All slice parameters must have the same length (numShards).
// Maximum supported shards: 256 (due to GlobalID encoding).
//
// Example:
//
//	indexes := make([]index.Index, 4)
//	for i := range indexes {
//	    indexes[i], _ = hnsw.NewSharded(i, 4, hnsw.WithDimension(128))
//	}
//	coord, _ := engine.NewSharded(indexes, dataStores, metaStores, nil, nil)
func NewSharded[T any](
	indexes []index.Index,
	dataStores []Store[T],
	metaStores []*metadata.UnifiedIndex,
	durabilities []Durability,
	codec codec.Codec,
) (Coordinator[T], error) {
	numShards := len(indexes)
	if numShards == 0 {
		return nil, fmt.Errorf("sharded: at least one shard required")
	}
	if numShards > MaxShards {
		return nil, fmt.Errorf("sharded: %d shards exceeds maximum %d", numShards, MaxShards)
	}
	if len(dataStores) != numShards {
		return nil, fmt.Errorf("dataStores length %d != numShards %d", len(dataStores), numShards)
	}
	if len(metaStores) != numShards {
		return nil, fmt.Errorf("metaStores length %d != numShards %d", len(metaStores), numShards)
	}

	// If durabilities not provided, create NoopDurability for each shard
	if durabilities == nil {
		durabilities = make([]Durability, numShards)
		for i := range durabilities {
			durabilities[i] = NoopDurability{}
		}
	}
	if len(durabilities) != numShards {
		return nil, fmt.Errorf("durabilities length %d != numShards %d", len(durabilities), numShards)
	}

	shards := make([]*Tx[T], numShards)
	for i := range indexes {
		coord, err := New(indexes[i], dataStores[i], metaStores[i], durabilities[i], codec)
		if err != nil {
			return nil, fmt.Errorf("failed to create shard %d: %w", i, err)
		}
		// Type assert since New returns interface but we know it's *Tx[T]
		shards[i] = coord.(*Tx[T])
	}

	// Create worker pool with one worker per shard (optimal CPU affinity)
	// This eliminates goroutine spam: 0 goroutines created per search vs N*QPS
	workerPool := NewWorkerPool[T](numShards)

	return &ShardedCoordinator[T]{
		shards:     shards,
		numShards:  numShards,
		nextShard:  0,
		workerPool: workerPool,
	}, nil
}

// shardForID returns the shard that owns the given global ID.
// Uses GlobalID encoding to extract shard index in O(1).
func (sc *ShardedCoordinator[T]) shardForID(id uint32) (*Tx[T], error) {
	gid := GlobalID(id)
	shardIdx := gid.ShardIndex()
	if shardIdx >= sc.numShards {
		return nil, fmt.Errorf("invalid shard index %d in ID %d (have %d shards)", shardIdx, id, sc.numShards)
	}
	return sc.shards[shardIdx], nil
}

// Insert adds a vector+payload+metadata atomically to a shard.
//
// Shard selection uses round-robin for balanced distribution.
// Returns a GlobalID that encodes the shard index for future Update/Delete routing.
func (sc *ShardedCoordinator[T]) Insert(ctx context.Context, vector []float32, data T, meta metadata.Metadata) (uint32, error) {
	// Round-robin shard selection
	shardIdx := sc.nextShard
	sc.nextShard = (sc.nextShard + 1) % sc.numShards

	// Insert into shard (shard allocates its own local ID)
	localID, err := sc.shards[shardIdx].Insert(ctx, vector, data, meta)
	if err != nil {
		return 0, err
	}

	// Return global ID with shard encoded
	return uint32(NewGlobalID(shardIdx, localID)), nil
}

// BatchInsert adds multiple vectors+payloads+metadata.
//
// Items are distributed across shards in round-robin fashion for balanced load.
// Returns GlobalIDs that encode shard routing information.
func (sc *ShardedCoordinator[T]) BatchInsert(ctx context.Context, vectors [][]float32, dataSlice []T, metadataSlice []metadata.Metadata) ([]uint32, error) {
	if len(vectors) == 0 {
		return nil, nil
	}

	// Group items by shard (round-robin distribution)
	type shardBatch struct {
		indices   []int // original indices for result ordering
		vectors   [][]float32
		dataSlice []T
		metadata  []metadata.Metadata
	}
	shardBatches := make([]*shardBatch, sc.numShards)
	for i := 0; i < sc.numShards; i++ {
		shardBatches[i] = &shardBatch{
			indices:   make([]int, 0),
			vectors:   make([][]float32, 0),
			dataSlice: make([]T, 0),
			metadata:  make([]metadata.Metadata, 0),
		}
	}

	// Distribute items round-robin starting from current nextShard
	startShard := sc.nextShard
	for i := range vectors {
		shardIdx := (startShard + i) % sc.numShards
		batch := shardBatches[shardIdx]
		batch.indices = append(batch.indices, i)
		batch.vectors = append(batch.vectors, vectors[i])
		batch.dataSlice = append(batch.dataSlice, dataSlice[i])
		if metadataSlice != nil && i < len(metadataSlice) {
			batch.metadata = append(batch.metadata, metadataSlice[i])
		}
	}
	// Update nextShard for subsequent inserts
	sc.nextShard = (startShard + len(vectors)) % sc.numShards

	// Insert batches into each shard in parallel
	type shardResult struct {
		shardIdx int
		ids      []uint32 // local IDs from shard
		err      error
	}
	resultsCh := make(chan shardResult, sc.numShards)

	for idx, batch := range shardBatches {
		if len(batch.vectors) == 0 {
			resultsCh <- shardResult{shardIdx: idx, ids: []uint32{}, err: nil}
			continue
		}
		go func(shardIdx int, b *shardBatch) {
			ids, err := sc.shards[shardIdx].BatchInsert(ctx, b.vectors, b.dataSlice, b.metadata)
			resultsCh <- shardResult{shardIdx: shardIdx, ids: ids, err: err}
		}(idx, batch)
	}

	// Collect results indexed by shard
	allIDs := make([]uint32, len(vectors))
	shardIDsMap := make(map[int][]uint32)
	for i := 0; i < sc.numShards; i++ {
		res := <-resultsCh
		if res.err != nil {
			return nil, res.err
		}
		shardIDsMap[res.shardIdx] = res.ids
	}

	// Map shard-local IDs to global IDs in original order
	for shardIdx, batch := range shardBatches {
		localIDs := shardIDsMap[shardIdx]
		for i, originalIdx := range batch.indices {
			// Convert local ID to global ID with shard encoding
			allIDs[originalIdx] = uint32(NewGlobalID(shardIdx, localIDs[i]))
		}
	}

	return allIDs, nil
}

// Update updates a vector+payload+metadata atomically in the appropriate shard.
// The shard is determined by extracting the shard index from the GlobalID.
func (sc *ShardedCoordinator[T]) Update(ctx context.Context, id uint32, vector []float32, data T, meta metadata.Metadata) error {
	gid := GlobalID(id)
	shard, err := sc.shardForID(id)
	if err != nil {
		return err
	}
	// Use local ID for shard operation
	return shard.Update(ctx, gid.LocalID(), vector, data, meta)
}

// Delete removes a vector+payload+metadata atomically from the appropriate shard.
// The shard is determined by extracting the shard index from the GlobalID.
func (sc *ShardedCoordinator[T]) Delete(ctx context.Context, id uint32) error {
	gid := GlobalID(id)
	shard, err := sc.shardForID(id)
	if err != nil {
		return err
	}
	// Use local ID for shard operation
	return shard.Delete(ctx, gid.LocalID())
}

// Get retrieves the data associated with an ID from the appropriate shard.
// The shard is determined by extracting the shard index from the GlobalID.
func (sc *ShardedCoordinator[T]) Get(id uint32) (T, bool) {
	shard, err := sc.shardForID(id)
	if err != nil {
		var zero T
		return zero, false
	}
	gid := GlobalID(id)
	return shard.Get(gid.LocalID())
}

// GetMetadata retrieves the metadata associated with an ID from the appropriate shard.
// The shard is determined by extracting the shard index from the GlobalID.
func (sc *ShardedCoordinator[T]) GetMetadata(id uint32) (metadata.Metadata, bool) {
	shard, err := sc.shardForID(id)
	if err != nil {
		return nil, false
	}
	gid := GlobalID(id)
	return shard.GetMetadata(gid.LocalID())
}

// KNNSearch performs a K-nearest neighbor search across all shards in parallel.
//
// This method fans out the query to all shards simultaneously, collects results,
// and merges them to return the global top-k results. Result IDs are GlobalIDs
// that encode shard routing information.
//
// Error Handling:
//   - Returns error if any shard fails (fail-fast behavior for correctness)
//   - Respects context cancellation and deadlines
//   - Includes shard index in error messages for debugging
//
// Performance:
//   - Parallel fan-out utilizes multiple cores for search
//   - Latency is roughly the same as single-shard (limited by slowest shard)
//   - Throughput can improve with concurrent searches
func (sc *ShardedCoordinator[T]) KNNSearch(ctx context.Context, query []float32, k int, opts *index.SearchOptions) ([]index.SearchResult, error) {
	if k <= 0 {
		return nil, index.ErrInvalidK
	}

	// Fan out to all shards via worker pool (zero goroutine creation)
	resultsCh := make(chan shardResult[T], sc.numShards)

	for i := 0; i < sc.numShards; i++ {
		// Wrap filter to convert local IDs to global IDs before checking
		shardOpts := opts
		if opts != nil && opts.Filter != nil {
			userFilter := opts.Filter
			shardIdx := i // capture for closure
			shardOpts = &index.SearchOptions{
				EFSearch: opts.EFSearch,
				Filter: func(localID uint32) bool {
					globalID := uint32(NewGlobalID(shardIdx, localID))
					return userFilter(globalID)
				},
			}
		}

		// Submit to worker pool instead of spawning goroutine
		req := WorkRequest[T]{
			shardIdx: i,
			shard:    sc.shards[i],
			query:    query,
			k:        k,
			opts:     shardOpts,
			resultCh: resultsCh,
		}

		if err := sc.workerPool.Submit(ctx, req); err != nil {
			// Pool closed or context cancelled
			return nil, fmt.Errorf("worker pool submit failed: %w", err)
		}
	}

	// Collect results from all shards and convert to global IDs
	allResults := make([]index.SearchResult, 0, k*sc.numShards)
	var errors []error

	for i := 0; i < sc.numShards; i++ {
		select {
		case res := <-resultsCh:
			if res.err != nil {
				errors = append(errors, fmt.Errorf("shard %d: %w", res.shardIdx, res.err))
			} else {
				// Convert local IDs to global IDs
				for j := range res.results {
					res.results[j].ID = uint32(NewGlobalID(res.shardIdx, res.results[j].ID))
				}
				allResults = append(allResults, res.results...)
			}
		case <-ctx.Done():
			// Context cancelled (timeout or explicit cancellation)
			return nil, fmt.Errorf("search cancelled: %w", ctx.Err())
		}
	}

	// Fail if any shard failed (fail-fast for correctness)
	if len(errors) > 0 {
		return nil, fmt.Errorf("parallel search failed (%d/%d shards): %v", len(errors), sc.numShards, errors)
	}

	// Merge results to get global top-k
	return mergeTopK(allResults, k), nil
}

// BruteSearch performs a brute-force search across all shards in parallel.
//
// Similar to KNNSearch, this fans out to all shards and merges results.
// Result IDs are GlobalIDs that encode shard routing information.
//
// Error Handling:
//   - Returns error if any shard fails (fail-fast behavior)
//   - Respects context cancellation and deadlines
//   - Includes shard index in error messages
func (sc *ShardedCoordinator[T]) BruteSearch(ctx context.Context, query []float32, k int, filter func(id uint32) bool) ([]index.SearchResult, error) {
	if k <= 0 {
		return nil, index.ErrInvalidK
	}

	// Fan out to all shards via worker pool (zero goroutine creation)
	resultsCh := make(chan shardResult[T], sc.numShards)

	for i := 0; i < sc.numShards; i++ {
		// Wrap filter to translate global IDs to local IDs for filtering
		var localFilter func(id uint32) bool
		if filter != nil {
			shardIdx := i // capture for closure
			localFilter = func(localID uint32) bool {
				globalID := uint32(NewGlobalID(shardIdx, localID))
				return filter(globalID)
			}
		}

		// Submit to worker pool instead of spawning goroutine
		req := WorkRequest[T]{
			shardIdx: i,
			shard:    sc.shards[i],
			query:    query,
			k:        k,
			resultCh: resultsCh,
		}

		if err := sc.workerPool.SubmitBrute(ctx, req, localFilter); err != nil {
			// Pool closed or context cancelled
			return nil, fmt.Errorf("worker pool submit failed: %w", err)
		}
	}

	// Collect results from all shards and convert to global IDs
	allResults := make([]index.SearchResult, 0, k*sc.numShards)
	var errors []error

	for i := 0; i < sc.numShards; i++ {
		select {
		case res := <-resultsCh:
			if res.err != nil {
				errors = append(errors, fmt.Errorf("shard %d: %w", res.shardIdx, res.err))
			} else {
				// Convert local IDs to global IDs
				for j := range res.results {
					res.results[j].ID = uint32(NewGlobalID(res.shardIdx, res.results[j].ID))
				}
				allResults = append(allResults, res.results...)
			}
		case <-ctx.Done():
			// Context cancelled (timeout or explicit cancellation)
			return nil, fmt.Errorf("brute search cancelled: %w", ctx.Err())
		}
	}

	// Fail if any shard failed (fail-fast for correctness)
	if len(errors) > 0 {
		return nil, fmt.Errorf("parallel brute search failed (%d/%d shards): %v", len(errors), sc.numShards, errors)
	}

	// Merge results to get global top-k
	return mergeTopK(allResults, k), nil
}

// mergeTopK merges results from multiple shards and returns the top-k globally.
// Results are sorted by distance (ascending) and limited to k results.
func mergeTopK(results []index.SearchResult, k int) []index.SearchResult {
	if len(results) == 0 {
		return nil
	}

	// Use insertion sort for small k (typical case)
	// This is more efficient than a full sort for k << len(results)
	if len(results) <= k {
		// All results fit, just sort them
		sortResults(results)
		return results
	}

	// Build min-heap of top-k results
	topK := make([]index.SearchResult, 0, k)
	for _, result := range results {
		if len(topK) < k {
			// Heap not full, add result
			topK = append(topK, result)
			if len(topK) == k {
				// Heap just filled, heapify
				buildMaxHeap(topK)
			}
		} else if result.Distance < topK[0].Distance {
			// Better than worst in heap, replace root
			topK[0] = result
			heapifyDown(topK, 0)
		}
	}

	// Sort final results (ascending by distance)
	sortResults(topK)
	return topK
}

// sortResults sorts results by distance (ascending)
func sortResults(results []index.SearchResult) {
	// Simple insertion sort for small k
	for i := 1; i < len(results); i++ {
		key := results[i]
		j := i - 1
		for j >= 0 && results[j].Distance > key.Distance {
			results[j+1] = results[j]
			j--
		}
		results[j+1] = key
	}
}

// buildMaxHeap builds a max-heap (largest distance at root) for top-k tracking
func buildMaxHeap(arr []index.SearchResult) {
	for i := len(arr)/2 - 1; i >= 0; i-- {
		heapifyDown(arr, i)
	}
}

// heapifyDown maintains max-heap property downward from index i
func heapifyDown(arr []index.SearchResult, i int) {
	n := len(arr)
	for {
		largest := i
		left := 2*i + 1
		right := 2*i + 2

		if left < n && arr[left].Distance > arr[largest].Distance {
			largest = left
		}
		if right < n && arr[right].Distance > arr[largest].Distance {
			largest = right
		}

		if largest == i {
			break
		}

		arr[i], arr[largest] = arr[largest], arr[i]
		i = largest
	}
}

// Close closes all shards and releases their resources.
// This method ensures proper cleanup of all shard resources, including
// the worker pool and any background workers in the underlying indexes.
//
// Shutdown order:
//  1. Close worker pool (waits for in-flight searches to complete)
//  2. Close all shard indexes (background workers, caches, etc.)
//
// This ensures graceful shutdown with no work interruption.
func (sc *ShardedCoordinator[T]) Close() error {
	// First, close worker pool to stop accepting new work
	if sc.workerPool != nil {
		sc.workerPool.Close()
	}

	// Then close all shards
	var errs []error
	for i, shard := range sc.shards {
		// Each shard's txIndex may have background workers (e.g., DiskANN compaction)
		// Close the index to ensure clean shutdown
		if closeable, ok := shard.txIndex.(interface{ Close() error }); ok {
			if err := closeable.Close(); err != nil {
				errs = append(errs, fmt.Errorf("shard %d index: %w", i, err))
			}
		}
	}
	if len(errs) > 0 {
		return fmt.Errorf("sharded close errors: %v", errs)
	}
	return nil
}
