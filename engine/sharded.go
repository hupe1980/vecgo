package engine

import (
	"context"
	"fmt"
	"io"
	"iter"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"sync"

	"github.com/hupe1980/vecgo/codec"
	"github.com/hupe1980/vecgo/index"
	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/searcher"
)

// ShardedCoordinator wraps multiple independent coordinators (shards) to enable
// parallel write throughput. This eliminates the global lock bottleneck by
// partitioning vectors across shards.
//
// # ID Encoding
//
// ShardedCoordinator uses GlobalID encoding: [ShardID:8 bits][LocalID:56 bits]
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
	nextShard  int         // round-robin counter for insert distribution
	workerPool *WorkerPool // Fixed-size pool for parallel searches
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
	optFns ...Option,
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
		coord, err := New(indexes[i], dataStores[i], metaStores[i], durabilities[i], codec, optFns...)
		if err != nil {
			return nil, fmt.Errorf("failed to create shard %d: %w", i, err)
		}
		// Type assert since New returns interface but we know it's *Tx[T]
		shards[i] = coord.(*Tx[T])
	}

	// Create worker pool with one worker per shard (optimal CPU affinity)
	// This eliminates goroutine spam: 0 goroutines created per search vs N*QPS
	poolSize := numShards
	if procs := runtime.GOMAXPROCS(0); procs > poolSize {
		poolSize = procs
	}
	workerPool := NewWorkerPool(poolSize)

	return &ShardedCoordinator[T]{
		shards:     shards,
		numShards:  numShards,
		nextShard:  0,
		workerPool: workerPool,
	}, nil
}

// shardForID returns the shard that owns the given global ID.
// Uses GlobalID encoding to extract shard index in O(1).
func (sc *ShardedCoordinator[T]) shardForID(id uint64) (*Tx[T], error) {
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
func (sc *ShardedCoordinator[T]) Insert(ctx context.Context, vector []float32, data T, meta metadata.Metadata) (uint64, error) {
	// Round-robin shard selection
	shardIdx := sc.nextShard
	sc.nextShard = (sc.nextShard + 1) % sc.numShards

	// Insert into shard (shard allocates its own local ID)
	localID, err := sc.shards[shardIdx].Insert(ctx, vector, data, meta)
	if err != nil {
		return 0, err
	}

	// Return global ID with shard encoded
	return uint64(NewGlobalID(shardIdx, localID)), nil
}

// BatchInsert adds multiple vectors+payloads+metadata.
//
// Items are distributed across shards in round-robin fashion for balanced load.
// Returns GlobalIDs that encode shard routing information.
func (sc *ShardedCoordinator[T]) BatchInsert(ctx context.Context, vectors [][]float32, dataSlice []T, metadataSlice []metadata.Metadata) ([]uint64, error) {
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
		ids      []uint64 // local IDs from shard
		err      error
	}
	resultsCh := make(chan shardResult, sc.numShards)

	for idx, batch := range shardBatches {
		if len(batch.vectors) == 0 {
			resultsCh <- shardResult{shardIdx: idx, ids: []uint64{}, err: nil}
			continue
		}
		go func(shardIdx int, b *shardBatch) {
			ids, err := sc.shards[shardIdx].BatchInsert(ctx, b.vectors, b.dataSlice, b.metadata)
			resultsCh <- shardResult{shardIdx: shardIdx, ids: ids, err: err}
		}(idx, batch)
	}

	// Collect results indexed by shard
	allIDs := make([]uint64, len(vectors))
	shardIDsMap := make(map[int][]uint64)
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
			allIDs[originalIdx] = uint64(NewGlobalID(shardIdx, localIDs[i]))
		}
	}

	return allIDs, nil
}

// Update updates a vector+payload+metadata atomically in the appropriate shard.
// The shard is determined by extracting the shard index from the GlobalID.
func (sc *ShardedCoordinator[T]) Update(ctx context.Context, id uint64, vector []float32, data T, meta metadata.Metadata) error {
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
func (sc *ShardedCoordinator[T]) Delete(ctx context.Context, id uint64) error {
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
func (sc *ShardedCoordinator[T]) Get(id uint64) (T, bool) {
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
func (sc *ShardedCoordinator[T]) GetMetadata(id uint64) (metadata.Metadata, bool) {
	shard, err := sc.shardForID(id)
	if err != nil {
		return nil, false
	}
	gid := GlobalID(id)
	return shard.GetMetadata(gid.LocalID())
}

type shardResult struct {
	shardIdx int
	results  []index.SearchResult
	err      error
}

// KNNSearch performs a K-nearest neighbor search across all shards in parallel.
func (sc *ShardedCoordinator[T]) KNNSearch(ctx context.Context, query []float32, k int, opts *index.SearchOptions) ([]index.SearchResult, error) {
	var results []index.SearchResult
	if err := sc.KNNSearchWithBuffer(ctx, query, k, opts, &results); err != nil {
		return nil, err
	}
	return results, nil
}

// KNNSearchWithBuffer performs approximate K-nearest neighbor search and appends results to the provided buffer.
func (sc *ShardedCoordinator[T]) KNNSearchWithBuffer(ctx context.Context, query []float32, k int, opts *index.SearchOptions, buf *[]index.SearchResult) error {
	if k <= 0 {
		return index.ErrInvalidK
	}

	resultsCh := make(chan shardResult, sc.numShards)

	for i := 0; i < sc.numShards; i++ {
		shardIdx := i
		shard := sc.shards[i]

		// Wrap filter to convert local IDs to global IDs before checking
		shardOpts := opts
		if opts != nil && opts.Filter != nil {
			userFilter := opts.Filter
			shardOpts = &index.SearchOptions{
				EFSearch: opts.EFSearch,
				Filter: func(localID uint64) bool {
					globalID := uint64(NewGlobalID(shardIdx, localID))
					return userFilter(globalID)
				},
			}
		}

		err := sc.workerPool.Submit(ctx, func() {
			res, err := shard.KNNSearch(ctx, query, k, shardOpts)
			select {
			case resultsCh <- shardResult{shardIdx: shardIdx, results: res, err: err}:
			case <-ctx.Done():
			}
		})

		if err != nil {
			return fmt.Errorf("worker pool submit failed: %w", err)
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
					res.results[j].ID = uint64(NewGlobalID(res.shardIdx, uint64(res.results[j].ID)))
				}
				allResults = append(allResults, res.results...)
			}
		case <-ctx.Done():
			return fmt.Errorf("search cancelled: %w", ctx.Err())
		}
	}

	if len(errors) > 0 {
		return fmt.Errorf("parallel search failed (%d/%d shards): %v", len(errors), sc.numShards, errors)
	}

	merged := mergeTopK(allResults, k)
	*buf = append(*buf, merged...)
	return nil
}

// KNNSearchWithContext performs approximate K-nearest neighbor search using the provided Searcher context.
func (sc *ShardedCoordinator[T]) KNNSearchWithContext(ctx context.Context, query []float32, k int, opts *index.SearchOptions, s *searcher.Searcher) error {
	if k <= 0 {
		return index.ErrInvalidK
	}

	resultsCh := make(chan shardResult, sc.numShards)

	for i := 0; i < sc.numShards; i++ {
		shardIdx := i
		shard := sc.shards[i]

		err := sc.workerPool.Submit(ctx, func() {
			// Acquire local searcher for this shard
			localS := searcher.AcquireSearcher(10000, len(query)) // Initial guess, will grow if needed
			defer searcher.ReleaseSearcher(localS)

			err := shard.KNNSearchWithContext(ctx, query, k, opts, localS)

			var res []index.SearchResult
			if err == nil {
				// Extract results from localS.Candidates
				res = make([]index.SearchResult, 0, localS.Candidates.Len())
				for localS.Candidates.Len() > 0 {
					item, _ := localS.Candidates.PopItem()
					res = append(res, index.SearchResult{ID: item.Node, Distance: item.Distance})
				}
			}

			select {
			case resultsCh <- shardResult{shardIdx: shardIdx, results: res, err: err}:
			case <-ctx.Done():
			}
		})

		if err != nil {
			return fmt.Errorf("worker pool submit failed: %w", err)
		}
	}

	// Collect results
	var errors []error
	for i := 0; i < sc.numShards; i++ {
		select {
		case res := <-resultsCh:
			if res.err != nil {
				errors = append(errors, fmt.Errorf("shard %d: %w", res.shardIdx, res.err))
			} else {
				// Convert local IDs to global IDs and push to s.Candidates
				for _, r := range res.results {
					globalID := uint64(NewGlobalID(res.shardIdx, r.ID))
					s.Candidates.PushItemBounded(searcher.PriorityQueueItem{Node: globalID, Distance: r.Distance}, k)
				}
			}
		case <-ctx.Done():
			return fmt.Errorf("search cancelled: %w", ctx.Err())
		}
	}

	if len(errors) > 0 {
		return fmt.Errorf("parallel search failed (%d/%d shards): %v", len(errors), sc.numShards, errors)
	}

	return nil
}

// BruteSearch performs a brute-force search across all shards in parallel.
func (sc *ShardedCoordinator[T]) BruteSearch(ctx context.Context, query []float32, k int, filter func(id uint64) bool) ([]index.SearchResult, error) {
	if k <= 0 {
		return nil, index.ErrInvalidK
	}

	resultsCh := make(chan shardResult, sc.numShards)

	for i := 0; i < sc.numShards; i++ {
		shardIdx := i
		shard := sc.shards[i]

		var localFilter func(id uint64) bool
		if filter != nil {
			localFilter = func(localID uint64) bool {
				globalID := uint64(NewGlobalID(shardIdx, localID))
				return filter(globalID)
			}
		}

		err := sc.workerPool.Submit(ctx, func() {
			res, err := shard.BruteSearch(ctx, query, k, localFilter)
			select {
			case resultsCh <- shardResult{shardIdx: shardIdx, results: res, err: err}:
			case <-ctx.Done():
			}
		})

		if err != nil {
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
					res.results[j].ID = uint64(NewGlobalID(res.shardIdx, uint64(res.results[j].ID)))
				}
				allResults = append(allResults, res.results...)
			}
		case <-ctx.Done():
			return nil, fmt.Errorf("brute search cancelled: %w", ctx.Err())
		}
	}

	if len(errors) > 0 {
		return nil, fmt.Errorf("parallel brute search failed (%d/%d shards): %v", len(errors), sc.numShards, errors)
	}

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

// EnableProductQuantization enables Product Quantization (PQ) on all shards.
func (sc *ShardedCoordinator[T]) EnableProductQuantization(cfg index.ProductQuantizationConfig) error {
	var errs []error
	for i, shard := range sc.shards {
		if err := shard.EnableProductQuantization(cfg); err != nil {
			errs = append(errs, fmt.Errorf("shard %d: %w", i, err))
		}
	}
	if len(errs) > 0 {
		return fmt.Errorf("failed to enable PQ on %d shards: %v", len(errs), errs)
	}
	return nil
}

// DisableProductQuantization disables Product Quantization (PQ) on all shards.
func (sc *ShardedCoordinator[T]) DisableProductQuantization() {
	for _, shard := range sc.shards {
		shard.DisableProductQuantization()
	}
}

// Close closes all shards and releases their resources.
// This method ensures proper cleanup of all shard resources, including
// the worker pool and any background workers in the underlying indexes.
//
// Shutdown order:
//  1. Close worker pool (waits for in-flight searches to complete)
//  2. Close all shards (durability, indexes, stores)
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
		if err := shard.Close(); err != nil {
			errs = append(errs, fmt.Errorf("shard %d close: %w", i, err))
		}
	}
	if len(errs) > 0 {
		return fmt.Errorf("sharded close errors: %v", errs)
	}
	return nil
}

// HybridSearch performs a hybrid search across all shards.
func (sc *ShardedCoordinator[T]) HybridSearch(ctx context.Context, query []float32, k int, opts *HybridSearchOptions) ([]index.SearchResult, error) {
	// Fan out to all shards
	results := make([][]index.SearchResult, sc.numShards)
	errs := make([]error, sc.numShards)
	var wg sync.WaitGroup

	for i := 0; i < sc.numShards; i++ {
		wg.Add(1)
		// Use worker pool if available, otherwise goroutine
		// But worker pool is for KNNSearch. Can we reuse it?
		// WorkerPool task is specific.
		// Let's just use goroutines for now, or extend WorkerPool.
		// Given "Breaking changes are no problem", I'll just use goroutines for simplicity.
		go func(shardIdx int) {
			defer wg.Done()
			res, err := sc.shards[shardIdx].HybridSearch(ctx, query, k, opts)
			if err == nil {
				// Convert local IDs to global IDs
				for j := range res {
					res[j].ID = uint64(NewGlobalID(shardIdx, res[j].ID))
				}
			}
			results[shardIdx] = res
			errs[shardIdx] = err
		}(i)
	}
	wg.Wait()

	// Check errors
	for _, err := range errs {
		if err != nil {
			return nil, err
		}
	}

	// Merge results
	// We can reuse the merge logic from KNNSearch if we extract it.
	// But for now, duplicate it or use a simple merge.
	// Since k is usually small, a simple merge is fine.
	merged := make([]index.SearchResult, 0, k*sc.numShards)
	for _, res := range results {
		merged = append(merged, res...)
	}

	// Sort and take top k
	// Note: This is not as efficient as a heap merge, but correct.
	// Optimization: Use heap merge.
	// Let's use the existing merge logic if possible.
	// But KNNSearch logic is embedded.
	// I'll implement a simple sort here.
	// index.SearchResult doesn't implement sort.Interface.
	// I need to sort manually.
	// Actually, index package has sort helpers? No.
	// I'll use sort.Slice.
	// Wait, I need to import "sort".
	// Or I can use the heap logic from KNNSearch if I copy it.
	// I'll use a simple insertion sort or selection sort if k is small, or sort.Slice.
	// I'll assume sort package is available or add it.
	// I'll add "sort" to imports.

	// For now, let's just return the merged results sorted.
	// I'll add "sort" to imports in a separate step if needed.
	// Or I can use a simple bubble sort if k is very small? No.
	// I'll use a heap.

	return sc.mergeResults(results, k), nil
}

// mergeResults merges sorted results from shards into a single sorted result.
func (sc *ShardedCoordinator[T]) mergeResults(shardResults [][]index.SearchResult, k int) []index.SearchResult {
	total := 0
	for _, res := range shardResults {
		total += len(res)
	}
	flat := make([]index.SearchResult, 0, total)
	for _, res := range shardResults {
		flat = append(flat, res...)
	}

	if len(flat) == 0 {
		return nil
	}

	// Sort by distance
	sort.Slice(flat, func(i, j int) bool {
		return flat[i].Distance < flat[j].Distance
	})

	// Take top k
	if len(flat) > k {
		return flat[:k]
	}
	return flat
}

// KNNSearchStream returns an iterator over K-nearest neighbor search results.
func (sc *ShardedCoordinator[T]) KNNSearchStream(ctx context.Context, query []float32, k int, opts *index.SearchOptions) iter.Seq2[index.SearchResult, error] {
	// For sharded streaming, we collect all results and yield them.
	// True streaming merge is complex.
	return func(yield func(index.SearchResult, error) bool) {
		results, err := sc.KNNSearch(ctx, query, k, opts)
		if err != nil {
			yield(index.SearchResult{}, err)
			return
		}
		for _, res := range results {
			if !yield(res, nil) {
				return
			}
		}
	}
}

// SaveToWriter is not supported for sharded coordinator.
func (sc *ShardedCoordinator[T]) SaveToWriter(w io.Writer) error {
	return fmt.Errorf("SaveToWriter not supported for sharded coordinator (use SaveToFile)")
}

// SaveToFile saves each shard to a subdirectory.
func (sc *ShardedCoordinator[T]) SaveToFile(path string) error {
	if err := os.MkdirAll(path, 0755); err != nil {
		return err
	}
	for i, shard := range sc.shards {
		shardPath := filepath.Join(path, fmt.Sprintf("shard-%d", i))
		if err := shard.SaveToFile(shardPath); err != nil {
			return fmt.Errorf("shard %d save: %w", i, err)
		}
	}
	return nil
}

// RecoverFromWAL recovers each shard from its WAL.
func (sc *ShardedCoordinator[T]) RecoverFromWAL(ctx context.Context) error {
	var wg sync.WaitGroup
	errCh := make(chan error, len(sc.shards))

	for i, shard := range sc.shards {
		wg.Add(1)
		go func(idx int, s Coordinator[T]) {
			defer wg.Done()
			if err := s.RecoverFromWAL(ctx); err != nil {
				errCh <- fmt.Errorf("shard %d recovery: %w", idx, err)
			}
		}(i, shard)
	}

	wg.Wait()
	close(errCh)

	// Return the first error encountered
	if err := <-errCh; err != nil {
		return err
	}
	return nil
}

// Stats returns statistics from the first shard (primary).
func (sc *ShardedCoordinator[T]) Stats() index.Stats {
	if len(sc.shards) > 0 {
		return sc.shards[0].Stats()
	}
	return index.Stats{}
}

// Checkpoint creates a checkpoint in all shards.
func (sc *ShardedCoordinator[T]) Checkpoint() error {
	var wg sync.WaitGroup
	errCh := make(chan error, len(sc.shards))

	for i, shard := range sc.shards {
		wg.Add(1)
		go func(idx int, s Coordinator[T]) {
			defer wg.Done()
			if err := s.Checkpoint(); err != nil {
				errCh <- fmt.Errorf("shard %d checkpoint: %w", idx, err)
			}
		}(i, shard)
	}

	wg.Wait()
	close(errCh)

	// Return the first error encountered
	if err := <-errCh; err != nil {
		return err
	}
	return nil
}
