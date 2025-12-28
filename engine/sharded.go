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
// partitioning vectors across shards based on ID hashing.
//
// Design:
//   - Hash-based sharding: shard = id % numShards
//   - Each shard is a complete Tx[T] with its own lock
//   - Write operations route to a single shard (shard-level lock only)
//   - Search operations fan out to all shards in parallel and merge results
//
// Performance Impact:
//   - Target: 3-8x write speedup on multi-core systems
//   - Search: Parallel fan-out maintains low latency
//
// Backward Compatibility:
//   - Implements same Coordinator[T] interface as Tx[T]
//   - Drop-in replacement for single-coordinator mode
type ShardedCoordinator[T any] struct {
	shards    []*Tx[T]
	numShards int
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
//
// Example:
//
//	indexes := make([]index.Index, 4)
//	for i := range indexes {
//	    indexes[i], _ = hnsw.NewSharded(i, 4, hnsw.WithDimension(128))
//	}
//	dataStores := ...  // create 4 data stores
//	metaStores := ...  // create 4 metadata stores
//	coord, _ := engine.NewSharded(indexes, dataStores, metaStores, nil, nil)
func NewSharded[T any](
	indexes []index.Index,
	dataStores []Store[T],
	metaStores []*metadata.UnifiedIndex,
	durabilities []Durability,
	codec codec.Codec,
) (*ShardedCoordinator[T], error) {
	numShards := len(indexes)
	if numShards == 0 {
		return nil, fmt.Errorf("sharded: at least one shard required")
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
		tx, err := New(indexes[i], dataStores[i], metaStores[i], durabilities[i], codec)
		if err != nil {
			return nil, fmt.Errorf("failed to create shard %d: %w", i, err)
		}
		shards[i] = tx
	}

	return &ShardedCoordinator[T]{
		shards:    shards,
		numShards: numShards,
	}, nil
}

// shardForID returns the shard that owns the given ID.
// Uses hash-based partitioning: shard = id % numShards
func (sc *ShardedCoordinator[T]) shardForID(id uint32) *Tx[T] {
	return sc.shards[int(id)%sc.numShards]
}

// Insert adds a vector+payload+metadata atomically to a shard.
//
// The shard is selected using a global counter (sum of vector counts) modulo numShards.
// This ensures balanced distribution across shards while maintaining HNSW's
// requirement that each shard's first insert gets ID=0.
//
// Each shard allocates its own local IDs (0, 1, 2, ...) independently.
func (sc *ShardedCoordinator[T]) Insert(ctx context.Context, vector []float32, data T, meta metadata.Metadata) (uint32, error) {
	// Simple round-robin: use the sum of all shards' vector counts
	total := 0
	for _, shard := range sc.shards {
		if shardIndex, ok := shard.txIndex.(index.Shard); ok {
			total += shardIndex.VectorCount()
		}
	}
	shardIdx := total % sc.numShards

	// Insert into that shard (shard allocates its own local ID)
	return sc.shards[shardIdx].Insert(ctx, vector, data, meta)
}

// BatchInsert adds multiple vectors+payloads+metadata.
//
// Items are distributed across shards in round-robin fashion for balanced load.
// Each shard uses its own local ID space (0, 1, 2, ...).
//
// Returns the list of IDs assigned (one per input vector). IDs are local to each shard.
func (sc *ShardedCoordinator[T]) BatchInsert(ctx context.Context, vectors [][]float32, dataSlice []T, metadataSlice []metadata.Metadata) ([]uint32, error) {
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

	// Distribute items round-robin
	for i := range vectors {
		shardIdx := i % sc.numShards
		batch := shardBatches[shardIdx]
		batch.indices = append(batch.indices, i)
		batch.vectors = append(batch.vectors, vectors[i])
		batch.dataSlice = append(batch.dataSlice, dataSlice[i])
		if metadataSlice != nil && i < len(metadataSlice) {
			batch.metadata = append(batch.metadata, metadataSlice[i])
		}
	}

	// Insert batches into each shard in parallel
	type shardResult struct {
		shardIdx int
		ids      []uint32
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

	// Map shard-local IDs back to original order
	for shardIdx, batch := range shardBatches {
		shardIDs := shardIDsMap[shardIdx]
		for i, originalIdx := range batch.indices {
			allIDs[originalIdx] = shardIDs[i]
		}
	}

	return allIDs, nil
}

// Update updates a vector+payload+metadata atomically in the appropriate shard.
func (sc *ShardedCoordinator[T]) Update(ctx context.Context, id uint32, vector []float32, data T, meta metadata.Metadata) error {
	shard := sc.shardForID(id)
	return shard.Update(ctx, id, vector, data, meta)
}

// Delete removes a vector+payload+metadata atomically from the appropriate shard.
func (sc *ShardedCoordinator[T]) Delete(ctx context.Context, id uint32) error {
	shard := sc.shardForID(id)
	return shard.Delete(ctx, id)
}

// KNNSearch performs a K-nearest neighbor search across all shards in parallel.
//
// This method fans out the query to all shards simultaneously, collects results,
// and merges them to return the global top-k results. This ensures that sharded
// mode returns the same results as non-sharded mode (with similar recall).
//
// Performance:
//   - Parallel fan-out utilizes multiple cores for search
//   - Latency is roughly the same as single-shard (limited by slowest shard)
//   - Throughput can improve with concurrent searches
func (sc *ShardedCoordinator[T]) KNNSearch(ctx context.Context, query []float32, k int, opts *index.SearchOptions) ([]index.SearchResult, error) {
	if k <= 0 {
		return nil, index.ErrInvalidK
	}

	// Fan out to all shards in parallel
	type shardResult struct {
		results []index.SearchResult
		err     error
	}
	resultsCh := make(chan shardResult, sc.numShards)

	for i := 0; i < sc.numShards; i++ {
		go func(shardIdx int) {
			results, err := sc.shards[shardIdx].KNNSearch(ctx, query, k, opts)
			resultsCh <- shardResult{results: results, err: err}
		}(i)
	}

	// Collect results from all shards
	allResults := make([]index.SearchResult, 0, k*sc.numShards)
	for i := 0; i < sc.numShards; i++ {
		res := <-resultsCh
		if res.err != nil {
			return nil, res.err
		}
		allResults = append(allResults, res.results...)
	}

	// Merge results to get global top-k
	return mergeTopK(allResults, k), nil
}

// BruteSearch performs a brute-force search across all shards in parallel.
//
// Similar to KNNSearch, this fans out to all shards and merges results.
func (sc *ShardedCoordinator[T]) BruteSearch(ctx context.Context, query []float32, k int, filter func(id uint32) bool) ([]index.SearchResult, error) {
	if k <= 0 {
		return nil, index.ErrInvalidK
	}

	// Fan out to all shards in parallel
	type shardResult struct {
		results []index.SearchResult
		err     error
	}
	resultsCh := make(chan shardResult, sc.numShards)

	for i := 0; i < sc.numShards; i++ {
		go func(shardIdx int) {
			results, err := sc.shards[shardIdx].BruteSearch(ctx, query, k, filter)
			resultsCh <- shardResult{results: results, err: err}
		}(i)
	}

	// Collect results from all shards
	allResults := make([]index.SearchResult, 0, k*sc.numShards)
	for i := 0; i < sc.numShards; i++ {
		res := <-resultsCh
		if res.err != nil {
			return nil, res.err
		}
		allResults = append(allResults, res.results...)
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
