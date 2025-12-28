// Package flat provides an implementation of a flat index for vector storage and search.
package flat

import (
	"container/heap"
	"context"
	"fmt"
	"iter"
	"sync"
	"sync/atomic"

	"github.com/hupe1980/vecgo/distance"
	"github.com/hupe1980/vecgo/index"
	"github.com/hupe1980/vecgo/internal/queue"
	"github.com/hupe1980/vecgo/quantization"
	"github.com/hupe1980/vecgo/vectorstore"
	"github.com/hupe1980/vecgo/vectorstore/columnar"
)

// Compile-time checks to ensure Flat satisfies required interfaces.
var _ index.Index = (*Flat)(nil)
var _ index.TransactionalIndex = (*Flat)(nil)
var _ index.ProductQuantizationEnabler = (*Flat)(nil)

// Node represents a node in the flat index with its vector and unique identifier.
type Node struct {
	ID uint32 // Unique identifier
}

// Options contains configuration options for the flat index.
type Options struct {
	// Dimension is the fixed vector dimensionality for this index.
	// It must be > 0 and is enforced for all inserts/updates/searches.
	Dimension int

	// DistanceType represents the type of distance function used for calculating distances between vectors.
	DistanceType index.DistanceType

	// NormalizeVectors enables L2 normalization for stored vectors and queries.
	// Commonly used for cosine search.
	NormalizeVectors bool
}

// DefaultOptions contains the default configuration options for the flat index.
var DefaultOptions = Options{
	Dimension:    0,
	DistanceType: index.DistanceTypeSquaredL2,
}

// indexState holds the immutable state of the index for lock-free reads.
type indexState struct {
	nodes    []*Node  // Nodes in the index (nil entries are tombstones)
	freeList []uint32 // IDs available for reuse from deleted nodes
	// pqCodes stores encoded PQ codes indexed by vector ID.
	// It is nil when PQ is disabled.
	pqCodes [][]byte
}

// Flat represents a flat index for vector storage and search.
// It uses a copy-on-write pattern for lock-free concurrent reads.
type Flat struct {
	state        atomic.Value // holds *indexState for lock-free reads
	writeMu      sync.Mutex   // Serializes writes only
	dimension    atomic.Int32 // Dimension of vectors (lock-free)
	distanceFunc index.DistanceFunc
	pq           atomic.Pointer[quantization.ProductQuantizer]
	opts         Options // Options for the index
	vectors      vectorstore.Store

	// Sharding support (immutable after construction)
	shardID   int // Which shard this instance represents (0-based)
	numShards int // Total number of shards (1 = no sharding)
}

func (*Flat) Name() string { return "Flat" }

// AllocateID reserves a new ID from the free list or by extending the nodes slice.
// The reserved slot remains nil until ApplyInsert is called.
func (f *Flat) AllocateID() uint32 {
	f.writeMu.Lock()
	defer f.writeMu.Unlock()

	oldState := f.getState()
	newState := f.cloneState(oldState)

	var id uint32
	if len(newState.freeList) > 0 {
		id = newState.freeList[len(newState.freeList)-1]
		newState.freeList = newState.freeList[:len(newState.freeList)-1]
	} else {
		id = uint32(len(newState.nodes))
		newState.nodes = append(newState.nodes, nil)
		if newState.pqCodes != nil {
			newState.pqCodes = append(newState.pqCodes, nil)
		}
	}

	f.state.Store(newState)
	return id
}

// ReleaseID returns a previously allocated but unused ID back to the free list.
func (f *Flat) ReleaseID(id uint32) {
	f.writeMu.Lock()
	defer f.writeMu.Unlock()

	oldState := f.getState()
	if int(id) >= len(oldState.nodes) {
		return
	}
	if oldState.nodes[id] != nil {
		// Already inserted; caller must not release.
		return
	}

	newState := f.cloneState(oldState)
	// Avoid obvious duplicates (best-effort; ReleaseID is not on the hot path).
	for _, free := range newState.freeList {
		if free == id {
			f.state.Store(newState)
			return
		}
	}
	newState.freeList = append(newState.freeList, id)
	f.state.Store(newState)
}

// New creates a new instance of the flat index.
// Dimension and DistanceType are required and must be set at creation time.
func New(optFns ...func(o *Options)) (*Flat, error) {
	opts := DefaultOptions

	for _, fn := range optFns {
		fn(&opts)
	}

	// Validate basic options using shared helper
	if err := index.ValidateBasicOptions(opts.Dimension, opts.DistanceType); err != nil {
		return nil, err
	}

	if opts.DistanceType == index.DistanceTypeCosine {
		// Match common vector-store behavior: cosine is implemented via L2-normalized vectors.
		opts.NormalizeVectors = true
	}

	f := &Flat{
		distanceFunc: index.NewDistanceFunc(opts.DistanceType),
		opts:         opts,
		vectors:      columnar.New(opts.Dimension),
		shardID:      0, // Default: shard 0 (non-sharded mode)
		numShards:    1, // Default: 1 shard (non-sharded mode)
	}
	f.dimension.Store(int32(opts.Dimension))

	// Initialize state with empty slices
	f.state.Store(&indexState{
		nodes:    make([]*Node, 0),
		freeList: make([]uint32, 0),
	})

	return f, nil
}

// getState returns the current immutable state (lock-free read).
func (f *Flat) getState() *indexState {
	return f.state.Load().(*indexState)
}

// cloneState creates a deep copy of the current state for copy-on-write.
func (f *Flat) cloneState(st *indexState) *indexState {
	newNodes := make([]*Node, len(st.nodes))
	copy(newNodes, st.nodes)

	newFreeList := make([]uint32, len(st.freeList))
	copy(newFreeList, st.freeList)

	var newPQCodes [][]byte
	if st.pqCodes != nil {
		newPQCodes = make([][]byte, len(st.pqCodes))
		copy(newPQCodes, st.pqCodes)
	}

	return &indexState{
		nodes:    newNodes,
		freeList: newFreeList,
		pqCodes:  newPQCodes,
	}
}

func (f *Flat) ProductQuantizationEnabled() bool {
	return f.pq.Load() != nil
}

func (f *Flat) DisableProductQuantization() {
	f.writeMu.Lock()
	defer f.writeMu.Unlock()

	f.pq.Store(nil)
	st := f.getState()
	newState := f.cloneState(st)
	newState.pqCodes = nil
	f.state.Store(newState)
}

func (f *Flat) EnableProductQuantization(cfg index.ProductQuantizationConfig) error {
	if cfg.NumSubvectors <= 0 {
		return fmt.Errorf("flat: invalid NumSubvectors: %d", cfg.NumSubvectors)
	}
	if cfg.NumCentroids <= 0 {
		return fmt.Errorf("flat: invalid NumCentroids: %d", cfg.NumCentroids)
	}
	switch f.opts.DistanceType {
	case index.DistanceTypeSquaredL2:
		// ok
	case index.DistanceTypeCosine:
		if !f.opts.NormalizeVectors {
			return fmt.Errorf("flat: PQ for cosine requires NormalizeVectors")
		}
	default:
		return fmt.Errorf("flat: PQ only supported for SquaredL2 or Cosine")
	}

	f.writeMu.Lock()
	defer f.writeMu.Unlock()

	st := f.getState()
	vectors := make([][]float32, 0, len(st.nodes)-len(st.freeList))
	for _, n := range st.nodes {
		if n != nil {
			v, ok := f.vectors.GetVector(n.ID)
			if ok {
				vectors = append(vectors, v)
			}
		}
	}
	if len(vectors) == 0 {
		return fmt.Errorf("flat: cannot enable PQ with no vectors")
	}

	dim := int(f.dimension.Load())
	pq, err := quantization.NewProductQuantizer(dim, cfg.NumSubvectors, cfg.NumCentroids)
	if err != nil {
		return err
	}
	if err := pq.Train(vectors); err != nil {
		return err
	}

	newState := f.cloneState(st)
	newState.pqCodes = make([][]byte, len(newState.nodes))
	for _, n := range newState.nodes {
		if n == nil {
			continue
		}
		v, ok := f.vectors.GetVector(n.ID)
		if !ok {
			continue
		}
		newState.pqCodes[n.ID] = pq.Encode(v)
	}

	// IMPORTANT: Publish state first, then pq pointer.
	// This guarantees pqCodes exist before pq is visible to readers.
	f.state.Store(newState)
	f.pq.Store(pq)
	return nil
}

// Insert inserts a vector into the flat index.
func (f *Flat) Insert(ctx context.Context, v []float32) (uint32, error) {
	// Check for cancellation
	if err := ctx.Err(); err != nil {
		return 0, err
	}

	if len(v) == 0 {
		return 0, index.ErrEmptyVector
	}

	f.writeMu.Lock()
	defer f.writeMu.Unlock()

	// Get configured dimension (atomic read)
	currentDim := int(f.dimension.Load())

	// Check if dimensions match
	if len(v) != currentDim {
		return 0, &index.ErrDimensionMismatch{Expected: currentDim, Actual: len(v)}
	}

	vec := v
	if f.opts.NormalizeVectors {
		norm, ok := distance.NormalizeL2Copy(v)
		if !ok {
			return 0, fmt.Errorf("flat: cannot normalize zero vector")
		}
		vec = norm
	}

	// Make a copy of the vector to ensure changes outside this function don't affect the node
	if f.vectors == nil {
		return 0, fmt.Errorf("flat: vector store not configured")
	}

	// Get current state and clone for modification
	oldState := f.getState()
	newState := f.cloneState(oldState)
	pq := f.pq.Load()

	// Reuse ID from free list if available, otherwise allocate new
	var id uint32
	if len(newState.freeList) > 0 {
		// Pop from free list
		id = newState.freeList[len(newState.freeList)-1]
		newState.freeList = newState.freeList[:len(newState.freeList)-1]
		// Reuse slot
		if err := f.vectors.SetVector(id, vec); err != nil {
			return 0, err
		}
		newState.nodes[id] = &Node{ID: id}
		if pq != nil {
			if newState.pqCodes == nil {
				newState.pqCodes = make([][]byte, len(newState.nodes))
			}
			newState.pqCodes[id] = pq.Encode(vec)
		}
	} else {
		// Allocate new ID
		id = uint32(len(newState.nodes))
		if err := f.vectors.SetVector(id, vec); err != nil {
			return 0, err
		}
		newState.nodes = append(newState.nodes, &Node{ID: id})
		if pq != nil {
			if newState.pqCodes == nil {
				newState.pqCodes = make([][]byte, 0, len(newState.nodes))
			}
			for len(newState.pqCodes) < len(newState.nodes)-1 {
				newState.pqCodes = append(newState.pqCodes, nil)
			}
			newState.pqCodes = append(newState.pqCodes, pq.Encode(vec))
		}
	}

	// Atomic swap to new state
	f.state.Store(newState)

	return id, nil
}

// ApplyInsert inserts a vector at an explicit ID.
// This is intended for deterministic WAL replay.
func (f *Flat) ApplyInsert(ctx context.Context, id uint32, v []float32) error {
	// Check for cancellation
	if err := ctx.Err(); err != nil {
		return err
	}

	if len(v) == 0 {
		return index.ErrEmptyVector
	}

	f.writeMu.Lock()
	defer f.writeMu.Unlock()

	currentDim := int(f.dimension.Load())
	if len(v) != currentDim {
		return &index.ErrDimensionMismatch{Expected: currentDim, Actual: len(v)}
	}

	vec := v
	if f.opts.NormalizeVectors {
		norm, ok := distance.NormalizeL2Copy(v)
		if !ok {
			return fmt.Errorf("flat: cannot normalize zero vector")
		}
		vec = norm
	}

	if f.vectors == nil {
		return fmt.Errorf("flat: vector store not configured")
	}

	oldState := f.getState()
	newState := f.cloneState(oldState)
	pq := f.pq.Load()

	// Ensure nodes slice can hold id.
	if int(id) >= len(newState.nodes) {
		oldLen := len(newState.nodes)
		newLen := int(id) + 1
		newState.nodes = append(newState.nodes, make([]*Node, newLen-oldLen)...)
		if pq != nil {
			if newState.pqCodes == nil {
				newState.pqCodes = make([][]byte, oldLen)
			}
			newState.pqCodes = append(newState.pqCodes, make([][]byte, newLen-oldLen)...)
		}
		// NOTE: Do NOT auto-generate freeList entries for gaps during WAL replay.
		// WAL replay may be out of order; IDs that appear as gaps now may be
		// inserted later. Only reuse IDs explicitly deleted in WAL.
	}

	if newState.nodes[id] != nil {
		return fmt.Errorf("node %d already exists", id)
	}

	// Remove id from free list if present.
	for i := 0; i < len(newState.freeList); i++ {
		if newState.freeList[i] == id {
			newState.freeList[i] = newState.freeList[len(newState.freeList)-1]
			newState.freeList = newState.freeList[:len(newState.freeList)-1]
			break
		}
	}

	if err := f.vectors.SetVector(id, vec); err != nil {
		return err
	}
	newState.nodes[id] = &Node{ID: id}
	if pq != nil {
		if newState.pqCodes == nil {
			newState.pqCodes = make([][]byte, len(newState.nodes))
		}
		newState.pqCodes[id] = pq.Encode(vec)
	}
	f.state.Store(newState)
	return nil
}

// ApplyUpdate updates a vector at an explicit ID (deterministic WAL replay helper).
func (f *Flat) ApplyUpdate(ctx context.Context, id uint32, v []float32) error {
	return f.Update(ctx, id, v)
}

// ApplyDelete deletes a vector at an explicit ID (deterministic WAL replay helper).
func (f *Flat) ApplyDelete(ctx context.Context, id uint32) error {
	return f.Delete(ctx, id)
}

// VectorByID returns the vector stored for the given ID.
// WARNING: The returned slice may alias internal memory (especially for mmap stores).
// Callers should not modify the returned slice. Make a copy if mutation is needed.
func (f *Flat) VectorByID(ctx context.Context, id uint32) ([]float32, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	st := f.getState()
	if int(id) >= len(st.nodes) {
		return nil, &index.ErrNodeNotFound{ID: id}
	}
	n := st.nodes[id]
	if n == nil {
		return nil, &index.ErrNodeDeleted{ID: id}
	}
	if f.vectors == nil {
		return nil, fmt.Errorf("flat: vector store not configured")
	}
	v, ok := f.vectors.GetVector(id)
	if !ok {
		return nil, &index.ErrNodeNotFound{ID: id}
	}
	// Note: The returned slice may alias internal/mmap memory.
	// This is intentional for zero-copy mmap performance.
	// Callers must treat the slice as read-only.
	return v, nil
}

// BatchInsert inserts multiple vectors into the flat index in a single operation.
// This is more efficient than calling Insert multiple times as it acquires the lock once.
// Note: BatchInsert reuses IDs from the freeList before allocating new IDs.
func (f *Flat) BatchInsert(ctx context.Context, vectors [][]float32) index.BatchInsertResult {
	result := index.BatchInsertResult{
		IDs:    make([]uint32, len(vectors)),
		Errors: make([]error, len(vectors)),
	}

	if len(vectors) == 0 {
		return result
	}

	// Check for cancellation
	if err := ctx.Err(); err != nil {
		for i := range result.Errors {
			result.Errors[i] = err
		}
		return result
	}

	f.writeMu.Lock()
	defer f.writeMu.Unlock()

	// Get configured dimension (atomic read)
	currentDim := int(f.dimension.Load())

	// Get current state and clone for modification
	oldState := f.getState()
	newState := f.cloneState(oldState)
	pq := f.pq.Load()
	if pq != nil && newState.pqCodes == nil {
		newState.pqCodes = make([][]byte, len(newState.nodes))
	}

	// Track next append ID (used after freeList is exhausted)
	nextAppendID := uint32(len(newState.nodes))

	for i, v := range vectors {
		if len(v) == 0 {
			result.Errors[i] = index.ErrEmptyVector
			continue
		}

		if len(v) != currentDim {
			result.Errors[i] = &index.ErrDimensionMismatch{Expected: currentDim, Actual: len(v)}
			continue
		}

		vec := v
		if f.opts.NormalizeVectors {
			norm, ok := distance.NormalizeL2Copy(v)
			if !ok {
				result.Errors[i] = fmt.Errorf("flat: cannot normalize zero vector")
				continue
			}
			vec = norm
		}

		if f.vectors == nil {
			result.Errors[i] = fmt.Errorf("flat: vector store not configured")
			continue
		}

		// Reuse ID from freeList if available, otherwise allocate new
		var id uint32
		if len(newState.freeList) > 0 {
			// Pop from free list
			id = newState.freeList[len(newState.freeList)-1]
			newState.freeList = newState.freeList[:len(newState.freeList)-1]
			// Reuse slot
			if err := f.vectors.SetVector(id, vec); err != nil {
				result.Errors[i] = err
				continue
			}
			newState.nodes[id] = &Node{ID: id}
			if pq != nil {
				newState.pqCodes[id] = pq.Encode(vec)
			}
		} else {
			// Allocate new ID
			id = nextAppendID
			nextAppendID++
			if err := f.vectors.SetVector(id, vec); err != nil {
				result.Errors[i] = err
				continue
			}
			newState.nodes = append(newState.nodes, &Node{ID: id})
			if pq != nil {
				for len(newState.pqCodes) < len(newState.nodes)-1 {
					newState.pqCodes = append(newState.pqCodes, nil)
				}
				newState.pqCodes = append(newState.pqCodes, pq.Encode(vec))
			}
		}
		result.IDs[i] = id
	}

	// Atomic swap to new state
	f.state.Store(newState)

	return result
}

// Delete removes a vector from the flat index by marking it as deleted.
func (f *Flat) Delete(ctx context.Context, id uint32) error {
	// Check for cancellation
	if err := ctx.Err(); err != nil {
		return err
	}

	f.writeMu.Lock()
	defer f.writeMu.Unlock()

	// Get current state and clone for modification
	oldState := f.getState()

	if int(id) >= len(oldState.nodes) {
		return &index.ErrNodeNotFound{ID: id}
	}

	if oldState.nodes[id] == nil {
		return &index.ErrNodeDeleted{ID: id}
	}

	newState := f.cloneState(oldState)

	// Mark as deleted by setting node to nil
	newState.nodes[id] = nil
	if newState.pqCodes != nil && int(id) < len(newState.pqCodes) {
		newState.pqCodes[id] = nil
	}

	// Add ID to free list for reuse (avoid duplicates)
	duplicateInFreeList := false
	for _, free := range newState.freeList {
		if free == id {
			duplicateInFreeList = true
			break
		}
	}
	if !duplicateInFreeList {
		newState.freeList = append(newState.freeList, id)
	}
	if f.vectors != nil {
		_ = f.vectors.DeleteVector(id)
	}

	// Atomic swap to new state
	f.state.Store(newState)

	return nil
}

// Update updates a vector in the flat index.
func (f *Flat) Update(ctx context.Context, id uint32, v []float32) error {
	// Check for cancellation
	if err := ctx.Err(); err != nil {
		return err
	}

	if len(v) == 0 {
		return index.ErrEmptyVector
	}

	f.writeMu.Lock()
	defer f.writeMu.Unlock()

	// Get current dimension (atomic read)
	currentDim := int(f.dimension.Load())

	// Get current state and clone for modification
	oldState := f.getState()

	if int(id) >= len(oldState.nodes) {
		return &index.ErrNodeNotFound{ID: id}
	}

	if oldState.nodes[id] == nil {
		return &index.ErrNodeDeleted{ID: id}
	}

	if len(v) != currentDim {
		return &index.ErrDimensionMismatch{Expected: currentDim, Actual: len(v)}
	}

	vec := v
	if f.opts.NormalizeVectors {
		norm, ok := distance.NormalizeL2Copy(v)
		if !ok {
			return fmt.Errorf("flat: cannot normalize zero vector")
		}
		vec = norm
	}

	if f.vectors == nil {
		return fmt.Errorf("flat: vector store not configured")
	}
	if err := f.vectors.SetVector(id, vec); err != nil {
		return err
	}

	newState := f.cloneState(oldState)
	pq := f.pq.Load()

	// Copy-on-write: replace the node pointer to avoid mutating a Node shared
	// with readers holding the previous immutable state.
	newState.nodes[id] = &Node{ID: id}
	if pq != nil {
		if newState.pqCodes == nil {
			newState.pqCodes = make([][]byte, len(newState.nodes))
		}
		newState.pqCodes[id] = pq.Encode(vec)
	}

	// Atomic swap to new state
	f.state.Store(newState)

	return nil
}

// KNNSearch performs a K-nearest neighbor search in the flat index.
func (f *Flat) KNNSearch(ctx context.Context, q []float32, k int, opts *index.SearchOptions) ([]index.SearchResult, error) {
	// Extract filter from options (Flat doesn't use efSearch)
	var filter func(id uint32) bool
	if opts != nil && opts.Filter != nil {
		filter = opts.Filter
	}
	return f.BruteSearch(ctx, q, k, filter)
}

// KNNSearchStream returns an iterator over K-nearest neighbor search results.
// Results are yielded in order from nearest to farthest.
// The iterator supports early termination - stop iterating to cancel.
//
// Example:
//
//	for result, err := range f.KNNSearchStream(ctx, query, 100, nil) {
//	    if err != nil {
//	        return err
//	    }
//	    if result.Distance > threshold {
//	        break // Early termination
//	    }
//	    process(result)
//	}
func (f *Flat) KNNSearchStream(ctx context.Context, q []float32, k int, opts *index.SearchOptions) iter.Seq2[index.SearchResult, error] {
	return func(yield func(index.SearchResult, error) bool) {
		// Extract filter from options
		var filter func(id uint32) bool
		if opts != nil && opts.Filter != nil {
			filter = opts.Filter
		}

		// Run the search
		results, err := f.BruteSearch(ctx, q, k, filter)
		if err != nil {
			yield(index.SearchResult{}, err)
			return
		}

		// Yield results one at a time
		for _, result := range results {
			if !yield(result, nil) {
				return // Early termination
			}
		}
	}
}

// BruteSearch performs a brute-force search in the flat index.
// This method is lock-free for reads using the copy-on-write pattern.
func (f *Flat) BruteSearch(ctx context.Context, query []float32, k int, filter func(id uint32) bool) ([]index.SearchResult, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	currentState := f.getState()
	currentDim := int(f.dimension.Load())

	if k <= 0 {
		return nil, index.ErrInvalidK
	}
	if len(currentState.nodes) == 0 {
		return nil, nil
	}
	if currentDim > 0 && len(query) != currentDim {
		return nil, &index.ErrDimensionMismatch{Expected: currentDim, Actual: len(query)}
	}

	q := query
	if f.opts.NormalizeVectors {
		norm, ok := distance.NormalizeL2Copy(query)
		if !ok {
			return nil, fmt.Errorf("flat: cannot normalize zero query")
		}
		q = norm
	}

	actualK := k
	if actualK > len(currentState.nodes) {
		actualK = len(currentState.nodes)
	}

	topCandidates := queue.NewMax(actualK)
	heap.Init(topCandidates)

	pq := f.pq.Load()

	// === BATCH DISTANCE OPTIMIZATION ===
	// For non-PQ mode, gather all vectors and compute distances in batch
	// This improves cache locality and SIMD vectorization
	if pq == nil {
		// Gather all valid node IDs and vectors
		batchIDs := make([]uint32, 0, len(currentState.nodes))
		batchVectors := make([][]float32, 0, len(currentState.nodes))

		for _, node := range currentState.nodes {
			if node == nil {
				continue
			}
			if filter != nil && !filter(node.ID) {
				continue
			}

			if f.vectors == nil {
				continue
			}
			vec, ok := f.vectors.GetVector(node.ID)
			if !ok {
				continue
			}

			batchIDs = append(batchIDs, node.ID)
			batchVectors = append(batchVectors, vec)
		}

		// Compute all distances using single-pair SIMD
		if len(batchIDs) > 0 {
			batchDistances := make([]float32, len(batchIDs))

			// Use appropriate function based on distance type
			if f.opts.DistanceType == index.DistanceTypeDotProduct {
				for i, vec := range batchVectors {
					batchDistances[i] = -distance.Dot(q, vec)
				}
			} else {
				// SquaredL2 or Cosine (cosine uses normalized vectors)
				for i, vec := range batchVectors {
					batchDistances[i] = distance.SquaredL2(q, vec)
				}
			}

			// Build heap from batch results
			for i, nodeID := range batchIDs {
				nodeDist := batchDistances[i]

				if topCandidates.Len() < actualK {
					heap.Push(topCandidates, queue.PriorityQueueItem{Node: nodeID, Distance: nodeDist})
					continue
				}

				largest := topCandidates.Top().(queue.PriorityQueueItem)
				if nodeDist < largest.Distance {
					heap.Pop(topCandidates)
					heap.Push(topCandidates, queue.PriorityQueueItem{Node: nodeID, Distance: nodeDist})
				}
			}
		}
	} else {
		// PQ mode: use asymmetric distance (can't batch due to custom codes)
		// NOTE: PQ asymmetric distance approximates L2 distance on normalized vectors.
		// For cosine, vectors are already normalized so L2 ordering is equivalent.
		for _, node := range currentState.nodes {
			if node == nil {
				continue
			}
			if filter != nil && !filter(node.ID) {
				continue
			}

			var nodeDist float32
			if currentState.pqCodes != nil && int(node.ID) < len(currentState.pqCodes) {
				if code := currentState.pqCodes[node.ID]; code != nil {
					// PQ distance is already squared L2; no scaling needed.
					// For cosine, vectors were normalized on insert, so L2
					// ordering preserves cosine ranking.
					nodeDist = pq.ComputeAsymmetricDistance(q, code)
				} else {
					if f.vectors == nil {
						continue
					}
					vec, ok := f.vectors.GetVector(node.ID)
					if !ok {
						continue
					}
					nodeDist = f.distanceFunc(q, vec)
				}
			} else {
				if f.vectors == nil {
					continue
				}
				vec, ok := f.vectors.GetVector(node.ID)
				if !ok {
					continue
				}
				nodeDist = f.distanceFunc(q, vec)
			}

			if topCandidates.Len() < actualK {
				heap.Push(topCandidates, queue.PriorityQueueItem{Node: node.ID, Distance: nodeDist})
				continue
			}

			largest := topCandidates.Top().(queue.PriorityQueueItem)
			if nodeDist < largest.Distance {
				heap.Pop(topCandidates)
				heap.Push(topCandidates, queue.PriorityQueueItem{Node: node.ID, Distance: nodeDist})
			}
		}
	}

	results := make([]index.SearchResult, topCandidates.Len())
	for i := topCandidates.Len() - 1; i >= 0; i-- {
		item := heap.Pop(topCandidates).(queue.PriorityQueueItem)
		results[i] = index.SearchResult{ID: item.Node, Distance: item.Distance}
	}
	return results, nil
}

// Shard interface implementation for sharded Flat indexes

// VectorCount returns the number of vectors in THIS shard only.
// For sharded indexes, total vectors = sum across all shards.
func (f *Flat) VectorCount() int {
	st := f.getState()
	count := 0
	for _, node := range st.nodes {
		if node != nil {
			count++
		}
	}
	return count
}

// ContainsID returns true if this shard owns the given ID.
// Ownership is determined by hash-based partitioning: id % numShards == shardID
func (f *Flat) ContainsID(id uint32) bool {
	if f.numShards <= 1 {
		return true // Non-sharded mode - owns all IDs
	}
	return int(id%uint32(f.numShards)) == f.shardID
}

// ShardID returns this shard's identifier (0-based).
// For non-sharded indexes, always returns 0.
func (f *Flat) ShardID() int {
	return f.shardID
}

// NumShards returns the total number of shards in the system.
// For non-sharded indexes, always returns 1.
func (f *Flat) NumShards() int {
	return f.numShards
}

// NewSharded creates a new sharded Flat index.
// Each shard is an independent flat vector array.
//
// Parameters:
//   - shardID: Zero-based shard identifier (0 to numShards-1)
//   - numShards: Total number of shards in the system
//   - optFns: Standard Flat options (dimension, distance type, etc.)
//
// Usage:
//
//	shard0, _ := flat.NewSharded(0, 4, WithDimension(128))
//	shard1, _ := flat.NewSharded(1, 4, WithDimension(128))
//	// ... create shards 2 and 3
//
// Each shard owns IDs where: id % numShards == shardID
func NewSharded(shardID, numShards int, optFns ...func(o *Options)) (*Flat, error) {
	if shardID < 0 || shardID >= numShards {
		return nil, fmt.Errorf("invalid shardID %d for numShards %d", shardID, numShards)
	}
	if numShards < 1 {
		return nil, fmt.Errorf("numShards must be >= 1, got %d", numShards)
	}

	f, err := New(optFns...)
	if err != nil {
		return nil, err
	}

	f.shardID = shardID
	f.numShards = numShards

	return f, nil
}
