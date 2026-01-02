// Package flat provides an implementation of a flat index for vector storage and searcher.
package flat

import (
	"context"
	"fmt"
	"iter"
	"slices"
	"sync"
	"sync/atomic"

	"github.com/hupe1980/vecgo/core"
	"github.com/hupe1980/vecgo/distance"
	"github.com/hupe1980/vecgo/index"
	"github.com/hupe1980/vecgo/internal/bitset"
	"github.com/hupe1980/vecgo/internal/container"
	"github.com/hupe1980/vecgo/internal/conv"
	"github.com/hupe1980/vecgo/quantization"
	"github.com/hupe1980/vecgo/searcher"
	"github.com/hupe1980/vecgo/vectorstore"
	"github.com/hupe1980/vecgo/vectorstore/columnar"
)

// Compile-time checks to ensure Flat satisfies required interfaces.
var _ index.Index = (*Flat)(nil)
var _ index.TransactionalIndex = (*Flat)(nil)
var _ index.ProductQuantizationEnabler = (*Flat)(nil)

// Options contains configuration options for the flat index.
type Options struct {
	// Dimension is the fixed vector dimensionality for this index.
	// It must be > 0 and is enforced for all inserts/updates/searches.
	Dimension int

	// DistanceType represents the type of distance function used for calculating distances between vectors.
	DistanceType index.DistanceType

	// NormalizeVectors enables L2 normalization for stored vectors and queries.
	// Commonly used for cosine searcher.
	NormalizeVectors bool
}

// DefaultOptions contains the default configuration options for the flat index.
var DefaultOptions = Options{
	Dimension:    0,
	DistanceType: index.DistanceTypeSquaredL2,
}

// Flat represents a flat index for vector storage and searcher.
// Flat is a brute-force index that stores vectors in a linear array.
// It uses a zero-overhead architecture (maxID + BitSet) for O(1) inserts.
type Flat struct {
	// State
	maxID   atomic.Uint32
	deleted *bitset.BitSet
	pqCodes *container.SegmentedArray[[]byte]

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

// Name returns the name of the index.
func (*Flat) Name() string { return "Flat" }

// Dimension returns the dimensionality of the vectors in the index.
func (f *Flat) Dimension() int {
	return int(f.dimension.Load())
}

// AllocateID reserves a new ID by atomically incrementing the counter.
func (f *Flat) AllocateID() core.LocalID {
	return core.LocalID(f.maxID.Add(1) - 1)
}

// ReleaseID marks an ID as unused.
// Note: IDs are never reused (ID Stability), so this just clears the slot.
func (f *Flat) ReleaseID(id core.LocalID) {
	f.writeMu.Lock()
	defer f.writeMu.Unlock()

	if id >= core.LocalID(f.maxID.Load()) {
		return
	}

	// Mark as deleted
	f.deleted.Set(uint32(id))
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
		deleted:      bitset.New(1024),
	}
	dimI32, err := conv.IntToInt32(opts.Dimension)
	if err != nil {
		return nil, err
	}
	f.dimension.Store(dimI32)

	return f, nil
}

// ProductQuantizationEnabled returns true if PQ is enabled.
func (f *Flat) ProductQuantizationEnabled() bool {
	return f.pq.Load() != nil
}

// DisableProductQuantization disables PQ.
func (f *Flat) DisableProductQuantization() {
	f.writeMu.Lock()
	defer f.writeMu.Unlock()

	f.pq.Store(nil)
	f.pqCodes = nil
}

// EnableProductQuantization enables PQ with the given config.
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

	maxID := f.maxID.Load()
	vectors := make([][]float32, 0, maxID)

	// Collect vectors for training
	for id := range maxID {
		if f.deleted.Test(id) {
			continue
		}
		v, ok := f.vectors.GetVector(core.LocalID(id))
		if ok {
			vectors = append(vectors, v)
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

	// Encode all existing vectors
	pqCodes := container.NewSegmentedArray[[]byte]()
	for id := range maxID {
		if f.deleted.Test(id) {
			continue
		}
		v, ok := f.vectors.GetVector(core.LocalID(id))
		if ok {
			code := pq.Encode(v)
			pqCodes.Set(id, code)
		}
	}

	f.pqCodes = pqCodes
	f.pq.Store(pq)
	return nil
}

// Insert inserts a vector into the flat index.
func (f *Flat) Insert(ctx context.Context, v []float32) (core.LocalID, error) {
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

	if f.vectors == nil {
		return 0, fmt.Errorf("flat: vector store not configured")
	}

	pq := f.pq.Load()

	// Allocate new ID
	id := f.maxID.Add(1) - 1

	if err := f.vectors.SetVector(core.LocalID(id), vec); err != nil {
		return 0, err
	}

	if pq != nil {
		if f.pqCodes == nil {
			f.pqCodes = container.NewSegmentedArray[[]byte]()
		}
		f.pqCodes.Set(id, pq.Encode(vec))
	}

	return core.LocalID(id), nil
}

// ApplyInsert inserts a vector at an explicit ID.
// This is intended for deterministic WAL replay.
func (f *Flat) ApplyInsert(ctx context.Context, id core.LocalID, v []float32) error {
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

	// Ensure maxID covers this ID (for recovery)
	if uint32(id) >= f.maxID.Load() {
		f.maxID.Store(uint32(id) + 1)
	}

	if err := f.vectors.SetVector(id, vec); err != nil {
		return err
	}

	pq := f.pq.Load()
	if pq != nil {
		if f.pqCodes == nil {
			f.pqCodes = container.NewSegmentedArray[[]byte]()
		}
		f.pqCodes.Set(uint32(id), pq.Encode(vec))
	}
	return nil
}

// ApplyBatchInsert inserts multiple vectors with specific IDs.
func (f *Flat) ApplyBatchInsert(ctx context.Context, ids []core.LocalID, vectors [][]float32) error {
	if len(ids) != len(vectors) {
		return fmt.Errorf("ids and vectors length mismatch")
	}
	if err := ctx.Err(); err != nil {
		return err
	}

	f.writeMu.Lock()
	defer f.writeMu.Unlock()

	currentDim := int(f.dimension.Load())
	pq := f.pq.Load()

	for i, v := range vectors {
		id := ids[i]
		if len(v) == 0 {
			return index.ErrEmptyVector
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

		// Update maxID if needed
		if uint32(id) >= f.maxID.Load() {
			f.maxID.Store(uint32(id) + 1)
		}

		// If the ID was previously deleted, undelete it
		if f.deleted.Test(uint32(id)) {
			f.deleted.Unset(uint32(id))
		}

		if err := f.vectors.SetVector(id, vec); err != nil {
			return err
		}

		if pq != nil {
			if f.pqCodes == nil {
				f.pqCodes = container.NewSegmentedArray[[]byte]()
			}
			f.pqCodes.Set(uint32(id), pq.Encode(vec))
		}
	}

	return nil
}

// ApplyUpdate updates a vector at an explicit ID (deterministic WAL replay helper).
func (f *Flat) ApplyUpdate(ctx context.Context, id core.LocalID, v []float32) error {
	return f.Update(ctx, id, v)
}

// ApplyDelete deletes a vector at an explicit ID (deterministic WAL replay helper).
func (f *Flat) ApplyDelete(ctx context.Context, id core.LocalID) error {
	return f.Delete(ctx, id)
}

// VectorByID returns the vector stored for the given ID.
// WARNING: The returned slice may alias internal memory (especially for mmap stores).
// Callers should not modify the returned slice. Make a copy if mutation is needed.
func (f *Flat) VectorByID(ctx context.Context, id core.LocalID) ([]float32, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if uint32(id) >= f.maxID.Load() {
		return nil, &index.ErrNodeNotFound{ID: id}
	}

	if f.deleted.Test(uint32(id)) {
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
// Note: BatchInsert appends new IDs.
func (f *Flat) BatchInsert(ctx context.Context, vectors [][]float32) index.BatchInsertResult {
	result := index.BatchInsertResult{
		IDs:    make([]core.LocalID, len(vectors)),
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
	pq := f.pq.Load()

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

		// Allocate new ID
		id := f.maxID.Add(1) - 1

		if err := f.vectors.SetVector(core.LocalID(id), vec); err != nil {
			// If storage fails, we must mark this ID as deleted to avoid "holes"
			// that look like valid zero-vectors or similar issues.
			f.deleted.Set(id)
			result.Errors[i] = err
			continue
		}

		if pq != nil {
			if f.pqCodes == nil {
				f.pqCodes = container.NewSegmentedArray[[]byte]()
			}
			f.pqCodes.Set(id, pq.Encode(vec))
		}
		result.IDs[i] = core.LocalID(id)
	}

	return result
}

// Delete removes a vector from the flat index by marking it as deleted.
func (f *Flat) Delete(ctx context.Context, id core.LocalID) error {
	// Check for cancellation
	if err := ctx.Err(); err != nil {
		return err
	}

	// We don't strictly need write lock for setting a bit in bitset if it's thread-safe,
	// but we need to coordinate with other operations.
	// BitSet is likely not thread-safe for concurrent Set/Test.
	f.writeMu.Lock()
	defer f.writeMu.Unlock()

	if uint32(id) >= f.maxID.Load() {
		return &index.ErrNodeNotFound{ID: id}
	}

	if f.deleted.Test(uint32(id)) {
		return &index.ErrNodeDeleted{ID: id}
	}

	// Mark as deleted
	f.deleted.Set(uint32(id))

	// Also remove from PQ codes if present?
	// We don't need to remove from SegmentedArray, just ignore it.
	// But we might want to clear it to save memory if it was a pointer?
	// []byte is small.

	if f.vectors != nil {
		_ = f.vectors.DeleteVector(id)
	}

	return nil
}

// Update updates a vector in the flat index.
func (f *Flat) Update(ctx context.Context, id core.LocalID, v []float32) error {
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

	if uint32(id) >= f.maxID.Load() {
		return &index.ErrNodeNotFound{ID: id}
	}

	if f.deleted.Test(uint32(id)) {
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

	pq := f.pq.Load()
	if pq != nil {
		if f.pqCodes == nil {
			f.pqCodes = container.NewSegmentedArray[[]byte]()
		}
		f.pqCodes.Set(uint32(id), pq.Encode(vec))
	}

	return nil
}

// SearchWithContext performs a K-nearest neighbor search using a reusable Searcher context.
// This method is allocation-free in the steady state (except for the result slice).
func (f *Flat) SearchWithContext(ctx context.Context, s *searcher.Searcher, q []float32, k int, opts *index.SearchOptions) ([]index.SearchResult, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if f.maxID.Load() == 0 {
		return nil, nil
	}

	query, err := f.prepareQuery(s, q, k)
	if err != nil {
		return nil, err
	}

	// Setup Heap
	// We use ScratchCandidates (MaxHeap) to keep top K smallest distances.
	s.ScratchCandidates.Reset()

	var filter func(id core.LocalID) bool
	if opts != nil && opts.Filter != nil {
		filter = opts.Filter
	}

	pq := f.pq.Load()
	maxID := f.maxID.Load()

	// Iterate
	for id := range maxID {
		if f.deleted.Test(id) {
			continue
		}
		if filter != nil && !filter(core.LocalID(id)) {
			continue
		}

		dist, ok := f.computeDistance(core.LocalID(id), query, pq)
		if !ok {
			continue
		}

		s.ScratchCandidates.PushItemBounded(searcher.PriorityQueueItem{Node: core.LocalID(id), Distance: dist}, k)
	}

	return f.extractResults(s.ScratchCandidates), nil
}

func (f *Flat) prepareQuery(s *searcher.Searcher, q []float32, k int) ([]float32, error) {
	if k <= 0 {
		return nil, index.ErrInvalidK
	}
	currentDim := int(f.dimension.Load())
	if currentDim > 0 && len(q) != currentDim {
		return nil, &index.ErrDimensionMismatch{Expected: currentDim, Actual: len(q)}
	}

	// Normalize query if needed
	if f.opts.NormalizeVectors {
		// Use Searcher scratch for normalization
		if len(s.ScratchVec) < len(q) {
			s.ScratchVec = make([]float32, len(q))
		}
		copy(s.ScratchVec, q)
		if !distance.NormalizeL2InPlace(s.ScratchVec) {
			return nil, fmt.Errorf("flat: cannot normalize zero query")
		}
		return s.ScratchVec, nil
	}
	return q, nil
}

func (f *Flat) computeDistance(id core.LocalID, query []float32, pq *quantization.ProductQuantizer) (float32, bool) {
	if f.pqCodes != nil {
		code, ok := f.pqCodes.Get(uint32(id))
		if ok && code != nil {
			return pq.ComputeAsymmetricDistance(query, code), true
		}
	}

	if f.vectors == nil {
		return 0, false
	}
	vec, ok := f.vectors.GetVector(id)
	if !ok {
		return 0, false
	}
	return f.distanceFunc(query, vec), true
}

func (f *Flat) extractResults(candidates *searcher.PriorityQueue) []index.SearchResult {
	// Extract results
	res := make([]index.SearchResult, 0, candidates.Len())
	for candidates.Len() > 0 {
		item, _ := candidates.PopItem()
		res = append(res, index.SearchResult{ID: uint32(item.Node), Distance: item.Distance})
	}

	// Reverse (MaxHeap pops largest first)
	slices.Reverse(res)
	return res
}

// KNNSearch performs a K-nearest neighbor search in the flat index.
func (f *Flat) KNNSearch(ctx context.Context, q []float32, k int, opts *index.SearchOptions) ([]index.SearchResult, error) {
	// Extract filter from options (Flat doesn't use efSearch)
	var filter func(id core.LocalID) bool
	if opts != nil && opts.Filter != nil {
		filter = opts.Filter
	}
	return f.BruteSearch(ctx, q, k, filter)
}

// KNNSearchWithBuffer performs a K-nearest neighbor search and appends results to the provided buffer.
func (f *Flat) KNNSearchWithBuffer(ctx context.Context, q []float32, k int, opts *index.SearchOptions, buf *[]index.SearchResult) error {
	var filter func(id core.LocalID) bool
	if opts != nil && opts.Filter != nil {
		filter = opts.Filter
	}
	return f.BruteSearchWithBuffer(ctx, q, k, filter, buf)
}

// KNNSearchWithContext performs a K-nearest neighbor search using the provided Searcher context.
func (f *Flat) KNNSearchWithContext(ctx context.Context, s *searcher.Searcher, q []float32, k int, opts *index.SearchOptions) error {
	var filter func(id core.LocalID) bool
	if opts != nil && opts.Filter != nil {
		filter = opts.Filter
	}
	return f.BruteSearchWithContext(ctx, s, q, k, filter)
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
		var filter func(id core.LocalID) bool
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
func (f *Flat) BruteSearch(ctx context.Context, query []float32, k int, filter func(id core.LocalID) bool) ([]index.SearchResult, error) {
	res := make([]index.SearchResult, 0, k)
	if err := f.BruteSearchWithBuffer(ctx, query, k, filter, &res); err != nil {
		return nil, err
	}
	return res, nil
}

// BruteSearchWithBuffer performs a brute-force search and appends results to the provided buffer.
func (f *Flat) BruteSearchWithBuffer(ctx context.Context, query []float32, k int, filter func(id core.LocalID) bool, buf *[]index.SearchResult) error {
	if err := ctx.Err(); err != nil {
		return err
	}

	if f.maxID.Load() == 0 {
		return nil
	}

	q, actualK, err := f.prepareBruteSearch(query, k)
	if err != nil {
		return err
	}

	topCandidates := searcher.NewPriorityQueue(true)

	pq := f.pq.Load()

	// === BATCH DISTANCE OPTIMIZATION ===
	// For non-PQ mode, gather all vectors and compute distances in batch
	// This improves cache locality and SIMD vectorization
	if pq == nil {
		f.bruteSearchBatch(q, actualK, filter, topCandidates)
	} else {
		f.bruteSearchPQ(q, actualK, filter, topCandidates, pq)
	}

	f.extractBruteResults(topCandidates, buf)
	return nil
}

func (f *Flat) prepareBruteSearch(query []float32, k int) ([]float32, int, error) {
	maxID := f.maxID.Load()
	currentDim := int(f.dimension.Load())

	if k <= 0 {
		return nil, 0, index.ErrInvalidK
	}
	if currentDim > 0 && len(query) != currentDim {
		return nil, 0, &index.ErrDimensionMismatch{Expected: currentDim, Actual: len(query)}
	}

	q := query
	if f.opts.NormalizeVectors {
		norm, ok := distance.NormalizeL2Copy(query)
		if !ok {
			return nil, 0, fmt.Errorf("flat: cannot normalize zero query")
		}
		q = norm
	}

	actualK := k
	actualKU32, err := conv.IntToUint32(actualK)
	if err != nil {
		return nil, 0, err
	}
	if actualKU32 > maxID {
		actualK, err = conv.Uint32ToInt(maxID)
		if err != nil {
			return nil, 0, err
		}
	}
	return q, actualK, nil
}

func (f *Flat) bruteSearchBatch(q []float32, k int, filter func(id core.LocalID) bool, topCandidates *searcher.PriorityQueue) {
	maxID := f.maxID.Load()
	// Gather all valid node IDs and vectors
	batchIDs := make([]core.LocalID, 0, maxID)
	batchVectors := make([][]float32, 0, maxID)

	for id := range maxID {
		if f.deleted.Test(id) {
			continue
		}
		if filter != nil && !filter(core.LocalID(id)) {
			continue
		}

		if f.vectors == nil {
			continue
		}

		v, ok := f.vectors.GetVector(core.LocalID(id))
		if !ok {
			continue
		}

		batchIDs = append(batchIDs, core.LocalID(id))
		batchVectors = append(batchVectors, v)
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
			topCandidates.PushItemBounded(searcher.PriorityQueueItem{Node: nodeID, Distance: nodeDist}, k)
		}
	}
}

func (f *Flat) bruteSearchPQ(q []float32, k int, filter func(id core.LocalID) bool, topCandidates *searcher.PriorityQueue, pq *quantization.ProductQuantizer) {
	// PQ mode: use asymmetric distance (can't batch due to custom codes)
	// NOTE: PQ asymmetric distance approximates L2 distance on normalized vectors.
	// For cosine, vectors are already normalized so L2 ordering is equivalent.
	maxID := f.maxID.Load()
	for i := range maxID {
		id := core.LocalID(i)
		if f.deleted.Test(uint32(id)) {
			continue
		}
		if filter != nil && !filter(id) {
			continue
		}

		dist, ok := f.computeDistance(id, q, pq)
		if !ok {
			continue
		}

		topCandidates.PushItemBounded(searcher.PriorityQueueItem{Node: id, Distance: dist}, k)
	}
}

func (f *Flat) extractBruteResults(topCandidates *searcher.PriorityQueue, buf *[]index.SearchResult) {
	// Extract results from heap
	// Note: MaxHeap pops largest distance first, so we get results in descending order (farthest to nearest)
	// We need to reverse them to get nearest to farthest.

	startLen := len(*buf)
	for topCandidates.Len() > 0 {
		item, _ := topCandidates.PopItem()
		*buf = append(*buf, index.SearchResult{ID: uint32(item.Node), Distance: item.Distance})
	}

	// Reverse the appended segment
	res := *buf
	for i, j := startLen, len(res)-1; i < j; i, j = i+1, j-1 {
		res[i], res[j] = res[j], res[i]
	}
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

// BruteSearchWithContext performs a brute-force search using the provided Searcher context.
func (f *Flat) BruteSearchWithContext(ctx context.Context, s *searcher.Searcher, query []float32, k int, filter func(id core.LocalID) bool) error {
	if err := ctx.Err(); err != nil {
		return err
	}

	if f.maxID.Load() == 0 {
		return nil
	}

	q, err := f.prepareQuery(s, query, k)
	if err != nil {
		return err
	}

	maxID := f.maxID.Load()
	actualK := k
	if int64(actualK) > int64(maxID) {
		actualK, err = conv.Uint32ToInt(maxID)
		if err != nil {
			return err
		}
	}

	s.Candidates.Reset()

	// Simple iteration
	for i := range maxID {
		id := core.LocalID(i)
		if f.deleted.Test(uint32(id)) {
			continue
		}
		if filter != nil && !filter(id) {
			continue
		}
		if f.vectors == nil {
			continue
		}
		v, ok := f.vectors.GetVector(id)
		if !ok {
			continue
		}

		dist := f.distanceFunc(q, v)
		s.Candidates.PushItemBounded(searcher.PriorityQueueItem{Node: id, Distance: dist}, actualK)
	}
	return nil
}
