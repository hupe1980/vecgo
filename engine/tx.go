package engine

import (
	"context"
	"fmt"
	"io"
	"iter"
	"os"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/hupe1980/vecgo/codec"
	"github.com/hupe1980/vecgo/index"
	"github.com/hupe1980/vecgo/index/memtable"
	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/wal"
)

// memState holds the active and frozen memtables for atomic snapshots.
type memState struct {
	active *memtable.MemTable
	frozen *memtable.MemTable
}

// Tx is the transaction/coordination unit for WAL-backed mutations.
//
// In the current implementation, Tx provides the concrete behavior while
// Coordinator is a compatibility alias used by vecgo.
type Tx[T any] struct {
	// Lock Stratification
	memMu    sync.RWMutex             // Protects memState transitions and recycledMemTable
	memState atomic.Pointer[memState] // Atomic snapshot of memtables
	metaMu   sync.RWMutex             // Protects metadata operations
	idMu     sync.Mutex               // Protects ID allocation and lifecycle

	// TransactionalIndex provides consolidated ID allocation, apply ops, and vector access
	txIndex index.TransactionalIndex

	recycledMemTable *memtable.MemTable // Recycled memtable for reuse
	flushCh          chan struct{}      // Signal to trigger flush
	flushMu          sync.Mutex         // Protects flushCond
	flushCond        *sync.Cond         // Signal for backpressure/flow control
	stopCh           chan struct{}      // Signal to stop background worker
	wg               sync.WaitGroup
	closed           bool // Protected by idMu

	dataStore Store[T]
	metaStore *metadata.UnifiedIndex

	durability   Durability
	codec        codec.Codec
	snapshotPath string
	distFunc     index.DistanceFunc
	syncWrite    bool
	dimension    int

	scratchPool *sync.Pool
}

type txScratch struct {
	indexResults  []index.SearchResult
	memResults    []index.SearchResult
	frozenResults []index.SearchResult
	mergedResults []index.SearchResult
}

// allocateID allocates a new ID from the index
func (tx *Tx[T]) allocateID() uint64 {
	tx.idMu.Lock()
	defer tx.idMu.Unlock()
	return tx.txIndex.AllocateID()
}

// releaseID releases an ID back to the index
func (tx *Tx[T]) releaseID(id uint64) {
	tx.idMu.Lock()
	defer tx.idMu.Unlock()
	tx.txIndex.ReleaseID(id)
}

// applyInsert applies an insert operation to the index
func (tx *Tx[T]) applyInsert(ctx context.Context, id uint64, vector []float32) error {
	return tx.txIndex.ApplyInsert(ctx, id, vector)
}

// applyUpdate applies an update operation to the index
func (tx *Tx[T]) applyUpdate(ctx context.Context, id uint64, vector []float32) error {
	return tx.txIndex.ApplyUpdate(ctx, id, vector)
}

// applyDelete applies a delete operation to the index
func (tx *Tx[T]) applyDelete(ctx context.Context, id uint64) error {
	return tx.txIndex.ApplyDelete(ctx, id)
}

// vectorByID retrieves a vector from the index by ID
func (tx *Tx[T]) vectorByID(ctx context.Context, id uint64) ([]float32, error) {
	// Zero-lock path for memtable check
	state := tx.memState.Load()
	if state != nil {
		if state.active != nil {
			if vec, found, isDeleted := state.active.Get(id); found {
				if isDeleted {
					return nil, ErrNotFound
				}
				return vec, nil
			}
		}
		if state.frozen != nil {
			if vec, found, isDeleted := state.frozen.Get(id); found {
				if isDeleted {
					return nil, ErrNotFound
				}
				return vec, nil
			}
		}
	}

	return tx.txIndex.VectorByID(ctx, id)
}

func (tx *Tx[T]) encodePayload(data T) ([]byte, error) {
	return tx.codec.Marshal(data)
}

// Insert inserts a new vector+payload+(optional) metadata atomically.
func (tx *Tx[T]) Insert(ctx context.Context, vector []float32, data T, meta metadata.Metadata) (uint64, error) {
	if err := ctx.Err(); err != nil {
		return 0, err
	}

	id := tx.allocateID()
	payload, err := tx.encodePayload(data)
	if err != nil {
		tx.releaseID(id)
		return 0, err
	}

	if err := tx.durability.LogPrepareInsert(id, vector, payload, meta); err != nil {
		tx.releaseID(id)
		return 0, err
	}

	if tx.syncWrite {
		if err := tx.txIndex.ApplyInsert(ctx, id, vector); err != nil {
			return 0, err
		}
	} else {
		// Async Indexing: Write to MemTable instead of Index
		tx.memMu.RLock()
		state := tx.memState.Load()
		state.active.Insert(id, vector)
		size := state.active.Size()
		tx.memMu.RUnlock()

		// Trigger flush if MemTable is getting large (simple backpressure)
		if size >= 1000 {
			select {
			case tx.flushCh <- struct{}{}:
			default:
			}
		}

		// Backpressure: Wait if MemTable is too large and flush is in progress
		if size >= 10000 {
			tx.flushMu.Lock()
			for {
				tx.memMu.RLock()
				state = tx.memState.Load()
				currentSize := state.active.Size()
				tx.memMu.RUnlock()
				if currentSize < 10000 {
					break
				}
				tx.flushCond.Wait()
			}
			tx.flushMu.Unlock()
		}
	}

	if err := tx.dataStore.Set(id, data); err != nil {
		// Rollback
		if tx.syncWrite {
			tx.txIndex.ApplyDelete(ctx, id)
		} else {
			tx.memMu.RLock()
			state := tx.memState.Load()
			state.active.Delete(id)
			tx.memMu.RUnlock()
		}
		return 0, err
	}

	// Safe-by-default: clone metadata to prevent external mutation
	if meta != nil {
		safeMeta := metadata.CloneIfNeeded(meta)
		tx.metaMu.Lock()
		tx.metaStore.Set(id, safeMeta)
		tx.metaMu.Unlock()
	}

	if err := tx.durability.LogCommitInsert(id); err != nil {
		if meta != nil {
			tx.metaMu.Lock()
			tx.metaStore.Delete(id)
			tx.metaMu.Unlock()
		}
		_ = tx.dataStore.Delete(id)
		// Rollback
		if tx.syncWrite {
			tx.txIndex.ApplyDelete(ctx, id)
		} else {
			tx.memMu.RLock()
			state := tx.memState.Load()
			state.active.Delete(id)
			tx.memMu.RUnlock()
		}
		return 0, err
	}

	return id, nil
}

// BatchInsert inserts multiple vectors+payloads+(optional) metadata atomically.
func (tx *Tx[T]) BatchInsert(ctx context.Context, vectors [][]float32, dataSlice []T, metadataSlice []metadata.Metadata) ([]uint64, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(vectors) != len(dataSlice) || len(vectors) != len(metadataSlice) {
		return nil, fmt.Errorf("batch size mismatch: vectors=%d data=%d metadata=%d", len(vectors), len(dataSlice), len(metadataSlice))
	}
	if len(vectors) == 0 {
		return nil, nil
	}

	ids := make([]uint64, len(vectors))
	payloads := make([][]byte, len(vectors))
	for i := range vectors {
		ids[i] = tx.allocateID()
		b, err := tx.encodePayload(dataSlice[i])
		if err != nil {
			for j := 0; j <= i; j++ {
				tx.releaseID(ids[j])
			}
			return nil, err
		}
		payloads[i] = b
	}

	if err := tx.durability.LogPrepareBatchInsert(ids, vectors, payloads, metadataSlice); err != nil {
		for _, id := range ids {
			tx.releaseID(id)
		}
		return nil, err
	}

	if tx.syncWrite {
		for i := range vectors {
			if err := tx.txIndex.ApplyInsert(ctx, ids[i], vectors[i]); err != nil {
				return nil, err
			}
		}
	} else {
		// Async Indexing: Write to MemTable
		tx.memMu.RLock()
		state := tx.memState.Load()
		for i := range vectors {
			state.active.Insert(ids[i], vectors[i])
		}
		size := state.active.Size()
		tx.memMu.RUnlock()

		// Trigger flush if MemTable is getting large
		if size >= 1000 {
			select {
			case tx.flushCh <- struct{}{}:
			default:
			}
		}

		// Backpressure: Wait if MemTable is too large and flush is in progress
		if size >= 10000 {
			tx.flushMu.Lock()
			for {
				tx.memMu.RLock()
				state = tx.memState.Load()
				currentSize := state.active.Size()
				tx.memMu.RUnlock()
				if currentSize < 10000 {
					break
				}
				tx.flushCond.Wait()
			}
			tx.flushMu.Unlock()
		}
	}

	items := make(map[uint64]T, len(ids))
	for i := range ids {
		items[ids[i]] = dataSlice[i]
	}
	if err := tx.dataStore.BatchSet(items); err != nil {
		for _, id := range ids {
			// Rollback
			if tx.syncWrite {
				tx.txIndex.ApplyDelete(ctx, id)
			} else {
				tx.memMu.RLock()
				state := tx.memState.Load()
				state.active.Delete(id)
				tx.memMu.RUnlock()
			}
		}
		return nil, err
	}

	// Safe-by-default: clone metadata to prevent external mutation
	metaItems := make(map[uint64]metadata.Metadata)
	for i := range ids {
		if metadataSlice[i] != nil {
			safeMeta := metadata.CloneIfNeeded(metadataSlice[i])
			metaItems[ids[i]] = safeMeta
		}
	}
	if len(metaItems) > 0 {
		tx.metaMu.Lock()
		for id, doc := range metaItems {
			tx.metaStore.Set(id, doc)
		}
		tx.metaMu.Unlock()
	}

	if err := tx.durability.LogCommitBatchInsert(ids); err != nil {
		if len(metaItems) > 0 {
			tx.metaMu.Lock()
			for id := range metaItems {
				tx.metaStore.Delete(id)
			}
			tx.metaMu.Unlock()
		}
		_ = tx.dataStore.BatchDelete(ids)
		for _, id := range ids {
			// Rollback: Remove from MemTable
			tx.memMu.RLock()
			state := tx.memState.Load()
			state.active.Delete(id)
			tx.memMu.RUnlock()
		}
		return nil, err
	}

	return ids, nil
}

// Update updates vector+payload and optionally metadata.
// If meta is nil, metadata is left unchanged (matches Vecgo.Update behavior).
func (tx *Tx[T]) Update(ctx context.Context, id uint64, vector []float32, data T, meta metadata.Metadata) error {
	if err := ctx.Err(); err != nil {
		return err
	}

	oldVector, err := tx.vectorByID(ctx, id)
	if err != nil {
		return err
	}
	oldData, ok := tx.dataStore.Get(id)
	if !ok {
		return ErrNotFound
	}

	var oldMeta metadata.Metadata
	oldMetaOK := false
	if meta != nil {
		tx.metaMu.RLock()
		oldMeta, oldMetaOK = tx.metaStore.Get(id)
		tx.metaMu.RUnlock()
	}

	payload, err := tx.encodePayload(data)
	if err != nil {
		return err
	}

	if err := tx.durability.LogPrepareUpdate(id, vector, payload, meta); err != nil {
		return err
	}

	// Async Indexing: Write to MemTable instead of Index
	tx.memMu.RLock()
	state := tx.memState.Load()
	state.active.Insert(id, vector)
	size := state.active.Size()
	tx.memMu.RUnlock()

	// Trigger flush if memtable is large enough
	if size >= 1000 {
		select {
		case tx.flushCh <- struct{}{}:
		default:
		}
	}

	if err := tx.dataStore.Set(id, data); err != nil {
		// Rollback: Restore old vector in MemTable
		tx.memMu.RLock()
		state := tx.memState.Load()
		state.active.Insert(id, oldVector)
		tx.memMu.RUnlock()
		return err
	}

	// Safe-by-default: clone metadata to prevent external mutation
	if meta != nil {
		safeMeta := metadata.CloneIfNeeded(meta)
		tx.metaMu.Lock()
		tx.metaStore.Set(id, safeMeta)
		tx.metaMu.Unlock()
	}

	if err := tx.durability.LogCommitUpdate(id); err != nil {
		if meta != nil {
			tx.metaMu.Lock()
			if oldMetaOK {
				tx.metaStore.Set(id, oldMeta)
			} else {
				tx.metaStore.Delete(id)
			}
			tx.metaMu.Unlock()
		}
		_ = tx.dataStore.Set(id, oldData)
		// Revert memtable change is hard, but since we haven't flushed,
		// the index is still consistent with oldVector.
		// We should ideally revert the memtable entry too.
		tx.memMu.RLock()
		state := tx.memState.Load()
		state.active.Insert(id, oldVector)
		tx.memMu.RUnlock()
		return err
	}

	return nil
}

// Delete removes a vector and associated data from the database.
func (tx *Tx[T]) Delete(ctx context.Context, id uint64) error {
	if err := ctx.Err(); err != nil {
		return err
	}

	oldVector, err := tx.vectorByID(ctx, id)
	if err != nil {
		return err
	}
	oldData, ok := tx.dataStore.Get(id)
	if !ok {
		return ErrNotFound
	}
	tx.metaMu.RLock()
	oldMeta, oldMetaOK := tx.metaStore.Get(id)
	tx.metaMu.RUnlock()

	if err := tx.durability.LogPrepareDelete(id); err != nil {
		return err
	}

	// Async Indexing: Remove from MemTable
	tx.memMu.RLock()
	state := tx.memState.Load()
	state.active.Delete(id)
	tx.memMu.RUnlock()

	// With Tombstones (LSM), we do NOT call applyDelete here.
	// The deletion is logical (in MemTable) and will be physically applied to HNSW
	// during the background flush.
	// vectorByIDLocked checks MemTable first, so it will see the tombstone and return ErrNotFound.

	if err := tx.dataStore.Delete(id); err != nil {
		// Rollback: Restore old vector in MemTable
		tx.memMu.RLock()
		state := tx.memState.Load()
		state.active.Insert(id, oldVector)
		tx.memMu.RUnlock()
		return err
	}
	if oldMetaOK {
		tx.metaMu.Lock()
		tx.metaStore.Delete(id)
		tx.metaMu.Unlock()
	}

	if err := tx.durability.LogCommitDelete(id); err != nil {
		// Rollback: Restore old vector in MemTable
		tx.memMu.RLock()
		state := tx.memState.Load()
		state.active.Insert(id, oldVector)
		tx.memMu.RUnlock()
		_ = tx.dataStore.Set(id, oldData)
		if oldMetaOK {
			tx.metaMu.Lock()
			tx.metaStore.Set(id, oldMeta)
			tx.metaMu.Unlock()
		}
		return err
	}

	tx.releaseID(id)
	return nil
}

// flushMemTableLocked flushes the MemTable to the main index (caller must hold lock).
// Get retrieves the data associated with an ID from the data store.
func (tx *Tx[T]) Get(id uint64) (T, bool) {
	return tx.dataStore.Get(id)
}

// GetMetadata retrieves the metadata associated with an ID from the metadata store.
func (tx *Tx[T]) GetMetadata(id uint64) (metadata.Metadata, bool) {
	return tx.metaStore.Get(id)
}

// KNNSearch performs a K-nearest neighbor search on the underlying index.
// This method is added to satisfy the coordinator[T] interface.
func (tx *Tx[T]) KNNSearch(ctx context.Context, query []float32, k int, opts *index.SearchOptions) ([]index.SearchResult, error) {
	var results []index.SearchResult
	if err := tx.KNNSearchWithBuffer(ctx, query, k, opts, &results); err != nil {
		return nil, err
	}
	return results, nil
}

// KNNSearchWithBuffer performs a K-nearest neighbor search and appends results to the provided buffer.
func (tx *Tx[T]) KNNSearchWithBuffer(ctx context.Context, query []float32, k int, opts *index.SearchOptions, buf *[]index.SearchResult) error {
	// Get scratch buffer
	scratch := tx.scratchPool.Get().(*txScratch)
	defer tx.scratchPool.Put(scratch)

	// Reset scratch buffers
	scratch.indexResults = scratch.indexResults[:0]
	scratch.memResults = scratch.memResults[:0]
	scratch.frozenResults = scratch.frozenResults[:0]
	scratch.mergedResults = scratch.mergedResults[:0]

	// 1. Search Main Index
	if err := tx.txIndex.KNNSearchWithBuffer(ctx, query, k, opts, &scratch.indexResults); err != nil {
		return err
	}

	// Capture MemTables safely (Atomic Snapshot)
	state := tx.memState.Load()
	memT := state.active
	frozenT := state.frozen

	// 2. Filter Index Results (remove items deleted in MemTables)
	// In-place filtering
	n := 0
	for _, res := range scratch.indexResults {
		// Check MemTable
		if _, found, isDeleted := memT.Get(res.ID); found && isDeleted {
			continue
		}
		// Check FrozenMemTable
		if frozenT != nil {
			if _, found, isDeleted := frozenT.Get(res.ID); found && isDeleted {
				continue
			}
		}
		scratch.indexResults[n] = res
		n++
	}
	scratch.indexResults = scratch.indexResults[:n]

	// 3. Search MemTables
	var filter func(uint64) bool
	if opts != nil && opts.Filter != nil {
		filter = opts.Filter
	}

	if err := memT.SearchWithBuffer(query, k, filter, &scratch.memResults); err != nil {
		return err
	}

	if frozenT != nil {
		if err := frozenT.SearchWithBuffer(query, k, filter, &scratch.frozenResults); err != nil {
			return err
		}
	}

	// 4. Merge Results
	index.MergeNSearchResultsInto(&scratch.mergedResults, k, scratch.indexResults, scratch.memResults, scratch.frozenResults)

	// Append to output buffer
	*buf = append(*buf, scratch.mergedResults...)

	return nil
}

// BruteSearch performs a brute-force search on the underlying index.
// This method is added to satisfy the coordinator[T] interface.
func (tx *Tx[T]) BruteSearch(ctx context.Context, query []float32, k int, filter func(id uint64) bool) ([]index.SearchResult, error) {
	// 1. Search Main Index
	indexResults, err := tx.txIndex.BruteSearch(ctx, query, k, filter)
	if err != nil {
		return nil, err
	}

	// Capture MemTables safely (Atomic Snapshot)
	state := tx.memState.Load()
	memT := state.active
	frozenT := state.frozen

	// 2. Filter Index Results
	filteredIndexResults := make([]index.SearchResult, 0, len(indexResults))
	for _, res := range indexResults {
		if _, found, isDeleted := memT.Get(res.ID); found && isDeleted {
			continue
		}
		if frozenT != nil {
			if _, found, isDeleted := frozenT.Get(res.ID); found && isDeleted {
				continue
			}
		}
		filteredIndexResults = append(filteredIndexResults, res)
	}

	// 3. Search MemTables
	var memFilter func(uint64) bool
	if filter != nil {
		memFilter = func(id uint64) bool {
			return filter(id)
		}
	}
	memResults := memT.Search(query, k, memFilter)

	var frozenResults []index.SearchResult
	if frozenT != nil {
		frozenResults = frozenT.Search(query, k, memFilter)
	}

	// 4. Merge Results
	merged := index.MergeNSearchResults(k, filteredIndexResults, memResults, frozenResults)

	return merged, nil
}

// EnableProductQuantization enables Product Quantization (PQ) on the underlying index.
func (tx *Tx[T]) EnableProductQuantization(cfg index.ProductQuantizationConfig) error {
	if cap, ok := tx.txIndex.(index.ProductQuantizationEnabler); ok {
		return cap.EnableProductQuantization(cfg)
	}
	return fmt.Errorf("index type %T does not support product quantization", tx.txIndex)
}

// DisableProductQuantization disables Product Quantization (PQ) on the underlying index.
func (tx *Tx[T]) DisableProductQuantization() {
	if cap, ok := tx.txIndex.(index.ProductQuantizationEnabler); ok {
		cap.DisableProductQuantization()
	}
}

// runFlushWorker handles background flushing of the MemTable to the main index.
func (tx *Tx[T]) runFlushWorker() {
	defer tx.wg.Done()

	for {
		select {
		case <-tx.flushCh:
			tx.flushMemTable()
		case <-tx.stopCh:
			// Final flush before exit
			tx.flushMemTable()
			return
		}
	}
}

// flushMemTable flushes the MemTable to the main index.
func (tx *Tx[T]) flushMemTable() {
	tx.memMu.Lock()

	state := tx.memState.Load()
	// If a flush is already in progress (frozen table exists), we skip.
	// The worker loop calls this repeatedly, so it will pick it up next time.
	if state.frozen != nil {
		tx.memMu.Unlock()
		return
	}

	if state.active.Size() == 0 {
		tx.memMu.Unlock()
		return
	}

	// Prepare new active table
	var newActive *memtable.MemTable
	// Reuse recycled memtable if available, otherwise create new
	if tx.recycledMemTable != nil {
		newActive = tx.recycledMemTable
		tx.recycledMemTable = nil
	} else {
		newActive = memtable.New(tx.dimension, tx.distFunc)
	}

	// Swap: active -> frozen, newActive -> active
	newState := &memState{
		active: newActive,
		frozen: state.active,
	}
	tx.memState.Store(newState)

	tx.flushMu.Lock()
	tx.flushCond.Broadcast() // Signal that MemTable is empty
	tx.flushMu.Unlock()

	// Get items to flush
	items := newState.frozen.Items()
	tx.memMu.Unlock()

	// Apply to main index
	// We use a background context as this is an async operation
	ctx := context.Background()

	var insertIDs []uint64
	var insertVecs [][]float32

	for _, item := range items {
		if item.IsDeleted {
			err := tx.txIndex.ApplyDelete(ctx, uint64(item.ID))
			// Ignore if node already deleted or not found
			if err != nil {
				errMsg := err.Error()
				if strings.Contains(errMsg, "not found") || strings.Contains(errMsg, "has been deleted") {
					err = nil
				}
			}
			if err != nil {
				// In a production system, we should log this error properly.
			}
		} else {
			insertIDs = append(insertIDs, uint64(item.ID))
			insertVecs = append(insertVecs, item.Vector)
		}
	}

	if len(insertIDs) > 0 {
		if err := tx.txIndex.ApplyBatchInsert(ctx, insertIDs, insertVecs); err != nil {
			// In a production system, we should log this error properly.
		}
	}

	// Clear frozen table after flush is done
	tx.memMu.Lock()
	// Reload state to be sure
	state = tx.memState.Load()

	// Reset and recycle the frozen memtable
	frozen := state.frozen
	if frozen != nil {
		frozen.Reset()
		tx.recycledMemTable = frozen
	}

	// Update state: frozen -> nil
	finalState := &memState{
		active: state.active,
		frozen: nil,
	}
	tx.memState.Store(finalState)

	tx.flushMu.Lock()
	tx.flushCond.Broadcast() // Signal that flush is done
	tx.flushMu.Unlock()
	tx.memMu.Unlock()
}

// HybridSearch performs a hybrid search combining vector similarity and metadata filtering.
func (tx *Tx[T]) HybridSearch(ctx context.Context, query []float32, k int, opts *HybridSearchOptions) ([]index.SearchResult, error) {
	if opts == nil {
		opts = &HybridSearchOptions{EF: 0}
	}

	// If no metadata filters, fall back to regular KNN search
	if opts.MetadataFilters == nil || len(opts.MetadataFilters.Filters) == 0 {
		return tx.KNNSearch(ctx, query, k, &index.SearchOptions{EFSearch: opts.EF})
	}

	// Create metadata filter function
	metadataFilter := func(id uint64) bool {
		meta, ok := tx.metaStore.Get(id)
		if !ok {
			return false
		}
		return opts.MetadataFilters.Matches(meta)
	}
	if tx.metaStore != nil {
		metadataFilter = tx.metaStore.CreateFilterFunc(opts.MetadataFilters)
	}

	if opts.PreFilter {
		// Pre-filtering
		searchOpts := &index.SearchOptions{EFSearch: opts.EF, Filter: metadataFilter}
		return tx.KNNSearch(ctx, query, k, searchOpts)
	}

	// Post-filtering
	oversampleK := min(k*3, 1000)
	searchOpts := &index.SearchOptions{EFSearch: opts.EF}
	bestCandidates, err := tx.KNNSearch(ctx, query, oversampleK, searchOpts)
	if err != nil {
		return nil, err
	}

	// Apply metadata filtering
	results := make([]index.SearchResult, 0, k)
	for _, item := range bestCandidates {
		if len(results) >= k {
			break
		}
		if metadataFilter(item.ID) {
			results = append(results, item)
		}
	}
	return results, nil
}

// KNNSearchStream returns an iterator over K-nearest neighbor search results.
func (tx *Tx[T]) KNNSearchStream(ctx context.Context, query []float32, k int, opts *index.SearchOptions) iter.Seq2[index.SearchResult, error] {
	// Capture MemTables safely (Atomic Snapshot)
	state := tx.memState.Load()
	memT := state.active
	frozenT := state.frozen

	// 1. Stream from Main Index (filtered)
	indexStream := tx.txIndex.KNNSearchStream(ctx, query, k, opts)

	filteredIndexStream := func(yield func(index.SearchResult, error) bool) {
		for res, err := range indexStream {
			if err != nil {
				if !yield(res, err) {
					return
				}
				continue
			}
			// Check MemTable
			if _, found, isDeleted := memT.Get(res.ID); found && isDeleted {
				continue
			}
			// Check FrozenMemTable
			if frozenT != nil {
				if _, found, isDeleted := frozenT.Get(res.ID); found && isDeleted {
					continue
				}
			}
			if !yield(res, nil) {
				return
			}
		}
	}

	// 2. Search MemTables (returns slice)
	var filter func(uint64) bool
	if opts != nil {
		filter = opts.Filter
	}
	memResults := memT.Search(query, k, filter)

	var frozenResults []index.SearchResult
	if frozenT != nil {
		frozenResults = frozenT.Search(query, k, filter)
	}

	// 3. Merge Streams
	allMemResults := memResults
	if len(frozenResults) > 0 {
		allMemResults = index.MergeSearchResults(memResults, frozenResults, k)
	}

	if len(allMemResults) == 0 {
		return filteredIndexStream
	}
	memStream := index.SliceToStream(allMemResults)
	return index.MergeSearchStreams(filteredIndexStream, memStream)
}

// SaveToWriter saves the database to an io.Writer.
func (tx *Tx[T]) SaveToWriter(w io.Writer) error {
	// Flush MemTable to ensure all data is in the index
	tx.flushMemTable()

	// Convert UnifiedIndex to map store for snapshot saving
	metadataStore := NewMapStore[metadata.Metadata]()
	tx.metaMu.RLock()
	for id, doc := range tx.metaStore.ToMap() {
		_ = metadataStore.Set(id, doc)
	}
	tx.metaMu.RUnlock()
	return SaveToWriter(w, tx.txIndex, tx.dataStore, metadataStore, tx.codec)
}

// SaveToFile saves the database to a file.
func (tx *Tx[T]) SaveToFile(path string) error {
	// Flush MemTable to ensure all data is in the index
	tx.flushMemTable()

	// Convert UnifiedIndex to map store for snapshot saving
	metadataStore := NewMapStore[metadata.Metadata]()
	tx.metaMu.RLock()
	for id, doc := range tx.metaStore.ToMap() {
		_ = metadataStore.Set(id, doc)
	}
	tx.metaMu.RUnlock()

	var w *wal.WAL
	if walLog, ok := tx.durability.(*wal.WAL); ok {
		w = walLog
	}

	return SaveToFile(path, tx.txIndex, tx.dataStore, metadataStore, w, tx.codec)
}

// RecoverFromWAL replays the write-ahead log.
func (tx *Tx[T]) RecoverFromWAL(ctx context.Context) error {
	w, ok := tx.durability.(*wal.WAL)
	if !ok {
		return fmt.Errorf("durability layer is not a WAL")
	}
	return RecoverFromWAL(ctx, tx.txIndex, tx.dataStore, tx.metaStore, w, tx.codec)
}

// Stats returns statistics about the underlying index.
func (tx *Tx[T]) Stats() index.Stats {
	stats := tx.txIndex.Stats()
	if stats.Storage == nil {
		stats.Storage = make(map[string]string)
	}
	state := tx.memState.Load()
	stats.Storage["MemTableSize"] = fmt.Sprintf("%d", state.active.Size())
	return stats
}

// Checkpoint creates a checkpoint in the durability layer (WAL).
func (tx *Tx[T]) Checkpoint() error {
	if w, ok := tx.durability.(*wal.WAL); ok {
		return w.Checkpoint()
	}
	return nil
}

// autoCheckpoint is called by WAL when auto-checkpoint thresholds are exceeded.
func (tx *Tx[T]) autoCheckpoint() error {
	if tx.snapshotPath == "" {
		return nil
	}
	// Create a temporary file for the snapshot
	tmpPath := tx.snapshotPath + ".tmp"
	if err := tx.SaveToFile(tmpPath); err != nil {
		return fmt.Errorf("failed to save snapshot: %w", err)
	}
	// Rename to final path (atomic)
	if err := os.Rename(tmpPath, tx.snapshotPath); err != nil {
		return fmt.Errorf("failed to rename snapshot: %w", err)
	}
	return nil
}

// Close releases all resources held by the transaction coordinator.
func (tx *Tx[T]) Close() error {
	tx.idMu.Lock()
	if tx.closed {
		tx.idMu.Unlock()
		return nil
	}
	tx.closed = true
	// Stop background worker
	close(tx.stopCh)
	tx.idMu.Unlock()

	tx.wg.Wait()

	var errs []error
	if err := tx.durability.Close(); err != nil {
		errs = append(errs, err)
	}

	// Close index if it implements io.Closer
	if c, ok := tx.txIndex.(io.Closer); ok {
		if err := c.Close(); err != nil {
			errs = append(errs, err)
		}
	}

	if len(errs) > 0 {
		return fmt.Errorf("failed to close transaction coordinator: %v", errs)
	}
	return nil
}
