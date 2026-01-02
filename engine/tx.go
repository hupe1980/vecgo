package engine

import (
	"context"
	"fmt"
	"io"
	"iter"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/hupe1980/vecgo/codec"
	"github.com/hupe1980/vecgo/core"
	"github.com/hupe1980/vecgo/index"
	"github.com/hupe1980/vecgo/index/memtable"
	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/searcher"
	"github.com/hupe1980/vecgo/wal"
)

// memState holds the active and immutable memtables for atomic snapshots.
type memState struct {
	active *memtable.MemTable
	queue  []*memtable.MemTable
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

	flushCh       chan struct{} // Signal to trigger flush
	flushMu       sync.Mutex    // Protects flushCond
	flushCond     *sync.Cond    // Signal for backpressure/flow control
	flushActionMu sync.Mutex    // Serializes flush operations to prevent concurrent HNSW writes
	stopCh        chan struct{} // Signal to stop background worker
	wg            sync.WaitGroup
	closed        bool // Protected by idMu

	dataStore Store[T]
	metaStore *metadata.UnifiedIndex

	durability Durability
	codec      codec.Codec
	distFunc   index.DistanceFunc
	syncWrite  bool
	dimension  int

	scratchPool *sync.Pool
}

type txScratch struct {
	indexResults  []index.SearchResult
	memResults    []index.SearchResult
	queueResults  []index.SearchResult
	mergedResults []index.SearchResult
}

// allocateID allocates a new ID from the index
func (tx *Tx[T]) allocateID() core.LocalID {
	tx.idMu.Lock()
	defer tx.idMu.Unlock()
	return tx.txIndex.AllocateID()
}

// releaseID releases an ID back to the index
func (tx *Tx[T]) releaseID(id core.LocalID) {
	tx.idMu.Lock()
	defer tx.idMu.Unlock()
	tx.txIndex.ReleaseID(id)
}

// vectorByID retrieves a vector from the index by ID
func (tx *Tx[T]) vectorByID(ctx context.Context, id core.LocalID) ([]float32, error) {
	// Zero-lock path for memtable check
	state := tx.memState.Load()
	if state != nil {
		if state.active != nil {
			if vec, found, isDeleted := state.active.Get(id); found {
				if isDeleted {
					return nil, &index.ErrNodeNotFound{ID: id}
				}
				return vec, nil
			}
		}
		// Check queue (newest to oldest)
		for i := len(state.queue) - 1; i >= 0; i-- {
			if vec, found, isDeleted := state.queue[i].Get(id); found {
				if isDeleted {
					return nil, &index.ErrNodeNotFound{ID: id}
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

		// Trigger flush if MemTable is getting large
		if size >= 1000 {
			tx.memMu.Lock()
			// Reload state to ensure we are still the one to swap
			state = tx.memState.Load()
			if state.active.Size() >= 1000 {
				// Swap: active -> queue, newActive -> active
				newActive := memtable.New(tx.dimension, tx.distFunc)
				// Create new queue slice to ensure immutability of the snapshot
				newQueue := make([]*memtable.MemTable, len(state.queue)+1)
				copy(newQueue, state.queue)
				newQueue[len(state.queue)] = state.active

				newState := &memState{
					active: newActive,
					queue:  newQueue,
				}
				tx.memState.Store(newState)

				// Signal flush
				select {
				case tx.flushCh <- struct{}{}:
				default:
				}
			}
			tx.memMu.Unlock()
		}

		// Backpressure: Wait if Queue is too large
		// We do this OUTSIDE the lock to avoid deadlock
		for {
			tx.memMu.RLock()
			state = tx.memState.Load()
			qLen := len(state.queue)
			tx.memMu.RUnlock()

			if qLen < 10 { // Limit queue depth to 10 tables
				break
			}
			time.Sleep(10 * time.Millisecond)
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

	return uint64(id), nil
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

	ids := make([]core.LocalID, len(vectors))
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
		if err := tx.txIndex.ApplyBatchInsert(ctx, ids, vectors); err != nil {
			return nil, err
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

	items := make(map[core.LocalID]T, len(ids))
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
	metaItems := make(map[core.LocalID]metadata.Metadata)
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

	// Convert to uint64 for return
	globalIDs := make([]uint64, len(ids))
	for i, id := range ids {
		globalIDs[i] = uint64(id)
	}
	return globalIDs, nil
}

// Update updates vector+payload and optionally metadata.
// If meta is nil, metadata is left unchanged (matches Vecgo.Update behavior).
func (tx *Tx[T]) Update(ctx context.Context, id uint64, vector []float32, data T, meta metadata.Metadata) error {
	if err := ctx.Err(); err != nil {
		return err
	}

	localID := core.LocalID(id)

	oldVector, err := tx.vectorByID(ctx, localID)
	if err != nil {
		return err
	}
	oldData, ok := tx.dataStore.Get(localID)
	if !ok {
		return &index.ErrNodeNotFound{ID: localID}
	}

	var oldMeta metadata.Metadata
	oldMetaOK := false
	if meta != nil {
		tx.metaMu.RLock()
		oldMeta, oldMetaOK = tx.metaStore.Get(localID)
		tx.metaMu.RUnlock()
	}

	payload, err := tx.encodePayload(data)
	if err != nil {
		return err
	}

	if err := tx.durability.LogPrepareUpdate(localID, vector, payload, meta); err != nil {
		return err
	}

	// Async Indexing: Write to MemTable instead of Index
	tx.memMu.RLock()
	state := tx.memState.Load()
	state.active.Insert(localID, vector)
	size := state.active.Size()
	tx.memMu.RUnlock()

	// Trigger flush if memtable is large enough
	if size >= 1000 {
		select {
		case tx.flushCh <- struct{}{}:
		default:
		}
	}

	if err := tx.dataStore.Set(localID, data); err != nil {
		// Rollback: Restore old vector in MemTable
		tx.memMu.RLock()
		state := tx.memState.Load()
		state.active.Insert(localID, oldVector)
		tx.memMu.RUnlock()
		return err
	}

	// Safe-by-default: clone metadata to prevent external mutation
	if meta != nil {
		safeMeta := metadata.CloneIfNeeded(meta)
		tx.metaMu.Lock()
		tx.metaStore.Set(localID, safeMeta)
		tx.metaMu.Unlock()
	}

	if err := tx.durability.LogCommitUpdate(localID); err != nil {
		if meta != nil {
			tx.metaMu.Lock()
			if oldMetaOK {
				tx.metaStore.Set(localID, oldMeta)
			} else {
				tx.metaStore.Delete(localID)
			}
			tx.metaMu.Unlock()
		}
		_ = tx.dataStore.Set(localID, oldData)
		// Revert memtable change is hard, but since we haven't flushed,
		// the index is still consistent with oldVector.
		// We should ideally revert the memtable entry too.
		tx.memMu.RLock()
		state := tx.memState.Load()
		state.active.Insert(localID, oldVector)
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

	localID := core.LocalID(id)

	oldVector, err := tx.vectorByID(ctx, localID)
	if err != nil {
		return err
	}
	oldData, ok := tx.dataStore.Get(localID)
	if !ok {
		return &index.ErrNodeNotFound{ID: localID}
	}
	tx.metaMu.RLock()
	oldMeta, oldMetaOK := tx.metaStore.Get(localID)
	tx.metaMu.RUnlock()

	if err := tx.durability.LogPrepareDelete(localID); err != nil {
		return err
	}

	// Async Indexing: Remove from MemTable
	tx.memMu.RLock()
	state := tx.memState.Load()
	state.active.Delete(localID)
	tx.memMu.RUnlock()

	// With Tombstones (LSM), we do NOT call applyDelete here.
	// The deletion is logical (in MemTable) and will be physically applied to HNSW
	// during the background flush.
	// vectorByIDLocked checks MemTable first, so it will see the tombstone and return ErrNotFound.

	if err := tx.dataStore.Delete(localID); err != nil {
		// Rollback: Restore old vector in MemTable
		tx.memMu.RLock()
		state := tx.memState.Load()
		state.active.Insert(localID, oldVector)
		tx.memMu.RUnlock()
		return err
	}
	if oldMetaOK {
		tx.metaMu.Lock()
		tx.metaStore.Delete(localID)
		tx.metaMu.Unlock()
	}

	if err := tx.durability.LogCommitDelete(localID); err != nil {
		// Rollback: Restore old vector in MemTable
		tx.memMu.RLock()
		state := tx.memState.Load()
		state.active.Insert(localID, oldVector)
		tx.memMu.RUnlock()
		_ = tx.dataStore.Set(localID, oldData)
		if oldMetaOK {
			tx.metaMu.Lock()
			tx.metaStore.Set(localID, oldMeta)
			tx.metaMu.Unlock()
		}
		return err
	}

	tx.releaseID(localID)
	return nil
}

// flushMemTableLocked flushes the MemTable to the main index (caller must hold lock).
// Get retrieves the data associated with an ID from the data store.
func (tx *Tx[T]) Get(id uint64) (T, bool) {
	return tx.dataStore.Get(core.LocalID(id))
}

// GetMetadata retrieves the metadata associated with an ID from the metadata store.
func (tx *Tx[T]) GetMetadata(id uint64) (metadata.Metadata, bool) {
	return tx.metaStore.Get(core.LocalID(id))
}

// KNNSearch performs a K-nearest neighbor search on the underlying index.
// This method is added to satisfy the coordinator[T] interface.
func (tx *Tx[T]) KNNSearch(ctx context.Context, query []float32, k int, opts *SearchOptions) ([]SearchResult, error) {
	var results []SearchResult
	if err := tx.KNNSearchWithBuffer(ctx, query, k, opts, &results); err != nil {
		return nil, err
	}
	return results, nil
}

// KNNSearchWithBuffer performs a K-nearest neighbor search and appends results to the provided buffer.
func (tx *Tx[T]) KNNSearchWithBuffer(ctx context.Context, query []float32, k int, opts *SearchOptions, buf *[]SearchResult) error {
	// Capture MemTables safely (Atomic Snapshot) BEFORE searching index.
	// This ensures we don't miss items that are flushed during the index search.
	// If an item is flushed during index search:
	// 1. It might appear in index results (if flush finished early).
	// 2. It will definitely appear in queue (which we captured).
	// We handle duplicates by filtering index results against MemTables.
	state := tx.memState.Load()
	memT := state.active
	queue := state.queue

	// Get scratch buffer
	scratch := tx.scratchPool.Get().(*txScratch)
	defer tx.scratchPool.Put(scratch)

	// Reset scratch buffers
	scratch.indexResults = scratch.indexResults[:0]
	scratch.memResults = scratch.memResults[:0]
	scratch.queueResults = scratch.queueResults[:0]
	scratch.mergedResults = scratch.mergedResults[:0]

	// Convert SearchOptions
	var indexOpts *index.SearchOptions
	if opts != nil {
		indexOpts = &index.SearchOptions{
			EFSearch: opts.EFSearch,
		}
		if opts.Filter != nil {
			indexOpts.Filter = func(id core.LocalID) bool {
				return opts.Filter(uint64(id))
			}
		}
	}

	// 1. Search Main Index
	if err := tx.txIndex.KNNSearchWithBuffer(ctx, query, k, indexOpts, &scratch.indexResults); err != nil {
		return err
	}

	// 2. Filter Index Results (remove items present in MemTables)
	// If an item is in MemTable (active or queue), it overrides the index version.
	// This handles both deletions (tombstones) and updates.
	n := 0
	for _, res := range scratch.indexResults {
		localID := core.LocalID(res.ID)
		// Check MemTable
		if found, _ := memT.Contains(localID); found {
			continue
		}
		// Check Queue
		inQueue := false
		for _, t := range queue {
			if found, _ := t.Contains(localID); found {
				inQueue = true
				break
			}
		}
		if inQueue {
			continue
		}
		scratch.indexResults[n] = res
		n++
	}
	scratch.indexResults = scratch.indexResults[:n]

	// 3. Search MemTables
	var filter func(core.LocalID) bool
	if indexOpts != nil && indexOpts.Filter != nil {
		filter = func(id core.LocalID) bool {
			return indexOpts.Filter(id)
		}
	}

	if err := memT.SearchWithBuffer(query, k, filter, &scratch.memResults); err != nil {
		return err
	}

	for _, t := range queue {
		if err := t.SearchWithBuffer(query, k, filter, &scratch.queueResults); err != nil {
			return err
		}
	}

	// Sort queue results as they are concatenated from multiple sorted sources
	if len(scratch.queueResults) > 0 {
		sort.Slice(scratch.queueResults, func(i, j int) bool {
			return scratch.queueResults[i].Distance < scratch.queueResults[j].Distance
		})
	}

	// 4. Merge Results
	index.MergeNSearchResultsInto(&scratch.mergedResults, k, scratch.indexResults, scratch.memResults, scratch.queueResults)

	// Append to output buffer
	for _, res := range scratch.mergedResults {
		*buf = append(*buf, SearchResult{
			ID:       uint64(res.ID),
			Distance: res.Distance,
		})
	}

	return nil
}

// KNNSearchWithContext performs a K-nearest neighbor search using the provided Searcher context.
func (tx *Tx[T]) KNNSearchWithContext(ctx context.Context, query []float32, k int, opts *index.SearchOptions, s *searcher.Searcher) error {
	// Capture MemTables safely (Atomic Snapshot) BEFORE searching index.
	state := tx.memState.Load()
	memT := state.active
	queue := state.queue

	// 1. Search Main Index
	// Results are in s.Candidates (MaxHeap)
	if err := tx.txIndex.KNNSearchWithContext(ctx, s, query, k, opts); err != nil {
		return err
	}

	// 2. Filter Index Results (remove items present in MemTables)
	// We need to check if any item in s.Candidates is in MemTable.
	// If so, we must remove it.
	// Since s.Candidates is a heap, we can't easily remove.
	// We pop all items from s.Candidates into a scratch slice.

	scratch := tx.scratchPool.Get().(*txScratch)
	defer tx.scratchPool.Put(scratch)
	scratch.indexResults = scratch.indexResults[:0]
	scratch.memResults = scratch.memResults[:0]
	scratch.queueResults = scratch.queueResults[:0]

	for s.Candidates.Len() > 0 {
		item, _ := s.Candidates.PopItem()
		scratch.indexResults = append(scratch.indexResults, index.SearchResult{ID: uint32(item.Node), Distance: item.Distance})
	}

	// Filter index results and push back valid ones
	for _, res := range scratch.indexResults {
		localID := core.LocalID(res.ID)
		// Check MemTable
		if found, _ := memT.Contains(localID); found {
			continue
		}
		// Check Queue
		inQueue := false
		for _, t := range queue {
			if found, _ := t.Contains(localID); found {
				inQueue = true
				break
			}
		}
		if inQueue {
			continue
		}
		// Keep it
		s.Candidates.PushItemBounded(searcher.PriorityQueueItem{Node: core.LocalID(res.ID), Distance: res.Distance}, k)
	}

	// 3. Search MemTables
	var filter func(core.LocalID) bool
	if opts != nil && opts.Filter != nil {
		filter = func(id core.LocalID) bool {
			return opts.Filter(id)
		}
	}

	if err := memT.SearchWithBuffer(query, k, filter, &scratch.memResults); err != nil {
		return err
	}

	for _, t := range queue {
		if err := t.SearchWithBuffer(query, k, filter, &scratch.queueResults); err != nil {
			return err
		}
	}

	// 4. Add MemTable results
	for _, res := range scratch.memResults {
		s.Candidates.PushItemBounded(searcher.PriorityQueueItem{Node: core.LocalID(res.ID), Distance: res.Distance}, k)
	}
	for _, res := range scratch.queueResults {
		s.Candidates.PushItemBounded(searcher.PriorityQueueItem{Node: core.LocalID(res.ID), Distance: res.Distance}, k)
	}

	return nil
}

// BruteSearch performs a brute-force search on the underlying index.
// This method is added to satisfy the coordinator[T] interface.
func (tx *Tx[T]) BruteSearch(ctx context.Context, query []float32, k int, filter func(id uint64) bool) ([]SearchResult, error) {
	// Capture MemTables safely (Atomic Snapshot) BEFORE searching index.
	state := tx.memState.Load()
	memT := state.active
	queue := state.queue

	// Adapter for filter: uint64 -> core.LocalID
	var localFilter func(core.LocalID) bool
	if filter != nil {
		localFilter = func(id core.LocalID) bool {
			return filter(uint64(id))
		}
	}

	// 1. Search Main Index
	indexResults, err := tx.txIndex.BruteSearch(ctx, query, k, localFilter)
	if err != nil {
		return nil, err
	}

	// 2. Filter Index Results
	filteredIndexResults := make([]index.SearchResult, 0, len(indexResults))
	for _, res := range indexResults {
		localID := core.LocalID(res.ID)
		if found, _ := memT.Contains(localID); found {
			continue
		}
		inQueue := false
		for _, t := range queue {
			if found, _ := t.Contains(localID); found {
				inQueue = true
				break
			}
		}
		if inQueue {
			continue
		}
		filteredIndexResults = append(filteredIndexResults, res)
	}

	// 3. Search MemTables
	memResults := memT.Search(query, k, localFilter)

	var queueResults []index.SearchResult
	for _, t := range queue {
		res := t.Search(query, k, localFilter)
		if len(res) > 0 {
			queueResults = append(queueResults, res...)
		}
	}
	if len(queueResults) > 0 {
		sort.Slice(queueResults, func(i, j int) bool {
			return queueResults[i].Distance < queueResults[j].Distance
		})
	}

	// 4. Merge Results
	merged := index.MergeNSearchResults(k, filteredIndexResults, memResults, queueResults)

	results := make([]SearchResult, len(merged))
	for i, res := range merged {
		results[i] = SearchResult{
			ID:       uint64(res.ID),
			Distance: res.Distance,
		}
	}
	return results, nil
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

// flushMemTable flushes the MemTable queue to the main index.
// Returns true if a flush was performed, false if skipped.
func (tx *Tx[T]) flushMemTable() bool {
	// 1. Rotate active to queue if not empty
	tx.memMu.Lock()
	state := tx.memState.Load()
	rotated := false
	if state.active.Size() > 0 {
		newActive := memtable.New(tx.dimension, tx.distFunc)
		newQueue := make([]*memtable.MemTable, len(state.queue)+1)
		copy(newQueue, state.queue)
		newQueue[len(state.queue)] = state.active

		newState := &memState{
			active: newActive,
			queue:  newQueue,
		}
		tx.memState.Store(newState)
		rotated = true
	}
	tx.memMu.Unlock()

	if rotated {
		// Signal waiting writers that active memtable is empty
		tx.flushMu.Lock()
		tx.flushCond.Broadcast()
		tx.flushMu.Unlock()
	}

	// Serialize flush actions to prevent concurrent writes to HNSW
	tx.flushActionMu.Lock()
	defer tx.flushActionMu.Unlock()

	flushedAny := false
	ctx := context.Background()

	for {
		tx.memMu.RLock()
		state = tx.memState.Load()
		if len(state.queue) == 0 {
			tx.memMu.RUnlock()
			break
		}
		tableToFlush := state.queue[0]
		tx.memMu.RUnlock()

		// Get items to flush
		items := tableToFlush.Items()

		// Deduplicate items (latest wins)
		unique := make(map[core.LocalID]memtable.Item)
		for _, item := range items {
			unique[item.ID] = item
		}

		var insertIDs []core.LocalID
		var insertVecs [][]float32

		for _, item := range unique {
			if item.IsDeleted {
				err := tx.txIndex.ApplyDelete(ctx, item.ID)
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
				insertIDs = append(insertIDs, item.ID)
				insertVecs = append(insertVecs, item.Vector)
			}
		}

		if len(insertIDs) > 0 {
			if err := tx.txIndex.ApplyBatchInsert(ctx, insertIDs, insertVecs); err != nil {
				// In a production system, we should log this error properly.
			}
		}

		// Remove from queue
		tx.memMu.Lock()
		state = tx.memState.Load()
		if len(state.queue) > 0 && state.queue[0] == tableToFlush {
			newQueue := make([]*memtable.MemTable, len(state.queue)-1)
			copy(newQueue, state.queue[1:])
			newState := &memState{
				active: state.active,
				queue:  newQueue,
			}
			tx.memState.Store(newState)
			flushedAny = true
		}
		tx.memMu.Unlock()
	}

	return flushedAny
}

// HybridSearch performs a hybrid search combining vector similarity and metadata filtering.
func (tx *Tx[T]) HybridSearch(ctx context.Context, query []float32, k int, opts *HybridSearchOptions) ([]SearchResult, error) {
	// Use a temporary searcher for the operation to benefit from zero-alloc filtering
	// We guess a capacity, it will grow if needed
	s := searcher.Get()
	defer searcher.Put(s)

	return tx.HybridSearchWithContext(ctx, query, k, opts, s)
}

// HybridSearchWithContext performs a hybrid search using the provided Searcher context.
// This allows reusing the Searcher's scratch buffers for metadata filtering.
func (tx *Tx[T]) HybridSearchWithContext(ctx context.Context, query []float32, k int, opts *HybridSearchOptions, s *searcher.Searcher) ([]SearchResult, error) {
	if opts == nil {
		opts = &HybridSearchOptions{EF: 0}
	}

	// If no metadata filters, fall back to regular KNN search
	if opts.MetadataFilters == nil || len(opts.MetadataFilters.Filters) == 0 {
		err := tx.KNNSearchWithContext(ctx, query, k, &index.SearchOptions{EFSearch: opts.EF}, s)
		if err != nil {
			return nil, err
		}

		// Zero-alloc conversion
		s.ScratchResults = s.ScratchResults[:0]
		for s.Candidates.Len() > 0 {
			item, _ := s.Candidates.PopItem()
			s.ScratchResults = append(s.ScratchResults, item)
		}
		// Reverse
		for i, j := 0, len(s.ScratchResults)-1; i < j; i, j = i+1, j-1 {
			s.ScratchResults[i], s.ScratchResults[j] = s.ScratchResults[j], s.ScratchResults[i]
		}

		results := make([]SearchResult, len(s.ScratchResults))
		for i, item := range s.ScratchResults {
			results[i] = SearchResult{ID: uint64(item.Node), Distance: item.Distance}
		}
		return results, nil
	}

	// Create metadata filter function
	var metadataFilter func(core.LocalID) bool

	if tx.metaStore != nil {
		// Try to compile to bitmap using Searcher's scratch bitmap
		if tx.metaStore.CompileFilterTo(opts.MetadataFilters, s.FilterBitmap) {
			// Fast path: use bitmap
			bitmap := s.FilterBitmap
			metadataFilter = func(id core.LocalID) bool {
				return bitmap.Contains(id)
			}
		} else {
			// Fallback to streaming filter (slow path)
			tx.metaStore.RLock()
			metadataFilter = tx.metaStore.CreateStreamingFilter(opts.MetadataFilters)
			tx.metaStore.RUnlock()
		}
	} else {
		// Should not happen if initialized correctly, but safe fallback
		return []SearchResult{}, nil
	}

	if opts.PreFilter {
		// Pre-filtering
		var filter func(core.LocalID) bool
		if metadataFilter != nil {
			filter = func(id core.LocalID) bool {
				return metadataFilter(id)
			}
		}
		searchOpts := &index.SearchOptions{EFSearch: opts.EF, Filter: filter}
		err := tx.KNNSearchWithContext(ctx, query, k, searchOpts, s)
		if err != nil {
			return nil, err
		}

		// Zero-alloc conversion using ScratchResults
		s.ScratchResults = s.ScratchResults[:0]
		for s.Candidates.Len() > 0 {
			item, _ := s.Candidates.PopItem()
			s.ScratchResults = append(s.ScratchResults, item)
		}
		// Reverse
		for i, j := 0, len(s.ScratchResults)-1; i < j; i, j = i+1, j-1 {
			s.ScratchResults[i], s.ScratchResults[j] = s.ScratchResults[j], s.ScratchResults[i]
		}

		results := make([]SearchResult, len(s.ScratchResults))
		for i, item := range s.ScratchResults {
			results[i] = SearchResult{ID: uint64(item.Node), Distance: item.Distance}
		}
		return results, nil
	}

	// Post-filtering
	oversampleK := min(k*3, 1000)
	searchOpts := &index.SearchOptions{EFSearch: opts.EF}
	err := tx.KNNSearchWithContext(ctx, query, oversampleK, searchOpts, s)
	if err != nil {
		return nil, err
	}

	// Apply metadata filtering
	// Use ScratchResults to avoid allocation during sorting
	s.ScratchResults = s.ScratchResults[:0]
	for s.Candidates.Len() > 0 {
		item, _ := s.Candidates.PopItem()
		s.ScratchResults = append(s.ScratchResults, item)
	}

	// Reverse to get best-first (MaxHeap pops worst-first)
	for i, j := 0, len(s.ScratchResults)-1; i < j; i, j = i+1, j-1 {
		s.ScratchResults[i], s.ScratchResults[j] = s.ScratchResults[j], s.ScratchResults[i]
	}

	results := make([]SearchResult, 0, k)
	for _, item := range s.ScratchResults {
		if len(results) >= k {
			break
		}
		if metadataFilter(item.Node) {
			results = append(results, SearchResult{
				ID:       uint64(item.Node),
				Distance: item.Distance,
			})
		}
	}
	return results, nil
}

// KNNSearchStream returns an iterator over K-nearest neighbor search results.
func (tx *Tx[T]) KNNSearchStream(ctx context.Context, query []float32, k int, opts *SearchOptions) iter.Seq2[SearchResult, error] {
	// Capture MemTables safely (Atomic Snapshot)
	state := tx.memState.Load()
	memT := state.active
	queue := state.queue

	// Convert SearchOptions
	var indexOpts *index.SearchOptions
	if opts != nil {
		indexOpts = &index.SearchOptions{
			EFSearch: opts.EFSearch,
		}
		if opts.Filter != nil {
			indexOpts.Filter = func(id core.LocalID) bool {
				return opts.Filter(uint64(id))
			}
		}
	}

	// 1. Stream from Main Index (filtered)
	indexStream := tx.txIndex.KNNSearchStream(ctx, query, k, indexOpts)

	filteredIndexStream := func(yield func(index.SearchResult, error) bool) {
		for res, err := range indexStream {
			if err != nil {
				if !yield(res, err) {
					return
				}
				continue
			}
			localID := core.LocalID(res.ID)
			// Check MemTable
			if _, found, isDeleted := memT.Get(localID); found && isDeleted {
				continue
			}
			// Check Queue
			inQueue := false
			for _, t := range queue {
				if _, found, isDeleted := t.Get(localID); found && isDeleted {
					inQueue = true
					break
				}
			}
			if inQueue {
				continue
			}
			if !yield(res, nil) {
				return
			}
		}
	}

	// 2. Search MemTables (returns slice)
	var filter func(core.LocalID) bool
	if indexOpts != nil && indexOpts.Filter != nil {
		filter = func(id core.LocalID) bool {
			return indexOpts.Filter(id)
		}
	}
	memResults := memT.Search(query, k, filter)

	var queueResults []index.SearchResult
	for _, t := range queue {
		res := t.Search(query, k, filter)
		if len(res) > 0 {
			queueResults = append(queueResults, res...)
		}
	}
	if len(queueResults) > 0 {
		sort.Slice(queueResults, func(i, j int) bool {
			return queueResults[i].Distance < queueResults[j].Distance
		})
	}

	// 3. Merge Streams
	allMemResults := memResults
	if len(queueResults) > 0 {
		allMemResults = index.MergeSearchResults(memResults, queueResults, k)
	}

	var mergedStream iter.Seq2[index.SearchResult, error]
	if len(allMemResults) == 0 {
		mergedStream = filteredIndexStream
	} else {
		memStream := index.SliceToStream(allMemResults)
		mergedStream = index.MergeSearchStreams(filteredIndexStream, memStream)
	}

	// Convert to engine.SearchResult
	return func(yield func(SearchResult, error) bool) {
		for res, err := range mergedStream {
			if err != nil {
				if !yield(SearchResult{}, err) {
					return
				}
				continue
			}
			if !yield(SearchResult{ID: uint64(res.ID), Distance: res.Distance}, nil) {
				return
			}
		}
	}
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

	queueSize := 0
	for _, t := range state.queue {
		queueSize += t.Size()
	}
	stats.Storage["MemTableQueueSize"] = fmt.Sprintf("%d", queueSize)
	stats.Storage["MemTableQueueCount"] = fmt.Sprintf("%d", len(state.queue))

	return stats
}

// Checkpoint creates a checkpoint in the durability layer (WAL).
func (tx *Tx[T]) Checkpoint() error {
	if w, ok := tx.durability.(*wal.WAL); ok {
		return w.Checkpoint()
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
