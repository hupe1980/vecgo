package memtable

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/hupe1980/vecgo/distance"
	"github.com/hupe1980/vecgo/internal/arena"
	imetadata "github.com/hupe1980/vecgo/internal/metadata"
	"github.com/hupe1980/vecgo/internal/searcher"
	"github.com/hupe1980/vecgo/internal/segment"
	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/model"
)

const (
	shardBits  = 4
	shardCount = 1 << shardBits // 16
	shardMask  = shardCount - 1
	rowIdMask  = (1 << (32 - shardBits)) - 1 // 28 bits
)

// fetchReq represents a fetch request with the original index and shard-local row ID.
type fetchReq struct {
	origIdx int
	rowID   uint32
}

// fetchReqSlices is used for sync.Pool to avoid SA6002 allocation issue.
type fetchReqSlices [][]fetchReq

// Pool for rerank shard dispatch slices to reduce allocations in hot path
var rerankByShardPool = sync.Pool{
	New: func() any {
		// Pre-allocate slice of slices
		s := make([][]model.Candidate, shardCount)
		return &s
	},
}

// Pool for fetch shard request dispatch slices
var fetchShardReqPool = sync.Pool{
	New: func() any {
		s := make(fetchReqSlices, shardCount)
		return &s
	},
}

type MemTable struct {
	id     model.SegmentID // This L0 segment ID
	shards []*shard
	refs   int64
}

// Close releases resources held by the memtable.
func (m *MemTable) Close() error {
	var errs []error
	for _, s := range m.shards {
		if err := s.Close(); err != nil {
			errs = append(errs, err)
		}
	}
	if len(errs) > 0 {
		return fmt.Errorf("memtable close errors: %v", errs)
	}
	return nil
}

func New(ctx context.Context, id model.SegmentID, dim int, metric distance.Metric, acquirer arena.MemoryAcquirer) (*MemTable, error) {
	shards := make([]*shard, shardCount)
	for i := range shardCount {
		s, err := newShard(ctx, id, uint8(i), dim, metric, acquirer)
		if err != nil {
			// Cleanup already created shards
			for j := 0; j < i; j++ {
				shards[j].Close()
			}
			return nil, err
		}
		shards[i] = s
	}
	return &MemTable{
		id:     id,
		shards: shards,
		refs:   1,
	}, nil
}

func (m *MemTable) IncRef() {
	atomic.AddInt64(&m.refs, 1)
}

func (m *MemTable) DecRef() {
	if atomic.AddInt64(&m.refs, -1) == 0 {
		m.Close()
	}
}

func (m *MemTable) ID() model.SegmentID {
	return m.id
}

func (m *MemTable) RowCount() uint32 {
	var count uint32
	for _, s := range m.shards {
		count += s.RowCount()
	}
	return count
}

func (m *MemTable) Metric() distance.Metric {
	if len(m.shards) > 0 {
		return m.shards[0].Metric()
	}
	return distance.MetricL2
}

func (m *MemTable) Size() int64 {
	var size int64
	for _, s := range m.shards {
		size += s.Size()
	}
	return size
}

// Advise is a no-op
func (m *MemTable) Advise(pattern segment.AccessPattern) error {
	return nil
}

// Write/Update Ops

func (m *MemTable) Insert(ctx context.Context, id model.ID, vec []float32) (model.RowID, error) {
	return m.InsertWithPayload(ctx, id, vec, nil, nil)
}

func (m *MemTable) InsertWithPayload(ctx context.Context, id model.ID, vec []float32, md metadata.Document, payload []byte) (model.RowID, error) {
	shardIdx := int(id & shardMask) // e.g. id % 16
	s := m.shards[shardIdx]

	localRowID, err := s.InsertWithPayload(ctx, id, vec, md, payload)
	if err != nil {
		return 0, err
	}

	// Encode Global RowID
	if localRowID > rowIdMask {
		return 0, fmt.Errorf("local rowID overflow")
	}

	return (model.RowID(shardIdx) << 28) | localRowID, nil
}

// BatchInsertWithPayload inserts multiple vectors efficiently by partitioning by shard
// and processing each shard in parallel. This minimizes lock contention and amortizes
// overhead compared to individual inserts.
func (m *MemTable) BatchInsertWithPayload(ctx context.Context, ids []model.ID, vectors [][]float32, mds []metadata.Document, payloads [][]byte) ([]model.RowID, error) {
	n := len(ids)
	if n == 0 {
		return nil, nil
	}

	// Pre-allocate result slice
	rowIDs := make([]model.RowID, n)

	// Partition by shard
	type shardItem struct {
		origIdx int
		id      model.ID
		vec     []float32
		md      metadata.Document
		payload []byte
	}
	shardBatches := make([][]shardItem, shardCount)
	for i := range shardBatches {
		shardBatches[i] = make([]shardItem, 0, n/shardCount+1)
	}

	for i, id := range ids {
		shardIdx := int(id & shardMask)
		var md metadata.Document
		var payload []byte
		if mds != nil && i < len(mds) {
			md = mds[i]
		}
		if payloads != nil && i < len(payloads) {
			payload = payloads[i]
		}
		shardBatches[shardIdx] = append(shardBatches[shardIdx], shardItem{
			origIdx: i,
			id:      id,
			vec:     vectors[i],
			md:      md,
			payload: payload,
		})
	}

	// Process shards in parallel
	var wg sync.WaitGroup
	var errMu sync.Mutex
	var firstErr error

	for shardIdx := range shardCount {
		batch := shardBatches[shardIdx]
		if len(batch) == 0 {
			continue
		}

		wg.Add(1)
		go func(shardIdx int, batch []shardItem) {
			defer wg.Done()

			s := m.shards[shardIdx]

			// Process all items in this shard under a single lock acquisition
			s.mu.Lock()
			defer s.mu.Unlock()

			if s.idx == nil {
				errMu.Lock()
				if firstErr == nil {
					firstErr = fmt.Errorf("memtable shard %d is closed", shardIdx)
				}
				errMu.Unlock()
				return
			}

			for _, item := range batch {
				// Check for cancellation
				if err := ctx.Err(); err != nil {
					errMu.Lock()
					if firstErr == nil {
						firstErr = err
					}
					errMu.Unlock()
					return
				}

				// Check for previous error
				errMu.Lock()
				if firstErr != nil {
					errMu.Unlock()
					return
				}
				errMu.Unlock()

				// Insert into HNSW
				localRowID, err := s.idx.Insert(ctx, item.vec)
				if err != nil {
					errMu.Lock()
					if firstErr == nil {
						firstErr = err
					}
					errMu.Unlock()
					return
				}

				// Commit metadata (inlined from commitInsert to avoid lock overhead)
				var interned metadata.InternedDocument
				if item.md != nil {
					interned = metadata.Intern(item.md)
				}

				idx := int(localRowID)
				expected := s.ids.Count()

				if uint64(idx) > expected {
					gap := uint64(idx) - expected
					for j := range gap {
						_ = s.idx.Delete(context.Background(), model.RowID(expected+j))
						s.ids.Append(0)
						s.metadata.Append(metadata.InternedDocument{})
						s.payloads.Append(nil)
					}
				} else if uint64(idx) < expected {
					errMu.Lock()
					if firstErr == nil {
						firstErr = fmt.Errorf("memtable rowID out of sync: expected %d, got %d", expected, idx)
					}
					errMu.Unlock()
					return
				}

				s.ids.Append(item.id)
				s.metadata.Append(interned)
				s.payloads.Append(item.payload)

				// Sync columnar metadata
				targetLen := idx + 1
				for key, col := range s.columns {
					val, hasVal := item.md[key]
					var v metadata.Value
					if hasVal {
						v = val
					}
					for col.Len() < targetLen-1 {
						col.Append(metadata.Value{})
					}
					if col.Len() == targetLen {
						col.Set(idx, v)
					} else {
						col.Append(v)
					}
				}
				for key, val := range item.md {
					if _, exists := s.columns[key]; !exists {
						col, err := createColumn(val.Kind, 1024)
						if err == nil {
							col.Grow(idx)
							col.Append(val)
							s.columns[key] = col
						}
					}
				}

				// Encode global RowID
				if localRowID > rowIdMask {
					errMu.Lock()
					if firstErr == nil {
						firstErr = fmt.Errorf("local rowID overflow")
					}
					errMu.Unlock()
					return
				}

				rowIDs[item.origIdx] = (model.RowID(shardIdx) << 28) | localRowID
			}
		}(shardIdx, batch)
	}

	wg.Wait()

	if firstErr != nil {
		return nil, firstErr
	}

	return rowIDs, nil
}

func (m *MemTable) InsertDeferred(ctx context.Context, id model.ID, vec []float32, md metadata.Document, payload []byte) (model.RowID, error) {
	shardIdx := int(id & shardMask)
	s := m.shards[shardIdx]

	localRowID, err := s.InsertDeferred(ctx, id, vec, md, payload)
	if err != nil {
		return 0, err
	}

	if localRowID > rowIdMask {
		return 0, fmt.Errorf("local rowID overflow")
	}

	return (model.RowID(shardIdx) << 28) | localRowID, nil
}

func (m *MemTable) Delete(rowID model.RowID) {
	shardIdx := int(rowID >> 28)
	localRowID := rowID & rowIdMask

	if shardIdx < len(m.shards) {
		m.shards[shardIdx].Delete(localRowID)
	}
}

// Read Ops

func (m *MemTable) GetID(ctx context.Context, rowID uint32) (model.ID, bool) {
	shardIdx := int(rowID >> 28)
	localRowID := rowID & rowIdMask

	if shardIdx < len(m.shards) {
		return m.shards[shardIdx].GetID(ctx, localRowID)
	}
	return 0, false
}

func (m *MemTable) Search(ctx context.Context, q []float32, k int, filter segment.Filter, opts model.SearchOptions, searcherCtx *searcher.Searcher) error {
	// Search all shards
	// Sequential loop is fine as it avoids overhead of goroutines for MemTable (usually small latency)
	// And allows re-using the same searcher/heap without mutex.
	// Actually, wait. The Heap is global. `shard.Search` pushes to `searcherCtx.Heap`.
	// Correct.

	// Track that this is an HNSW segment search and track timing
	if searcherCtx != nil {
		searcherCtx.FilterGateStats.HNSWSegments++
		searchStart := time.Now()
		defer func() {
			searcherCtx.FilterGateStats.SearchTimeNanos += time.Since(searchStart).Nanoseconds()
		}()
	}

	for _, s := range m.shards {
		if err := s.Search(ctx, q, k, filter, opts, searcherCtx); err != nil {
			return err
		}
	}
	return nil
}

func (m *MemTable) Rerank(ctx context.Context, q []float32, cands []model.Candidate, dst []model.Candidate) ([]model.Candidate, error) {
	// We need to route candidates to shards
	// But `shard.Rerank` expects a slice of candidates.
	// We could split the slice OR just iterate shards and let them filter by `c.Loc.RowID`.
	// But `shard.Rerank` filters by `c.Loc.SegmentID`.
	// Both checks pass (same SegmentID).
	// We need check by `RowID` shard index.
	// `shard.Rerank` doesn't check shard index currently.
	// It assumes it owns the SegmentID.
	// Check `shard.Rerank`:
	// if c.Loc.SegmentID != m.id { continue }
	// rowID := uint32(c.Loc.RowID)
	// vec, ok := m.vectors.GetVector(model.RowID(rowID))

	// If we pass a RowID belonging to Shard 1 to Shard 0:
	// Shard 0: localRowID = RowID (if we don't mask).
	// But global RowID has high bits set for Shard 1.
	// Shard 0 will try to access vector at very high index (overflow or out of bounds).
	// So we MUST dispatch correctly or update `shard.Rerank` to mask.

	// Update `shard.Rerank`?
	// `shard.Rerank` uses `m.vectors.GetVector`.
	// `VectorStore` assumes 0-based indexing.
	// So we should group candidates by shard here.

	// Get pooled slice of slices
	byShardPtr := rerankByShardPool.Get().(*[][]model.Candidate)
	byShard := *byShardPtr

	// Clear previous contents
	for i := range byShard {
		byShard[i] = byShard[i][:0]
	}

	for _, c := range cands {
		shardIdx := int(c.Loc.RowID >> 28)
		if shardIdx < shardCount {
			localC := c
			localC.Loc.RowID = c.Loc.RowID & rowIdMask
			byShard[shardIdx] = append(byShard[shardIdx], localC)
		}
	}

	startIdx := len(dst)
	for i, s := range m.shards {
		if len(byShard[i]) > 0 {
			var err error
			dst, err = s.Rerank(ctx, q, byShard[i], dst)
			if err != nil {
				rerankByShardPool.Put(byShardPtr)
				return nil, err
			}
			// Fixup the newly added candidates
			for k := startIdx; k < len(dst); k++ {
				// dst[k] has Local RowID.
				// Restore Global.
				dst[k].Loc.RowID = (model.RowID(i) << 28) | dst[k].Loc.RowID
			}
			startIdx = len(dst)
		}
	}

	rerankByShardPool.Put(byShardPtr)
	return dst, nil
}

func (m *MemTable) Fetch(ctx context.Context, rows []uint32, cols []string) (segment.RecordBatch, error) {
	total := len(rows)

	// Detect columns
	fetchVectors := cols == nil
	fetchMetadata := cols == nil
	fetchPayload := cols == nil
	if cols != nil {
		fetchVectors = false
		for _, c := range cols {
			switch c {
			case "vector":
				fetchVectors = true
			case "metadata":
				fetchMetadata = true
			case "payload":
				fetchPayload = true
			}
		}
	}

	// Allocate the result batch once - shards write directly into it
	resBatch := &segment.SimpleRecordBatch{
		IDs: make([]model.ID, total),
	}
	if fetchVectors {
		resBatch.Vectors = make([][]float32, total)
	}
	if fetchMetadata {
		resBatch.Metadatas = make([]metadata.Document, total)
	}
	if fetchPayload {
		resBatch.Payloads = make([][]byte, total)
	}

	// Get pooled shard request slices
	shardReqsPtr := fetchShardReqPool.Get().(*fetchReqSlices)
	shardReqs := *shardReqsPtr

	// Dispatch rows to shards
	for i, r := range rows {
		sIdx := r >> 28
		if int(sIdx) >= shardCount {
			// Reset and return to pool before returning error
			for j := range shardReqs {
				shardReqs[j] = shardReqs[j][:0]
			}
			fetchShardReqPool.Put(shardReqsPtr)
			return nil, fmt.Errorf("invalid rowID %d", r)
		}
		shardReqs[sIdx] = append(shardReqs[sIdx], fetchReq{i, r & uint32(rowIdMask)})
	}

	// Process each shard's requests
	for sIdx, reqs := range shardReqs {
		if len(reqs) == 0 {
			continue
		}

		// Use FetchInto to write directly into resBatch, avoiding intermediate allocation
		if err := m.shards[sIdx].FetchIntoReqs(ctx, reqs, resBatch); err != nil {
			// Reset and return to pool before returning error
			for j := range shardReqs {
				shardReqs[j] = shardReqs[j][:0]
			}
			fetchShardReqPool.Put(shardReqsPtr)
			return nil, err
		}
	}

	// Reset slices and return to pool
	for j := range shardReqs {
		shardReqs[j] = shardReqs[j][:0]
	}
	fetchShardReqPool.Put(shardReqsPtr)

	return resBatch, nil
}

func (m *MemTable) FetchIDs(ctx context.Context, rows []uint32, dst []model.ID) error {
	if len(dst) != len(rows) {
		return fmt.Errorf("dst length must match rows length")
	}

	// Fast path: directly call GetID for each row (avoids allocations)
	for i, r := range rows {
		sIdx := r >> 28
		if int(sIdx) >= shardCount {
			return fmt.Errorf("invalid rowID")
		}
		localRowID := r & uint32(rowIdMask)
		if id, ok := m.shards[sIdx].GetID(ctx, localRowID); ok {
			dst[i] = id
		} else {
			dst[i] = 0 // Invalid/deleted ID
		}
	}
	return nil
}

// FetchVectorsInto copies vectors for the given rows into dst.
// dst must have len >= len(rows)*dim.
// Returns validMask indicating which rows have valid vectors (nil if all valid).
func (m *MemTable) FetchVectorsInto(_ context.Context, rows []uint32, dim int, dst []float32) ([]bool, error) {
	if len(dst) < len(rows)*dim {
		return nil, fmt.Errorf("dst too small: need %d, got %d", len(rows)*dim, len(dst))
	}

	var hasInvalid bool
	var validMask []bool

	for i, r := range rows {
		sIdx := r >> 28
		if int(sIdx) >= shardCount {
			return nil, fmt.Errorf("invalid rowID")
		}
		localRowID := r & uint32(rowIdMask)

		// Directly get vector from shard
		m.shards[sIdx].mu.RLock()
		if m.shards[sIdx].idx == nil {
			m.shards[sIdx].mu.RUnlock()
			if !hasInvalid {
				hasInvalid = true
				validMask = make([]bool, len(rows))
				for j := 0; j < i; j++ {
					validMask[j] = true
				}
			}
			continue
		}
		vec, ok := m.shards[sIdx].vectors.GetVector(model.RowID(localRowID))
		if !ok || vec == nil {
			m.shards[sIdx].mu.RUnlock()
			if !hasInvalid {
				hasInvalid = true
				validMask = make([]bool, len(rows))
				for j := 0; j < i; j++ {
					validMask[j] = true
				}
			}
			continue
		}

		// Copy into caller's buffer
		dstSlice := dst[i*dim : (i+1)*dim]
		copy(dstSlice, vec)
		m.shards[sIdx].mu.RUnlock()

		if hasInvalid {
			validMask[i] = true
		}
	}

	return validMask, nil
}

// FetchVectorDirect is not supported for MemTable due to locking requirements.
// Returns nil. Use FetchVectorsInto instead.
func (m *MemTable) FetchVectorDirect(rowID uint32) []float32 {
	return nil // Not supported for MemTable
}

// EvaluateFilter returns a bitmap of rows matching the filter.
func (m *MemTable) EvaluateFilter(ctx context.Context, filter *metadata.FilterSet) (segment.Bitmap, error) {
	if filter == nil || len(filter.Filters) == 0 {
		return nil, nil // All matches
	}

	result := imetadata.GetPooledBitmap() // Use pooled bitmap

	// Scratch buffer for shard rowIDs (avoid Iterator allocation)
	var scratchIDs []uint32

	for i, s := range m.shards {
		shardBitmap, err := s.EvaluateFilter(ctx, filter)
		if err != nil {
			imetadata.PutPooledBitmap(result)
			return nil, err
		}

		if shardBitmap != nil {
			// Shard-local offsets need to be shifted
			shardOffset := uint32(i) << 28

			// Use ToArrayInto to avoid Iterator allocation
			scratchIDs = shardBitmap.ToArrayInto(scratchIDs[:0])
			for _, id := range scratchIDs {
				result.Add(id | shardOffset)
			}
			// Return shard bitmap to pool
			imetadata.PutPooledBitmap(shardBitmap)
		}
	}
	return result, nil
}

// EvaluateFilterResult returns filter results using the zero-alloc FilterResult type.
// This is the optimized path that avoids roaring bitmaps for low cardinality.
// Uses QueryScratch for scratch space (zero allocations in steady state).
func (m *MemTable) EvaluateFilterResult(ctx context.Context, filter *metadata.FilterSet, qs *imetadata.QueryScratch) (imetadata.FilterResult, error) {
	if filter == nil || len(filter.Filters) == 0 {
		return imetadata.AllResult(), nil // No filter = match all
	}

	// For memtable with shards, we cannot use qs.TmpRowIDs directly because
	// each shard call would overwrite the previous shard's results.
	// Use TmpRowIDs2 as the accumulator (zero-alloc after warmup).
	accumulated := qs.TmpRowIDs2[:0]

	for i, s := range m.shards {
		// Each shard writes to qs.TmpRowIDs, so we accumulate to TmpRowIDs2
		shardResult, err := s.EvaluateFilterResult(ctx, filter, qs)
		if err != nil {
			return imetadata.EmptyResult(), err
		}

		// Handle FilterAll (shouldn't happen since we checked filter above, but be defensive)
		if shardResult.IsAll() {
			// AllResult from shard = match all in that shard
			// We need to fallback to HNSW for this segment
			return imetadata.AllResult(), nil
		}

		if !shardResult.IsEmpty() {
			// Shard-local offsets need to be shifted to global offsets
			shardOffset := uint32(i) << 28

			// Copy rows with offset applied
			shardRows := shardResult.Rows()
			for _, id := range shardRows {
				accumulated = append(accumulated, id|shardOffset)
			}
		}
	}

	if len(accumulated) == 0 {
		return imetadata.EmptyResult(), nil
	}

	// Store back to TmpRowIDs2 for capacity growth tracking
	qs.TmpRowIDs2 = accumulated

	// Copy to TmpRowIDs for the caller (this is the contract)
	qs.TmpRowIDs = append(qs.TmpRowIDs[:0], accumulated...)
	return imetadata.RowsResult(qs.TmpRowIDs), nil
}

// FilterCursor returns a push-based cursor for streaming filtered search.
// This is the zero-allocation hot path that eliminates roaring bitmap overhead.
// The cursor evaluates filters lazily during iteration - no materialization.
//
// Usage:
//
//	cursor := m.FilterCursor(filter)
//	cursor.ForEach(func(rowID uint32) bool {
//	    // rowID is global (includes shard prefix)
//	    score := computeDistance(query, vectors[rowID])
//	    heap.Push(rowID, score)
//	    return true // continue
//	})
func (m *MemTable) FilterCursor(filter *metadata.FilterSet) imetadata.FilterCursor {
	if filter == nil || len(filter.Filters) == 0 {
		// No filter = all rows match, return multi-shard all cursor
		return &multiShardAllCursor{shards: m.shards}
	}

	return &multiShardFilterCursor{
		shards: m.shards,
		filter: filter,
	}
}

// multiShardAllCursor iterates all rows across all shards.
type multiShardAllCursor struct {
	shards []*shard
}

func (c *multiShardAllCursor) ForEach(fn func(rowID uint32) bool) {
	for sIdx, s := range c.shards {
		shardOffset := uint32(sIdx) << 28
		s.mu.RLock()
		count := uint32(s.ids.Count())
		s.mu.RUnlock()

		for id := uint32(0); id < count; id++ {
			if !fn(id | shardOffset) {
				return
			}
		}
	}
}

func (c *multiShardAllCursor) EstimateCardinality() int {
	total := 0
	for _, s := range c.shards {
		s.mu.RLock()
		total += int(s.ids.Count())
		s.mu.RUnlock()
	}
	return total
}

func (c *multiShardAllCursor) IsEmpty() bool {
	for _, s := range c.shards {
		s.mu.RLock()
		count := s.ids.Count()
		s.mu.RUnlock()
		if count > 0 {
			return false
		}
	}
	return true
}

func (c *multiShardAllCursor) IsAll() bool { return true }

// multiShardFilterCursor iterates filtered rows across all shards.
type multiShardFilterCursor struct {
	shards []*shard
	filter *metadata.FilterSet
}

func (c *multiShardFilterCursor) ForEach(fn func(rowID uint32) bool) {
	for sIdx, s := range c.shards {
		shardOffset := uint32(sIdx) << 28

		s.mu.RLock()
		count := int(s.ids.Count())
		columns := s.columns
		filters := c.filter.Filters

		// Check if all filter columns exist in this shard
		allColumnsExist := true
		for _, f := range filters {
			if _, ok := columns[f.Key]; !ok {
				allColumnsExist = false
				break
			}
		}

		if !allColumnsExist {
			s.mu.RUnlock()
			continue // No matches in this shard
		}

	rowLoop:
		for id := 0; id < count; id++ {
			// Check all filters (AND logic) with short-circuit
			for _, f := range filters {
				col := columns[f.Key]
				if !col.Matches(id, f.Value, f.Operator) {
					continue rowLoop
				}
			}
			// Passed all filters
			globalID := uint32(id) | shardOffset
			if !fn(globalID) {
				s.mu.RUnlock()
				return
			}
		}
		s.mu.RUnlock()
	}
}

func (c *multiShardFilterCursor) EstimateCardinality() int {
	total := 0
	for _, s := range c.shards {
		s.mu.RLock()
		count := int(s.ids.Count())
		s.mu.RUnlock()
		// Rough heuristic: 10% selectivity per filter
		estimate := count
		for range c.filter.Filters {
			estimate = estimate * 10 / 100
			if estimate < 1 {
				estimate = 1
			}
		}
		total += estimate
	}
	return total
}

func (c *multiShardFilterCursor) IsEmpty() bool {
	for _, s := range c.shards {
		s.mu.RLock()
		count := s.ids.Count()
		s.mu.RUnlock()
		if count > 0 {
			return false
		}
	}
	return true
}

func (c *multiShardFilterCursor) IsAll() bool { return false }

func (m *MemTable) Iterate(ctx context.Context, fn func(rowID uint32, id model.ID, vec []float32, md metadata.Document, payload []byte) error) error {
	// Iterate all shards
	for sIdx, s := range m.shards {
		// Check context between shards
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		// We need to yield Global RowIDs
		// shard.Iterate yields Local RowIds.
		base := uint32(sIdx) << 28

		err := s.Iterate(ctx, func(rid uint32, id model.ID, v []float32, md metadata.Document, p []byte) error {
			globalRID := base | rid
			return fn(globalRID, id, v, md, p)
		})
		if err != nil {
			return err
		}
	}
	return nil
}
