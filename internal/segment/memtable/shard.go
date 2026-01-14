package memtable

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"

	"github.com/hupe1980/vecgo/distance"
	"github.com/hupe1980/vecgo/internal/arena"
	idxhnsw "github.com/hupe1980/vecgo/internal/hnsw"
	imetadata "github.com/hupe1980/vecgo/internal/metadata"
	"github.com/hupe1980/vecgo/internal/searcher"
	"github.com/hupe1980/vecgo/internal/segment"
	"github.com/hupe1980/vecgo/internal/vectorstore"
	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/model"
)

// shard is a mutable in-memory segment partition backed by HNSW.
type shard struct {
	mu       sync.RWMutex
	refs     int64
	id       model.SegmentID // Virtual ID (same as MemTable ID)? Or derived?
	shardIdx uint8           // Index of this shard (0-15)
	dim      int
	metric   distance.Metric
	idx      *idxhnsw.HNSW
	vectors  vectorstore.VectorStore
	ids      *PagedIDStore
	metadata *PagedMetaStore
	payloads *PagedPayloadStore
	columns  map[string]Column
}

// Close releases resources held by the shard.
func (s *shard) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	var errs []error

	if s.idx != nil {
		if err := s.idx.Close(); err != nil {
			errs = append(errs, err)
		}
		s.idx = nil
	}

	if s.vectors != nil {
		if err := s.vectors.Close(); err != nil {
			errs = append(errs, err)
		}
		s.vectors = nil
	}

	if len(errs) > 0 {
		return fmt.Errorf("shard close errors: %v", errs)
	}
	return nil
}

// newShard creates a new shard.
func newShard(ctx context.Context, id model.SegmentID, shardIdx uint8, dim int, metric distance.Metric, acquirer arena.MemoryAcquirer) (*shard, error) {
	store, err := vectorstore.New(dim, acquirer)
	if err != nil {
		return nil, err
	}

	// Preallocate memory with caller-controlled context
	if err := store.Preallocate(ctx); err != nil {
		return nil, err
	}

	// Configure HNSW for L0:
	// - M=32, EF=300 (tuned for higher recall)
	// - Use the shared vector store
	opts := idxhnsw.DefaultOptions
	opts.M = 32
	opts.EF = 300
	opts.Dimension = dim
	opts.DistanceType = metric
	opts.Vectors = store
	opts.MemoryAcquirer = acquirer
	opts.InitialArenaSize = 256 * 1024 // 256KB initial chunk size (16 shards = 4MB total)

	h, err := idxhnsw.New(func(o *idxhnsw.Options) {
		*o = opts
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create memtable hnsw: %w", err)
	}

	m := &shard{
		refs:     1,
		id:       id,
		shardIdx: shardIdx,
		dim:      dim,
		metric:   metric,
		idx:      h,
		vectors:  store,
		columns:  make(map[string]Column),
		ids:      NewPagedIDStore(),
		metadata: NewPagedMetaStore(),
		payloads: NewPagedPayloadStore(),
	}

	return m, nil
}

// IncRef increments the reference count.
func (s *shard) IncRef() {
	atomic.AddInt64(&s.refs, 1)
}

// DecRef decrements the reference count and closes the memtable if it reaches zero.
func (s *shard) DecRef() {
	if atomic.AddInt64(&s.refs, -1) == 0 {
		s.Close()
	}
}

// Insert adds a vector to the memtable.
// Returns the assigned RowID.
func (s *shard) Insert(ctx context.Context, id model.ID, vec []float32) (model.RowID, error) {
	return s.InsertWithPayload(ctx, id, vec, nil, nil)
}

// InsertDeferred adds a vector without building the HNSW graph (Bulk Load).
func (s *shard) InsertDeferred(ctx context.Context, id model.ID, vec []float32, md metadata.Document, payload []byte) (model.RowID, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.idx == nil {
		return 0, fmt.Errorf("memtable is closed")
	}

	// Insert into HNSW (Deferred Mode) - respects caller's context for cancellation
	rowID, err := s.idx.InsertDeferred(ctx, vec)
	if err != nil {
		return 0, err
	}

	// Common metadata/payload/ID logic
	return s.commitInsert(rowID, id, md, payload)
}

// InsertWithPayload adds a vector and metadata to the memtable.
func (s *shard) InsertWithPayload(ctx context.Context, id model.ID, vec []float32, md metadata.Document, payload []byte) (model.RowID, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.idx == nil {
		return 0, fmt.Errorf("memtable is closed")
	}

	// Insert into HNSW - respects caller's context for cancellation/timeout
	rowID, err := s.idx.Insert(ctx, vec)
	if err != nil {
		return 0, err
	}

	return s.commitInsert(rowID, id, md, payload)
}

// commitInsert completes an insert by storing metadata, payload, and columnar data.
// Must be called under s.mu lock.
func (s *shard) commitInsert(rowID model.RowID, id model.ID, md metadata.Document, payload []byte) (model.RowID, error) {
	// Intern metadata
	var interned metadata.InternedDocument
	if md != nil {
		interned = metadata.Intern(md)
	}

	// Check sequence alignment with HNSW
	idx := int(rowID)
	expected := s.ids.Count()

	if uint64(idx) > expected {
		// HNSW advanced ahead of us (likely due to failed inserts burning IDs).
		// We catch up by filling the gap with placeholders.
		gap := uint64(idx) - expected
		for i := range gap {
			// Mark the skipped ID as deleted in HNSW to ensure it's treated as a tombstone.
			// Use Background context to ensure cleanup happens even if the current request is tight on time.
			_ = s.idx.Delete(context.Background(), model.RowID(expected+i))

			s.ids.Append(0) // Placeholder ID
			s.metadata.Append(metadata.InternedDocument{})
			s.payloads.Append(nil)
		}
	} else if uint64(idx) < expected {
		// This implies HNSW reused an ID we already have committed. Critical failure.
		return 0, fmt.Errorf("memtable rowID out of sync: expected %d, got %d", expected, idx)
	}

	// Append to Paged Stores (O(1))
	s.ids.Append(id)
	s.metadata.Append(interned)
	s.payloads.Append(payload)

	// Sync columnar metadata
	// 1. Sync existing columns
	targetLen := idx + 1
	for key, col := range s.columns {
		val, hasVal := md[key]
		var v metadata.Value
		if hasVal {
			v = val
		}

		// Ensure column catches up to targetLen - 1 (backfill if needed)
		for col.Len() < targetLen-1 {
			col.Append(metadata.Value{}) // Append nulls
		}

		if col.Len() == targetLen {
			col.Set(idx, v)
		} else {
			col.Append(v)
		}
	}

	// 2. Handle new columns
	for key, val := range md {
		if _, exists := s.columns[key]; !exists {
			// Create new column with initial capacity hint
			col, err := createColumn(val.Kind, 1024)
			if err == nil {
				// Grow to current index (fill nulls)
				col.Grow(idx)
				// Append current value
				col.Append(val)
				s.columns[key] = col
			}
		}
	}

	return rowID, nil
}

// GetID returns the external ID for a given internal row ID.
// Safe for concurrent use without locking.
func (s *shard) GetID(rowID uint32) (model.ID, bool) {
	return s.ids.Get(rowID)
}

// Delete marks a RowID as deleted.
func (s *shard) Delete(rowID model.RowID) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.idx == nil {
		return
	}

	// HNSW handles tombstones
	// We assume internal/hnsw has a Delete method.
	// If not, we need to check.
	// Checking... internal/hnsw/hnsw.go has Delete(id).
	_ = s.idx.Delete(context.Background(), rowID)
}

// Segment Interface Implementation

func (s *shard) ID() model.SegmentID {
	return s.id
}

func (s *shard) RowCount() uint32 {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.idx == nil {
		return 0
	}
	return uint32(s.idx.VectorCount())
}

func (s *shard) Metric() distance.Metric {
	return s.metric
}

func (s *shard) Search(ctx context.Context, q []float32, k int, filter segment.Filter, opts model.SearchOptions, searcherCtx *searcher.Searcher) error {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.idx == nil {
		return fmt.Errorf("memtable is closed")
	}

	// Map segment.Filter to HNSW filter
	hnswFilter := filter

	// Handle user metadata filter
	if opts.Filter != nil {
		hnswFilter = newColumnarFilterWrapper(filter, opts.Filter, s.columns, uint32(s.ids.Count()))
	}

	hnswOpts := &idxhnsw.SearchOptions{
		EFSearch: opts.NProbes,
		Filter:   hnswFilter,
	}
	if hnswOpts.EFSearch == 0 {
		// Default EFSearch logic:
		// Ensure at least 200 for good recall, but scale with k.
		hnswOpts.EFSearch = max(k+100, 200)
	}

	var sr *searcher.Searcher
	if searcherCtx != nil {
		sr = searcherCtx
	} else {
		sr = searcher.Get()
		defer searcher.Put(sr)
		// If we created a local searcher, we need to init the heap
		sr.Heap.Reset(s.metric != distance.MetricL2)
	}

	if err := s.idx.KNNSearchWithContext(ctx, sr, q, k, hnswOpts); err != nil {
		return err
	}

	// Extract results from s.Candidates (HNSW results) and push to s.Heap (Global results)
	// s.Candidates is a MaxHeap (worst at top) of size EF.
	// We want to take the best K from it and push to s.Heap.
	// Actually, s.Candidates contains the best EF candidates found.
	// We should push ALL of them (or at least the best K) to s.Heap.
	// s.Heap will handle the global top-K logic.

	results := sr.Candidates
	// Pop everything from results
	for results.Len() > 0 {
		item, _ := results.PopItem()
		// Encode Local RowID to Global RowID
		// RowID = (ShardIdx << 28) | LocalRowID
		globalRowID := (uint32(s.shardIdx) << 28) | uint32(item.Node)

		if sr.Heap.Len() < k {
			sr.Heap.Push(searcher.InternalCandidate{
				SegmentID: uint32(s.id),
				RowID:     globalRowID,
				Score:     item.Distance,
				Approx:    true,
			})
		} else {
			top := sr.Heap.Candidates[0]
			if searcher.InternalCandidateBetter(searcher.InternalCandidate{
				SegmentID: uint32(s.id),
				RowID:     globalRowID,
				Score:     item.Distance,
				Approx:    true,
			}, top, sr.Heap.Descending()) {
				sr.Heap.ReplaceTop(searcher.InternalCandidate{
					SegmentID: uint32(s.id),
					RowID:     globalRowID,
					Score:     item.Distance,
					Approx:    true,
				})
			}
		}
	}

	return nil
}

func (s *shard) Rerank(ctx context.Context, q []float32, cands []model.Candidate, dst []model.Candidate) ([]model.Candidate, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.idx == nil {
		return nil, fmt.Errorf("memtable is closed")
	}

	distFunc, err := distance.Provider(s.metric)
	if err != nil {
		return nil, err
	}

	for _, c := range cands {
		if c.Loc.SegmentID != s.id {
			continue
		}

		rowID := uint32(c.Loc.RowID)
		vec, ok := s.vectors.GetVector(model.RowID(rowID))
		if !ok {
			continue
		}

		dist := distFunc(q, vec)

		c.Score = dist
		c.Approx = false
		dst = append(dst, c)
	}

	return dst, nil
}

func (s *shard) Fetch(ctx context.Context, rows []uint32, cols []string) (segment.RecordBatch, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.idx == nil {
		return nil, fmt.Errorf("memtable is closed")
	}

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

	batch := &segment.SimpleRecordBatch{
		IDs: make([]model.ID, len(rows)),
	}
	if fetchVectors {
		batch.Vectors = make([][]float32, len(rows))
	}
	if fetchMetadata {
		batch.Metadatas = make([]metadata.Document, len(rows))
	}
	if fetchPayload {
		batch.Payloads = make([][]byte, len(rows))
	}

	for i, rowID := range rows {
		if id, ok := s.ids.Get(rowID); ok {
			batch.IDs[i] = id
		} else {
			return nil, fmt.Errorf("rowID %d out of bounds (count=%d, shard=%d)", rowID, uint32(s.ids.Count()), s.shardIdx)
		}

		if fetchVectors {
			vec, ok := s.vectors.GetVector(model.RowID(rowID))
			if !ok {
				return nil, fmt.Errorf("vector not found for rowID %d", rowID)
			}

			// Vectors are immutable in LSM (Update = Delete + Insert), so we can return
			// the slice directly without copying. The underlying storage (ColumnarStore)
			// uses copy-on-write for growth, so old slices remain valid.
			batch.Vectors[i] = vec
		}

		if fetchMetadata {
			if interned, ok := s.metadata.Get(rowID); ok && interned != nil {
				batch.Metadatas[i] = metadata.Unintern(interned)
			}
		}

		if fetchPayload {
			if pl, ok := s.payloads.Get(rowID); ok {
				batch.Payloads[i] = pl
			}
		}
	}

	return batch, nil
}

func (s *shard) FetchIDs(ctx context.Context, rows []uint32, dst []model.ID) error {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.idx == nil {
		return fmt.Errorf("memtable is closed")
	}

	if len(dst) != len(rows) {
		return fmt.Errorf("dst length mismatch")
	}

	for i, rowID := range rows {
		// If rowID is out of bounds (race between HNSW index and s.ids update/fill),
		// we treat it as an invalid/deleted ID (0).
		if id, ok := s.ids.Get(rowID); ok {
			dst[i] = id
		} else {
			// This happens if HNSW allocated an ID but the transaction failed (timeout)
			// and we haven't "filled the gap" yet in s.ids.
			// Return 0 (invalid ID) instead of failing the whole batch.
			dst[i] = 0
		}
	}
	return nil
}

// Iterate iterates over all valid (non-deleted) vectors in the memtable.
// The context is used for cancellation during long iterations.
func (s *shard) Iterate(ctx context.Context, fn func(rowID uint32, id model.ID, vec []float32, md metadata.Document, payload []byte) error) error {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.idx == nil {
		return fmt.Errorf("memtable is closed")
	}

	total := uint32(s.ids.Count())

	// Iterate 0..Count
	for i := range total {
		// Periodic context check (every 256 rows)
		if i&255 == 0 {
			select {
			case <-ctx.Done():
				return ctx.Err()
			default:
			}
		}

		rowID := i

		// Check if deleted in HNSW
		if !s.idx.ContainsID(uint64(rowID)) {
			continue
		}

		id, ok := s.ids.Get(rowID)
		if !ok {
			// Should not happen if count is correct
			continue
		}

		vec, ok := s.vectors.GetVector(model.RowID(rowID))
		if !ok {
			continue
		}

		var md metadata.Document
		if interned, ok := s.metadata.Get(rowID); ok && interned != nil {
			md = make(metadata.Document, len(interned))
			for k, v := range interned {
				md[k.Value()] = v
			}
		}

		var payload []byte
		if pl, ok := s.payloads.Get(rowID); ok {
			payload = pl
		}

		if err := fn(rowID, id, vec, md, payload); err != nil {
			return err
		}
	}
	return nil
}

// Size returns the estimated memory usage in bytes.
func (s *shard) Size() int64 {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.idx == nil {
		return 0
	}

	size := int64(0)
	// HNSW size
	size += s.idx.Size()

	// Vectors size
	if v, ok := s.vectors.(interface{ Size() int64 }); ok {
		size += v.Size()
	}

	// IDs size (approx)
	size += int64(uint32(s.ids.Count())) * 8
	// Metadata pointers size? (approx)
	size += int64(s.metadata.Count()) * 8 // Pointers
	// Payloads pointers (slice headers)
	size += int64(s.payloads.Count()) * 24

	return size
}

func (s *shard) EvaluateFilter(ctx context.Context, filter *metadata.FilterSet) (*imetadata.LocalBitmap, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if filter == nil || len(filter.Filters) == 0 {
		return nil, nil // All
	}

	// Check context before potentially expensive filter evaluation
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	count := int(s.ids.Count())
	result := imetadata.NewLocalBitmap()

	first := true

	for _, f := range filter.Filters {
		col, ok := s.columns[f.Key]
		if !ok {
			// Column doesn't exist.
			return imetadata.NewLocalBitmap(), nil
		}

		// Type-specific optimizations to avoid per-row overhead (e.g. handle resolution)
		var matchFunc func(i int) bool

		// Optimization for Int (Common)
		if ic, ok := col.(*intColumn); ok && f.Value.Kind == metadata.KindInt && f.Operator == metadata.OpEqual {
			val, _ := f.Value.AsInt64()
			matchFunc = func(i int) bool {
				return i < len(ic.data) && ic.valid[i] && ic.data[i] == val
			}
		}

		if first {
			for i := range count {
				if matchFunc != nil {
					if matchFunc(i) {
						result.Add(uint32(i))
					}
				} else if col.Matches(i, f.Value, f.Operator) {
					result.Add(uint32(i))
				}
			}
			first = false
		} else {
			// Intersect with existing result
			newResult := imetadata.NewLocalBitmap()

			result.ForEach(func(id uint32) bool {
				idx := int(id)
				if matchFunc != nil {
					if matchFunc(idx) {
						newResult.Add(id)
					}
				} else if col.Matches(idx, f.Value, f.Operator) {
					newResult.Add(id)
				}
				return true
			})
			result = newResult
		}

		if result.IsEmpty() {
			return result, nil
		}
	}

	return result, nil
}

// Advise is a no-op for MemTable.
func (s *shard) Advise(pattern segment.AccessPattern) error {
	return nil
}
