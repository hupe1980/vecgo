package memtable

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"

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

// Pool for rerank shard dispatch slices to reduce allocations in hot path
var rerankByShardPool = sync.Pool{
	New: func() any {
		// Pre-allocate slice of slices
		s := make([][]model.Candidate, shardCount)
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

	// We need to hold partial results until merge.
	// But `segment.RecordBatch` is interface.
	// `memtable` uses `SimpleRecordBatch`.
	// We can construct a combined batch.

	resBatch := &segment.SimpleRecordBatch{
		IDs: make([]model.ID, total),
	}
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
	if fetchVectors {
		resBatch.Vectors = make([][]float32, total)
	}
	if fetchMetadata {
		resBatch.Metadatas = make([]metadata.Document, total)
	}
	if fetchPayload {
		resBatch.Payloads = make([][]byte, total)
	}

	type req struct {
		origIdx int
		rowID   uint32
	}
	shardReqs := make([][]req, shardCount)
	for i, r := range rows {
		sIdx := r >> 28
		if int(sIdx) >= shardCount {
			return nil, fmt.Errorf("invalid rowID %d", r)
		}
		shardReqs[sIdx] = append(shardReqs[sIdx], req{i, r & uint32(rowIdMask)})
	}

	for sIdx, reqs := range shardReqs {
		if len(reqs) == 0 {
			continue
		}

		rowIDs := make([]uint32, len(reqs))
		for k, r := range reqs {
			rowIDs[k] = r.rowID
		}

		subBatch, err := m.shards[sIdx].Fetch(ctx, rowIDs, cols)
		if err != nil {
			return nil, err
		}

		// Copy back
		simple, ok := subBatch.(*segment.SimpleRecordBatch)
		if !ok {
			return nil, fmt.Errorf("unexpected batch type")
		}

		for k, r := range reqs {
			resBatch.IDs[r.origIdx] = simple.IDs[k]
			if fetchVectors {
				resBatch.Vectors[r.origIdx] = simple.Vectors[k]
			}
			if fetchMetadata {
				resBatch.Metadatas[r.origIdx] = simple.Metadatas[k]
			}
			if fetchPayload {
				resBatch.Payloads[r.origIdx] = simple.Payloads[k]
			}
		}
	}

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

// EvaluateFilter returns a bitmap of rows matching the filter.
func (m *MemTable) EvaluateFilter(ctx context.Context, filter *metadata.FilterSet) (segment.Bitmap, error) {
	if filter == nil || len(filter.Filters) == 0 {
		return nil, nil // All matches
	}

	result := imetadata.NewLocalBitmap()

	for i, s := range m.shards {
		shardBitmap, err := s.EvaluateFilter(ctx, filter)
		if err != nil {
			return nil, err
		}

		if shardBitmap != nil {
			// Shard-local offsets need to be shifted
			shardOffset := uint32(i) << 28

			shardBitmap.ForEach(func(id uint32) bool {
				result.Add(id | shardOffset)
				return true
			})
		}
	}
	return result, nil
}

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
