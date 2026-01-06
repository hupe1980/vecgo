package memtable

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"

	"github.com/hupe1980/vecgo/distance"
	"github.com/hupe1980/vecgo/internal/arena"
	idxhnsw "github.com/hupe1980/vecgo/internal/hnsw"
	"github.com/hupe1980/vecgo/internal/searcher"
	"github.com/hupe1980/vecgo/internal/segment"
	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/model"
	"github.com/hupe1980/vecgo/vectorstore"
)

// MemTable is a mutable in-memory segment backed by HNSW.
type MemTable struct {
	mu       sync.RWMutex
	refs     int64
	id       model.SegmentID
	dim      int
	metric   distance.Metric
	idx      *idxhnsw.HNSW
	vectors  vectorstore.VectorStore
	pks      []model.PK
	metadata []metadata.InternedDocument
	payloads [][]byte
}

// New creates a new MemTable.
func New(id model.SegmentID, dim int, metric distance.Metric, acquirer arena.MemoryAcquirer) (*MemTable, error) {
	store, err := vectorstore.New(dim)
	if err != nil {
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
	opts.InitialArenaSize = 4 * 1024 * 1024 // 4MB initial chunk size

	h, err := idxhnsw.New(func(o *idxhnsw.Options) {
		*o = opts
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create memtable hnsw: %w", err)
	}

	return &MemTable{
		refs:     1,
		id:       id,
		dim:      dim,
		metric:   metric,
		idx:      h,
		vectors:  store,
		pks:      make([]model.PK, 0, 1024),
		metadata: make([]metadata.InternedDocument, 0, 1024),
	}, nil
}

// IncRef increments the reference count.
func (m *MemTable) IncRef() {
	atomic.AddInt64(&m.refs, 1)
}

// DecRef decrements the reference count and closes the memtable if it reaches zero.
func (m *MemTable) DecRef() {
	if atomic.AddInt64(&m.refs, -1) == 0 {
		m.closeInternal()
	}
}

// Insert adds a vector to the memtable.
// Returns the assigned RowID.
func (m *MemTable) Insert(pk model.PK, vec []float32) (model.RowID, error) {
	return m.InsertWithPayload(pk, vec, nil, nil)
}

// InsertWithPayload adds a vector and metadata to the memtable.
func (m *MemTable) InsertWithPayload(pk model.PK, vec []float32, md metadata.Document, payload []byte) (model.RowID, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.idx == nil {
		return 0, fmt.Errorf("memtable is closed")
	}

	// Insert into HNSW (this also adds to the vector store via the interface)
	id, err := m.idx.Insert(context.Background(), vec)
	if err != nil {
		return 0, err
	}

	// Intern metadata
	var interned metadata.InternedDocument
	if md != nil {
		interned = metadata.Intern(md)
	}

	// Track PK, Metadata, and Payload
	if int(id) == len(m.pks) {
		m.pks = append(m.pks, pk)
		m.metadata = append(m.metadata, interned)
		m.payloads = append(m.payloads, payload)
	} else if int(id) < len(m.pks) {
		m.pks[id] = pk
		m.metadata[id] = interned
		m.payloads[id] = payload
	} else {
		// Resize if needed
		if int(id) >= cap(m.pks) {
			newPks := make([]model.PK, int(id)+1, int(id)*2)
			copy(newPks, m.pks)
			m.pks = newPks

			newMd := make([]metadata.InternedDocument, int(id)+1, int(id)*2)
			copy(newMd, m.metadata)
			m.metadata = newMd

			newPayloads := make([][]byte, int(id)+1, int(id)*2)
			copy(newPayloads, m.payloads)
			m.payloads = newPayloads
		} else {
			m.pks = m.pks[:int(id)+1]
			m.metadata = m.metadata[:int(id)+1]
			m.payloads = m.payloads[:int(id)+1]
		}
		m.pks[id] = pk
		m.metadata[id] = interned
		m.payloads[id] = payload
	}

	return model.RowID(id), nil
}

// Delete marks a RowID as deleted.
func (m *MemTable) Delete(rowID model.RowID) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.idx == nil {
		return
	}

	// HNSW handles tombstones
	// We assume internal/hnsw has a Delete method.
	// If not, we need to check.
	// Checking... internal/hnsw/hnsw.go has Delete(id).
	_ = m.idx.Delete(context.Background(), rowID)
}

// Segment Interface Implementation

func (m *MemTable) ID() model.SegmentID {
	return m.id
}

func (m *MemTable) RowCount() uint32 {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.idx == nil {
		return 0
	}
	return uint32(m.idx.VectorCount())
}

func (m *MemTable) Metric() distance.Metric {
	return m.metric
}

func (m *MemTable) Search(ctx context.Context, q []float32, k int, filter segment.Filter, opts model.SearchOptions, searcherCtx *searcher.Searcher) error {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if m.idx == nil {
		return fmt.Errorf("memtable is closed")
	}

	// Map segment.Filter to HNSW filter
	var hnswFilter func(id model.RowID) bool
	if filter != nil {
		hnswFilter = func(id model.RowID) bool {
			return filter.Matches(uint32(id))
		}
	}

	// Handle user metadata filter
	if opts.Filter != nil {
		fs := opts.Filter
		prevFilter := hnswFilter
		hnswFilter = func(id model.RowID) bool {
			if prevFilter != nil && !prevFilter(id) {
				return false
			}
			if int(id) >= len(m.metadata) {
				return false
			}
			doc := m.metadata[int(id)]
			if doc == nil {
				return false
			}
			return fs.MatchesInterned(doc)
		}
	}

	hnswOpts := &idxhnsw.SearchOptions{
		EFSearch: opts.NProbes,
		Filter:   hnswFilter,
	}
	if hnswOpts.EFSearch == 0 {
		// Default EFSearch logic:
		// Ensure at least 200 for good recall, but scale with k.
		hnswOpts.EFSearch = k + 100
		if hnswOpts.EFSearch < 200 {
			hnswOpts.EFSearch = 200
		}
	}

	var s *searcher.Searcher
	if searcherCtx != nil {
		s = searcherCtx
	} else {
		s = searcher.Get()
		defer searcher.Put(s)
		// If we created a local searcher, we need to init the heap
		s.Heap.Reset(m.metric != distance.MetricL2)
	}

	if err := m.idx.KNNSearchWithContext(ctx, s, q, k, hnswOpts); err != nil {
		return err
	}

	// Extract results from s.Candidates (HNSW results) and push to s.Heap (Global results)
	// s.Candidates is a MaxHeap (worst at top) of size EF.
	// We want to take the best K from it and push to s.Heap.
	// Actually, s.Candidates contains the best EF candidates found.
	// We should push ALL of them (or at least the best K) to s.Heap.
	// s.Heap will handle the global top-K logic.

	results := s.Candidates
	// Pop everything from results
	for results.Len() > 0 {
		item, _ := results.PopItem()
		cand := model.Candidate{
			Loc: model.Location{
				SegmentID: m.id,
				RowID:     model.RowID(item.Node),
			},
			Score:  item.Distance,
			Approx: true,
		}

		if s.Heap.Len() < k {
			s.Heap.Push(cand)
		} else {
			top := s.Heap.Candidates[0]
			if searcher.CandidateBetter(cand, top, s.Heap.Descending()) {
				s.Heap.ReplaceTop(cand)
			}
		}
	}

	return nil
}

func (m *MemTable) Rerank(ctx context.Context, q []float32, cands []model.Candidate, dst []model.Candidate) ([]model.Candidate, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if m.idx == nil {
		return nil, fmt.Errorf("memtable is closed")
	}

	distFunc, err := distance.Provider(m.metric)
	if err != nil {
		return nil, err
	}

	for _, c := range cands {
		if c.Loc.SegmentID != m.id {
			continue
		}

		rowID := uint32(c.Loc.RowID)
		vec, ok := m.vectors.GetVector(model.RowID(rowID))
		if !ok {
			continue
		}

		// Check if deleted in HNSW?
		// HNSW Delete marks tombstone but vector remains in store?
		// We should check if HNSW considers it deleted.
		// idx.ContainsID checks existence, but maybe not tombstone status?
		// Actually, if it was returned by Search, it's not deleted.
		// But Rerank might be called on candidates from other sources (if we had them).
		// For MemTable, candidates come from Search.

		dist := distFunc(q, vec)

		c.Score = dist
		c.Approx = false
		dst = append(dst, c)
	}

	return dst, nil
}

func (m *MemTable) Fetch(ctx context.Context, rows []uint32, cols []string) (segment.RecordBatch, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if m.idx == nil {
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
		PKs: make([]model.PK, len(rows)),
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
		if int(rowID) >= len(m.pks) {
			return nil, fmt.Errorf("rowID %d out of bounds", rowID)
		}

		batch.PKs[i] = m.pks[rowID]

		if fetchVectors {
			vec, ok := m.vectors.GetVector(model.RowID(rowID))
			if !ok {
				return nil, fmt.Errorf("vector not found for rowID %d", rowID)
			}

			// Copy vector to avoid aliasing issues if store reuses memory (columnar usually doesn't but safe is better)
			v := make([]float32, len(vec))
			copy(v, vec)
			batch.Vectors[i] = v
		}

		if fetchMetadata {
			batch.Metadatas[i] = metadata.Unintern(m.metadata[rowID])
		}

		if fetchPayload {
			batch.Payloads[i] = m.payloads[rowID]
		}
	}

	return batch, nil
}

func (m *MemTable) FetchPKs(ctx context.Context, rows []uint32, dst []model.PK) error {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if m.idx == nil {
		return fmt.Errorf("memtable is closed")
	}

	if len(dst) != len(rows) {
		return fmt.Errorf("dst length mismatch")
	}

	for i, rowID := range rows {
		if int(rowID) >= len(m.pks) {
			return fmt.Errorf("rowID out of bounds")
		}
		dst[i] = m.pks[rowID]
	}
	return nil
}

// Iterate iterates over all valid (non-deleted) vectors in the memtable.
func (m *MemTable) Iterate(fn func(rowID uint32, pk model.PK, vec []float32, md metadata.Document, payload []byte) error) error {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if m.idx == nil {
		return fmt.Errorf("memtable is closed")
	}

	// Iterate 0..NextID
	// We can use pks length as upper bound
	for i := 0; i < len(m.pks); i++ {
		rowID := uint32(i)
		// Check if deleted in HNSW
		// idx.ContainsID might be useful
		if !m.idx.ContainsID(uint64(rowID)) {
			continue
		}

		vec, ok := m.vectors.GetVector(model.RowID(rowID))
		if !ok {
			continue
		}

		var md metadata.Document
		if i < len(m.metadata) {
			interned := m.metadata[i]
			if interned != nil {
				md = make(metadata.Document, len(interned))
				for k, v := range interned {
					md[k.Value()] = v
				}
			}
		}

		var payload []byte
		if i < len(m.payloads) {
			payload = m.payloads[i]
		}

		if err := fn(rowID, m.pks[i], vec, md, payload); err != nil {
			return err
		}
	}
	return nil
}

// Size returns the estimated memory usage in bytes.
func (m *MemTable) Size() int64 {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if m.idx == nil {
		return 0
	}

	size := int64(0)
	// HNSW size
	size += m.idx.Size()

	// Vectors size
	if s, ok := m.vectors.(interface{ Size() int64 }); ok {
		size += s.Size()
	}

	// PKs size
	// model.PK is struct { Kind uint8; U64 uint64; S string }
	// Size is roughly 1 + 8 + 16 = 25 bytes + padding -> 32 bytes.
	// Plus string content if any.
	size += int64(cap(m.pks)) * 32

	return size
}

func (m *MemTable) Close() error {
	m.DecRef()
	return nil
}

func (m *MemTable) closeInternal() {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.idx != nil {
		_ = m.idx.Close()
	}

	m.idx = nil
	m.vectors = nil
	m.pks = nil
}

// Advise is a no-op for MemTable.
func (m *MemTable) Advise(pattern segment.AccessPattern) error {
	return nil
}
