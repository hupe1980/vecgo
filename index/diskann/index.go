package diskann

import (
	"container/heap"
	"context"
	"fmt"
	"iter"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/hupe1980/vecgo/index"
	"github.com/hupe1980/vecgo/index/hnsw"
	"github.com/hupe1980/vecgo/internal/bitset"
	"github.com/hupe1980/vecgo/searcher"
)

// Compile-time interface checks
var (
	_ index.Index              = (*Index)(nil)
	_ index.TransactionalIndex = (*Index)(nil)
)

// Index is a disk-resident approximate nearest neighbor index with incremental update support.
//
// It implements both index.Index and index.TransactionalIndex for full Vecgo integration.
// Supports:
//   - Incremental inserts (new vectors added to Vamana graph)
//   - Soft deletes (deletion bitmap, skip during search)
//   - Updates (delete + insert)
//   - Background compaction (removes deleted vectors, rebuilds graph, re-trains PQ)
//   - Both mutable mode (New) and read-only mode (Open for Builder-created indexes)
//
// Index is a disk-resident approximate nearest neighbor index using an LSM-tree architecture.
//
// Architecture:
//   - MemTable: In-memory HNSW index for recent inserts.
//   - Segments: Immutable, disk-resident DiskANN indexes (Vamana graph).
//   - Global ID Space: IDs are monotonic and partitioned across segments.
//
// Operations:
//   - Insert: Adds to MemTable.
//   - Search: Queries MemTable and all Segments, merges results.
//   - Flush: Converts MemTable to a new immutable Segment.
//   - Compact: Merges multiple Segments into a larger one (background).
type Index struct {
	dim       int
	distType  index.DistanceType
	distFunc  index.DistanceFunc
	opts      *Options
	indexPath string

	// LSM Components
	mu              sync.RWMutex
	segments        []*Segment
	memTable        *hnsw.HNSW
	memTableStartID uint64         // Global ID of the first vector in MemTable
	memTableCount   atomic.Uint64  // Number of vectors in MemTable
	memTablePresent *bitset.BitSet // IDs currently in MemTable (to mask stale segment data)

	// Logical Count
	count atomic.Uint64

	// ID Management
	nextIDAtomic atomic.Uint64
	deleted      *bitset.BitSet

	// Compaction
	compacting      atomic.Bool
	stopCompaction  chan struct{}
	compactionWg    sync.WaitGroup
	compactionStats CompactionStats
	compactionMu    sync.RWMutex

	// Pooling
	scratchPool sync.Pool
}

// searchScratch holds temporary buffers for search to avoid allocations.
type searchScratch struct {
	results        []index.SearchResult   // Final merged results
	segmentResults [][]index.SearchResult // Per-segment results (slice of slices)
	flatResults    []index.SearchResult   // Flat buffer for all segment results
	seen           map[uint64]struct{}    // Deduplication map (fallback)
	seenBits       *bitset.BitSet         // Deduplication bitset (primary)

	// Segment Search Buffers
	candidates candidateHeap  // Min-heap for beam search
	visited    *bitset.BitSet // Visited set for beam search
	beamBuf    []distNode     // Buffer for beam search results
}

// CompactionStats tracks compaction statistics.
type CompactionStats struct {
	LastCompactionTime     int64  // Unix timestamp
	TotalCompactions       uint64 // Total number of compactions performed
	VectorsRemovedTotal    uint64 // Total vectors removed across all compactions
	LastVectorsRemoved     uint32 // Vectors removed in last compaction
	LastCompactionDuration int64  // Duration in milliseconds
}

// New creates a new DiskANN index.
func New(dim int, distType index.DistanceType, indexPath string, opts *Options) (*Index, error) {
	if opts == nil {
		opts = DefaultOptions()
	}

	if dim <= 0 {
		return nil, &index.ErrInvalidDimension{Dimension: dim}
	}

	// Ensure directory exists
	if err := os.MkdirAll(indexPath, 0755); err != nil {
		return nil, fmt.Errorf("diskann: create directory: %w", err)
	}

	idx := &Index{
		dim:             dim,
		distType:        distType,
		distFunc:        index.NewDistanceFunc(distType),
		opts:            opts,
		indexPath:       indexPath,
		segments:        make([]*Segment, 0),
		deleted:         bitset.New(1024),
		memTablePresent: bitset.New(1024),
		stopCompaction:  make(chan struct{}),
	}

	idx.initPools()

	// Initialize MemTable
	if err := idx.resetMemTable(); err != nil {
		return nil, err
	}

	// Start background compaction
	if opts.EnableAutoCompaction {
		idx.compactionWg.Add(1)
		go idx.backgroundCompaction()
	}

	return idx, nil
}

// Open opens an existing DiskANN index.
func Open(indexPath string, opts *Options) (*Index, error) {
	if opts == nil {
		opts = DefaultOptions()
	}

	idx := &Index{
		opts:            opts,
		indexPath:       indexPath,
		segments:        make([]*Segment, 0),
		deleted:         bitset.New(1024),
		memTablePresent: bitset.New(1024),
		stopCompaction:  make(chan struct{}),
	}

	idx.initPools()

	// Scan for segments
	entries, err := os.ReadDir(indexPath)
	if err != nil {
		return nil, fmt.Errorf("diskann: read dir: %w", err)
	}

	var segmentPaths []string
	for _, entry := range entries {
		if entry.IsDir() && strings.HasPrefix(entry.Name(), "segment-") {
			segmentPaths = append(segmentPaths, filepath.Join(indexPath, entry.Name()))
		}
	}
	sort.Strings(segmentPaths) // Ensure deterministic order

	// Load segments
	maxID := uint64(0)
	for _, path := range segmentPaths {
		var baseID uint64
		var dummy string
		n, err := fmt.Sscanf(filepath.Base(path), "segment-%d%s", &baseID, &dummy)
		if n < 1 {
			continue
		}

		seg, err := OpenSegment(path, baseID, opts)
		if err != nil {
			return nil, fmt.Errorf("diskann: open segment %s: %w", path, err)
		}
		idx.segments = append(idx.segments, seg)
		idx.count.Add(seg.count)

		segMax := baseID + seg.count
		if segMax > maxID {
			maxID = segMax
		}

		if idx.dim == 0 {
			idx.dim = seg.dim
			idx.distType = seg.distType
			idx.distFunc = seg.distFunc
		}
	}

	// Check for legacy single-file index (migration)
	legacyMeta := filepath.Join(indexPath, MetaFilename)
	if _, err := os.Stat(legacyMeta); err == nil {
		seg, err := OpenSegment(indexPath, 0, opts)
		if err != nil {
			return nil, fmt.Errorf("diskann: open legacy segment: %w", err)
		}
		idx.segments = append(idx.segments, seg)
		idx.count.Add(seg.count)
		if seg.count > maxID {
			maxID = seg.count
		}
		if idx.dim == 0 {
			idx.dim = seg.dim
			idx.distType = seg.distType
			idx.distFunc = seg.distFunc
		}
	}

	idx.nextIDAtomic.Store(maxID)
	idx.memTableStartID = maxID

	// Initialize MemTable
	if err := idx.resetMemTable(); err != nil {
		return nil, err
	}

	// Start background compaction
	if opts.EnableAutoCompaction {
		idx.compactionWg.Add(1)
		go idx.backgroundCompaction()
	}

	return idx, nil
}

// initPools initializes the sync.Pools.
func (idx *Index) initPools() {
	idx.scratchPool.New = func() interface{} {
		return &searchScratch{
			results:        make([]index.SearchResult, 0, 1024),
			segmentResults: make([][]index.SearchResult, 0, 16),
			flatResults:    make([]index.SearchResult, 0, 4096),
			seen:           make(map[uint64]struct{}, 1024),
			seenBits:       bitset.New(1024),
			candidates:     make(candidateHeap, 0, 128),
			visited:        bitset.New(1024),
			beamBuf:        make([]distNode, 0, 256),
		}
	}
}

// Close releases resources held by the index.
func (idx *Index) Close() error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	// Stop background compaction
	if idx.stopCompaction != nil {
		close(idx.stopCompaction)
		idx.compactionWg.Wait()
		idx.stopCompaction = nil
	}

	// Close segments
	for _, seg := range idx.segments {
		if err := seg.Close(); err != nil {
			return err
		}
	}

	// Close MemTable
	if idx.memTable != nil {
		return idx.memTable.Close()
	}

	return nil
}

func (idx *Index) resetMemTable() error {
	memTable, err := hnsw.New(func(o *hnsw.Options) {
		o.Dimension = idx.dim
		o.M = 32 // Fixed for MemTable
		o.EF = 100
		o.Heuristic = true
		o.DistanceType = idx.distType
		// MemTable manages its own vectors in memory
	})
	if err != nil {
		return fmt.Errorf("diskann: create memtable: %w", err)
	}
	idx.memTable = memTable
	idx.memTableCount.Store(0)
	if idx.memTablePresent != nil {
		idx.memTablePresent.ClearAll()
	}
	return nil
}

// ============================================================================
// TransactionalIndex Implementation
// ============================================================================

// AllocateID reserves a new ID for insertion.
func (idx *Index) AllocateID() uint64 {
	return idx.nextIDAtomic.Add(1) - 1
}

// ReleaseID is a no-op in this LSM implementation.
func (idx *Index) ReleaseID(id uint64) {}

// ============================================================================
// Index Mutations (Insert/Delete/Update)
// ============================================================================

// Insert adds a vector to the index.
func (idx *Index) Insert(ctx context.Context, v []float32) (uint64, error) {
	id := idx.AllocateID()
	if err := idx.ApplyInsert(ctx, id, v); err != nil {
		return 0, err
	}
	idx.count.Add(1)
	return id, nil
}

// ApplyInsert adds a vector with a specific ID to the MemTable.
func (idx *Index) ApplyInsert(ctx context.Context, id uint64, v []float32) error {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	memTable := idx.memTable
	if memTable == nil {
		return fmt.Errorf("diskann: memtable not initialized")
	}

	// Insert into MemTable
	if err := memTable.ApplyInsert(ctx, id, v); err != nil {
		return fmt.Errorf("diskann: memtable insert: %w", err)
	}

	// Update shadowing bitset and stats
	idx.memTablePresent.Set(id)
	idx.memTableCount.Add(1)

	// Clear deletion bit if it was previously deleted
	idx.deleted.Unset(id)

	return nil
}

// ApplyBatchInsert adds multiple vectors with specific IDs to the MemTable.
func (idx *Index) ApplyBatchInsert(ctx context.Context, ids []uint64, vectors [][]float32) error {
	if len(ids) != len(vectors) {
		return fmt.Errorf("ids and vectors length mismatch")
	}

	idx.mu.RLock()
	memTable := idx.memTable
	idx.mu.RUnlock()

	if memTable == nil {
		return fmt.Errorf("diskann: memtable not initialized")
	}

	// Insert into MemTable
	if err := memTable.ApplyBatchInsert(ctx, ids, vectors); err != nil {
		return fmt.Errorf("diskann: memtable batch insert: %w", err)
	}

	// Update shadowing bitset and stats
	for _, id := range ids {
		idx.memTablePresent.Set(id)
		idx.deleted.Unset(id)
	}
	idx.memTableCount.Add(uint64(len(ids)))

	return nil
}

// Delete removes a vector from the index.
func (idx *Index) Delete(ctx context.Context, id uint64) error {
	return idx.ApplyDelete(ctx, id)
}

// ApplyDelete marks a vector as deleted.
func (idx *Index) ApplyDelete(ctx context.Context, id uint64) error {
	idx.deleted.Set(id)

	// Also delete from MemTable if present to keep it clean
	idx.mu.RLock()
	memTable := idx.memTable
	idx.mu.RUnlock()

	if memTable != nil {
		_ = memTable.ApplyDelete(ctx, id)
	}

	return nil
}

// Update updates a vector in the index.
func (idx *Index) Update(ctx context.Context, id uint64, v []float32) error {
	return idx.ApplyUpdate(ctx, id, v)
}

// ApplyUpdate updates a vector with a specific ID.
func (idx *Index) ApplyUpdate(ctx context.Context, id uint64, v []float32) error {
	// In LSM, update is Insert with same ID.
	return idx.ApplyInsert(ctx, id, v)
}

// BatchInsert adds multiple vectors.
func (idx *Index) BatchInsert(ctx context.Context, vectors [][]float32) index.BatchInsertResult {
	result := index.BatchInsertResult{
		IDs:    make([]uint64, len(vectors)),
		Errors: make([]error, len(vectors)),
	}

	for i, v := range vectors {
		id, err := idx.Insert(ctx, v)
		result.IDs[i] = id
		result.Errors[i] = err
	}

	return result
}

// VectorByID retrieves a vector by its ID.
func (idx *Index) VectorByID(ctx context.Context, id uint64) ([]float32, error) {
	idx.mu.RLock()
	deleted := idx.deleted
	memTable := idx.memTable
	segments := idx.segments
	idx.mu.RUnlock()

	if deleted.Test(id) {
		return nil, &index.ErrNodeDeleted{ID: id}
	}

	if memTable != nil {
		vec, err := memTable.VectorByID(ctx, id)
		if err == nil {
			return vec, nil
		}
	}

	// Check Segments (newest to oldest)
	for i := len(segments) - 1; i >= 0; i-- {
		vec, err := segments[i].VectorByID(ctx, id)
		if err == nil {
			return vec, nil
		}
	}

	return nil, &index.ErrNodeNotFound{ID: id}
}

// KNNSearch performs k-nearest neighbor search with optional pre-filtering.
// filter (in opts): Applied DURING graph traversal for correct recall and performance.
func (idx *Index) KNNSearch(ctx context.Context, query []float32, k int, opts *index.SearchOptions) ([]index.SearchResult, error) {
	var results []index.SearchResult
	if err := idx.KNNSearchWithBuffer(ctx, query, k, opts, &results); err != nil {
		return nil, err
	}
	return results, nil
}

func (idx *Index) KNNSearchWithBuffer(ctx context.Context, query []float32, k int, opts *index.SearchOptions, buf *[]index.SearchResult) error {
	if len(query) != idx.dim {
		return &index.ErrDimensionMismatch{Expected: idx.dim, Actual: len(query)}
	}
	if k <= 0 {
		return index.ErrInvalidK
	}

	idx.mu.RLock()
	memTable := idx.memTable
	segments := idx.segments
	deleted := idx.deleted
	memTablePresent := idx.memTablePresent
	idx.mu.RUnlock()

	// Get scratch buffer from pool
	scratch := idx.scratchPool.Get().(*searchScratch)
	defer func() {
		// Clear and return to pool
		scratch.results = scratch.results[:0]
		scratch.segmentResults = scratch.segmentResults[:0]
		scratch.flatResults = scratch.flatResults[:0]
		for k := range scratch.seen {
			delete(scratch.seen, k)
		}
		if scratch.seenBits != nil {
			scratch.seenBits.ClearAll()
		}
		scratch.candidates = scratch.candidates[:0]
		scratch.beamBuf = scratch.beamBuf[:0]
		if scratch.visited != nil {
			scratch.visited.ClearAll()
		}
		idx.scratchPool.Put(scratch)
	}()

	// Wrap filter to exclude deleted items
	var userFilter func(uint64) bool
	if opts != nil {
		userFilter = opts.Filter
	}

	// Base filter: deleted check
	baseFilter := func(id uint64) bool {
		if deleted.Test(id) {
			return false
		}
		if userFilter != nil {
			return userFilter(id)
		}
		return true
	}

	// Segment filter: base filter + check if shadowed by MemTable
	segmentFilter := func(id uint64) bool {
		if !baseFilter(id) {
			return false
		}
		// Check shadowing
		return !memTablePresent.Test(id)
	}

	// Search MemTable
	if memTable != nil {
		memOpts := index.SearchOptions{
			Filter: baseFilter,
		}
		if opts != nil {
			memOpts.EFSearch = opts.EFSearch
		}

		// Use Zero-Alloc KNNSearchWithBuffer
		if err := memTable.KNNSearchWithBuffer(ctx, query, k, &memOpts, &scratch.results); err != nil {
			return fmt.Errorf("diskann: memtable search: %w", err)
		}
	}

	// Search DiskANN Segments
	if len(segments) > 0 {
		// Pre-allocate segment results slice in scratch
		if cap(scratch.segmentResults) < len(segments) {
			scratch.segmentResults = make([][]index.SearchResult, 0, len(segments))
		}

		// Calculate capacity per segment
		capPerSeg := k * 2
		if opts != nil && opts.EFSearch > capPerSeg {
			capPerSeg = opts.EFSearch
		}

		// Ensure flat buffer has enough capacity
		totalCap := capPerSeg * len(segments)
		if cap(scratch.flatResults) < totalCap {
			scratch.flatResults = make([]index.SearchResult, 0, totalCap)
		}

		// Search segments sequentially (Zero-Alloc)
		for _, seg := range segments {
			// Ensure capacity for this segment in flat buffer
			if len(scratch.flatResults)+capPerSeg > cap(scratch.flatResults) {
				// Grow flat buffer
				newCap := cap(scratch.flatResults) * 2
				if newCap < len(scratch.flatResults)+capPerSeg {
					newCap = len(scratch.flatResults) + capPerSeg
				}
				newFlat := make([]index.SearchResult, 0, newCap)
				newFlat = append(newFlat, scratch.flatResults...)
				scratch.flatResults = newFlat
			}

			// Start of this segment's results in flat buffer
			startIdx := len(scratch.flatResults)

			// Search and append directly to flatResults
			if err := seg.SearchWithBuffer(ctx, query, k, nil, segmentFilter, scratch, &scratch.flatResults); err != nil {
				return fmt.Errorf("diskann: segment search: %w", err)
			}

			// End of this segment's results
			endIdx := len(scratch.flatResults)

			// Capture the slice for this segment (points into flatResults)
			scratch.segmentResults = append(scratch.segmentResults, scratch.flatResults[startIdx:endIdx])
		}

		// Merge results
		// We can use MergeNSearchResultsInto to merge all segment results + memtable results
		// But here we are just collecting them into scratch.results
		// The original code appended to scratch.results.

		// Let's append all segment results to scratch.results
		// Actually, scratch.results already contains MemTable results.
		// We should merge MemTable results + Segment results.

		// Current state:
		// scratch.results: [MemTable Results...]
		// scratch.segmentResults: [[Seg1 Results...], [Seg2 Results...]]

		// We can just append everything to scratch.results and sort/dedup as before.
		// This is what the original code did.
		// But wait, scratch.flatResults ALREADY contains all segment results!
		// So we just need to append scratch.flatResults to scratch.results.

		scratch.results = append(scratch.results, scratch.flatResults...)
	}

	// Merge and Deduplicate
	// Sort by distance
	slices.SortFunc(scratch.results, func(a, b index.SearchResult) int {
		if a.Distance < b.Distance {
			return -1
		}
		if a.Distance > b.Distance {
			return 1
		}
		return 0
	})

	// Deduplicate and take top-k
	count := 0
	maxID := idx.nextIDAtomic.Load()
	useBits := maxID < 10_000_000 // Use bitset for < 10M items (~1.25MB)

	if useBits {
		scratch.seenBits.Grow(maxID)
	}

	for _, res := range scratch.results {
		if count >= k {
			break
		}

		seen := false
		if useBits {
			if scratch.seenBits.Test(res.ID) {
				seen = true
			} else {
				scratch.seenBits.Set(res.ID)
			}
		} else {
			if _, ok := scratch.seen[res.ID]; ok {
				seen = true
			} else {
				scratch.seen[res.ID] = struct{}{}
			}
		}

		if !seen {
			*buf = append(*buf, res)
			count++
		}
	}

	return nil
}

// KNNSearchWithContext performs a K-nearest neighbor search using the provided Searcher context.
func (idx *Index) KNNSearchWithContext(ctx context.Context, s *searcher.Searcher, query []float32, k int, opts *index.SearchOptions) error {
	if len(query) != idx.dim {
		return &index.ErrDimensionMismatch{Expected: idx.dim, Actual: len(query)}
	}
	if k <= 0 {
		return index.ErrInvalidK
	}

	idx.mu.RLock()
	memTable := idx.memTable
	segments := idx.segments
	deleted := idx.deleted
	memTablePresent := idx.memTablePresent
	idx.mu.RUnlock()

	// Wrap filter to exclude deleted items
	var userFilter func(uint64) bool
	if opts != nil {
		userFilter = opts.Filter
	}

	// Base filter: deleted check
	baseFilter := func(id uint64) bool {
		if deleted.Test(id) {
			return false
		}
		if userFilter != nil {
			return userFilter(id)
		}
		return true
	}

	// Segment filter: base filter + check if shadowed by MemTable
	segmentFilter := func(id uint64) bool {
		if !baseFilter(id) {
			return false
		}
		// Check shadowing
		return !memTablePresent.Test(id)
	}

	// Search MemTable
	if memTable != nil {
		memOpts := index.SearchOptions{
			Filter: baseFilter,
		}
		if opts != nil {
			memOpts.EFSearch = opts.EFSearch
		}

		// Use Zero-Alloc KNNSearchWithContext
		if err := memTable.KNNSearchWithContext(ctx, s, query, k, &memOpts); err != nil {
			return fmt.Errorf("diskann: memtable search: %w", err)
		}
	}

	// Search DiskANN Segments
	for _, seg := range segments {
		// Search and push directly to s.Candidates
		if err := seg.SearchWithContext(ctx, query, k, nil, segmentFilter, s); err != nil {
			return fmt.Errorf("diskann: segment search: %w", err)
		}
	}

	return nil
}

// BruteSearch performs a brute-force search on the index.
func (idx *Index) BruteSearch(ctx context.Context, query []float32, k int, filter func(id uint64) bool) ([]index.SearchResult, error) {
	idx.mu.RLock()
	maxID := idx.nextIDAtomic.Load()
	idx.mu.RUnlock()

	h := &maxDistHeap{}
	heap.Init(h)

	for id := uint64(0); id < maxID; id++ {
		// Check filter
		if filter != nil && !filter(id) {
			continue
		}

		// Check deleted
		if idx.deleted.Test(id) {
			continue
		}

		vec, err := idx.VectorByID(ctx, id)
		if err != nil {
			continue
		}

		dist := idx.distFunc(query, vec)

		if h.Len() < k {
			heap.Push(h, distNode{id: id, dist: dist})
		} else if dist < (*h)[0].dist {
			heap.Pop(h)
			heap.Push(h, distNode{id: id, dist: dist})
		}
	}

	// Extract results
	results := make([]index.SearchResult, h.Len())
	for i := h.Len() - 1; i >= 0; i-- {
		node := heap.Pop(h).(distNode)
		results[i] = index.SearchResult{ID: node.id, Distance: node.dist}
	}

	return results, nil
}

// KNNSearchStream returns an iterator over search results.
func (idx *Index) KNNSearchStream(ctx context.Context, query []float32, k int, opts *index.SearchOptions) iter.Seq2[index.SearchResult, error] {
	return func(yield func(index.SearchResult, error) bool) {
		results, err := idx.KNNSearch(ctx, query, k, opts)
		if err != nil {
			yield(index.SearchResult{}, err)
			return
		}

		for _, r := range results {
			if !yield(r, nil) {
				return
			}
		}
	}
}

// ============================================================================
// Statistics and Info
// ============================================================================

// Stats returns index statistics.
func (idx *Index) Stats() index.Stats {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	totalCount := idx.count.Load()

	deletedCount := idx.deleted.Count()

	liveCount := int(totalCount) - int(deletedCount)

	storage := map[string]string{
		"VectorCount":   fmt.Sprintf("%d", liveCount),
		"MemTableCount": fmt.Sprintf("%d", idx.memTableCount.Load()),
		"SegmentCount":  fmt.Sprintf("%d", len(idx.segments)),
	}

	// Include deletion stats if there are any deleted vectors
	if deletedCount > 0 {
		storage["TotalCount"] = fmt.Sprintf("%d", totalCount)
		storage["DeletedCount"] = fmt.Sprintf("%d", deletedCount)
	}

	return index.Stats{
		Options: map[string]string{
			"Dimension":    fmt.Sprintf("%d", idx.dim),
			"DistanceType": idx.distType.String(),
			"Mode":         "LSM",
		},
		Parameters: map[string]string{
			"R":            fmt.Sprintf("%d", idx.opts.R),
			"L":            fmt.Sprintf("%d", idx.opts.L),
			"Alpha":        fmt.Sprintf("%.2f", idx.opts.Alpha),
			"PQSubvectors": fmt.Sprintf("%d", idx.opts.PQSubvectors),
			"PQCentroids":  fmt.Sprintf("%d", idx.opts.PQCentroids),
		},
		Storage: storage,
	}
}

// Dimension returns the vector dimension.
func (idx *Index) Dimension() int {
	return idx.dim
}

// Count returns the number of live (non-deleted) vectors.
func (idx *Index) Count() int {
	deletedCount := idx.deleted.Count()
	total := idx.count.Load()

	return int(total) - int(deletedCount)
}

// ============================================================================
// Compaction (removes deleted vectors, rebuilds graph, re-trains PQ)
// ============================================================================

// ============================================================================
// LSM Operations (Flush & Compaction)
// ============================================================================

// Flush forces the MemTable to be written to a new disk segment.
func (idx *Index) Flush(ctx context.Context) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	if idx.memTable == nil || idx.memTableCount.Load() == 0 {
		return nil
	}

	// Prepare segment path
	baseID := idx.memTableStartID
	segmentName := fmt.Sprintf("segment-%019d", baseID)
	segmentPath := filepath.Join(idx.indexPath, segmentName)

	// Create builder
	builder, err := NewBuilder(idx.dim, idx.distType, segmentPath, idx.opts)
	if err != nil {
		return fmt.Errorf("diskann: create builder: %w", err)
	}

	// Extract vectors from MemTable
	// We iterate from baseID to nextID
	endID := idx.nextIDAtomic.Load()
	vectors := make([][]float32, 0, endID-baseID)

	for id := baseID; id < endID; id++ {
		// Check if deleted
		isDeleted := idx.deleted.Test(id)

		var vec []float32
		if !isDeleted {
			var err error
			vec, err = idx.memTable.VectorByID(ctx, id)
			if err != nil {
				// Should not happen for live vectors in MemTable
				// But if it does, we treat it as deleted/zero
				vec = make([]float32, idx.dim)
			}
		} else {
			// Placeholder for deleted vector to maintain ID alignment
			vec = make([]float32, idx.dim)
		}
		vectors = append(vectors, vec)
	}

	// Build segment
	if _, err := builder.AddBatch(vectors); err != nil {
		return fmt.Errorf("diskann: add batch: %w", err)
	}
	if err := builder.Build(ctx); err != nil {
		return fmt.Errorf("diskann: build segment: %w", err)
	}

	// Open new segment
	seg, err := OpenSegment(segmentPath, baseID, idx.opts)
	if err != nil {
		return fmt.Errorf("diskann: open new segment: %w", err)
	}

	// Add to segments list
	idx.segments = append(idx.segments, seg)

	// Reset MemTable
	if err := idx.resetMemTable(); err != nil {
		return err
	}
	idx.memTableStartID = endID

	return nil
}

// Compact merges all segments into a single new segment (Major Compaction).
// This is a blocking operation.
func (idx *Index) Compact(ctx context.Context) error {
	// Prevent concurrent compactions
	if !idx.compacting.CompareAndSwap(false, true) {
		return fmt.Errorf("diskann: compaction already in progress")
	}
	defer idx.compacting.Store(false)

	// 1. Flush MemTable first to ensure everything is on disk
	if err := idx.Flush(ctx); err != nil {
		return fmt.Errorf("diskann: flush before compact: %w", err)
	}

	idx.mu.Lock()
	defer idx.mu.Unlock()

	if len(idx.segments) == 0 {
		return nil // Nothing to compact
	}

	start := time.Now()

	// Create builder
	// Use a unique name to avoid conflicts
	mergedName := fmt.Sprintf("segment-merged-%d", time.Now().UnixNano())
	mergedPath := filepath.Join(idx.indexPath, mergedName)

	builder, err := NewBuilder(idx.dim, idx.distType, mergedPath, idx.opts)
	if err != nil {
		return fmt.Errorf("diskann: create builder: %w", err)
	}

	// Iterate all segments
	vectorsRemoved := 0
	oldSegments := idx.segments

	for _, seg := range oldSegments {
		for i := uint64(0); i < seg.count; i++ {
			globalID := seg.baseID + i

			// Check deleted
			isDeleted := idx.deleted.Test(globalID)

			if isDeleted {
				vectorsRemoved++
				continue
			}

			vec := seg.getVector(i)
			if vec == nil {
				continue // Should not happen
			}

			if _, err := builder.Add(vec); err != nil {
				return fmt.Errorf("diskann: add vector: %w", err)
			}
		}
	}

	// Build
	if err := builder.Build(ctx); err != nil {
		return fmt.Errorf("diskann: build merged segment: %w", err)
	}

	// Open new segment
	newSeg, err := OpenSegment(mergedPath, 0, idx.opts)
	if err != nil {
		return fmt.Errorf("diskann: open merged segment: %w", err)
	}

	// Close old segments
	for _, seg := range oldSegments {
		_ = seg.Close()
		// We could delete files here, but safer to do it after swap?
		// Or keep them as backup?
		// For now, we delete them to save space.
		_ = os.RemoveAll(seg.path)
	}

	// Replace segments
	idx.segments = []*Segment{newSeg}
	idx.count.Store(newSeg.count)
	idx.memTableStartID = newSeg.count
	idx.nextIDAtomic.Store(newSeg.count)

	// Reset deleted
	idx.deleted = bitset.New(newSeg.count)

	// Reset shadowing
	idx.memTablePresent = bitset.New(newSeg.count)

	// Update stats
	idx.compactionMu.Lock()
	idx.compactionStats.TotalCompactions++
	idx.compactionStats.VectorsRemovedTotal += uint64(vectorsRemoved)
	idx.compactionStats.LastVectorsRemoved = uint32(vectorsRemoved)
	idx.compactionStats.LastCompactionTime = time.Now().Unix()
	idx.compactionStats.LastCompactionDuration = time.Since(start).Milliseconds()
	idx.compactionMu.Unlock()

	return nil
}

// ShouldCompact returns true if compaction is recommended.
func (idx *Index) ShouldCompact() bool {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	// Compact if too many segments
	if len(idx.segments) > 4 {
		return true
	}

	// Compact if deleted ratio is high
	deletedCount := idx.deleted.Count()
	count := idx.count.Load()

	if count > 0 && float32(deletedCount)/float32(count) >= idx.opts.CompactionThreshold {
		return true
	}

	return false
}

// CompactionStats returns current compaction statistics.
func (idx *Index) CompactionStats() CompactionStats {
	idx.compactionMu.RLock()
	defer idx.compactionMu.RUnlock()
	return idx.compactionStats
}

// backgroundCompaction runs periodic compaction checks.
func (idx *Index) backgroundCompaction() {
	defer idx.compactionWg.Done()

	ticker := newTicker(idx.opts.CompactionInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			if idx.ShouldCompact() {
				ctx := context.Background()
				_ = idx.Compact(ctx)
			}
		case <-idx.stopCompaction:
			return
		}
	}
}

// ============================================================================
// Heap types for search (maxDistHeap only - distHeap/distNode in builder.go)
// ============================================================================

// maxDistHeap is a max-heap of distNodes for top-k selection.
type maxDistHeap []distNode

func (h maxDistHeap) Len() int           { return len(h) }
func (h maxDistHeap) Less(i, j int) bool { return h[i].dist > h[j].dist } // Max heap
func (h maxDistHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *maxDistHeap) Push(x interface{}) {
	*h = append(*h, x.(distNode))
}

func (h *maxDistHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

// ============================================================================
// Helper functions
// ============================================================================

// nowMillis returns current time in milliseconds since Unix epoch.
func nowMillis() int64 {
	return time.Now().UnixMilli()
}

// newTicker creates a ticker that ticks every interval seconds.
func newTicker(intervalSeconds int) *time.Ticker {
	return time.NewTicker(time.Duration(intervalSeconds) * time.Second)
}
