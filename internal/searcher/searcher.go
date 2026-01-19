package searcher

import (
	"sync"

	"github.com/hupe1980/vecgo/internal/bitmap"
	imetadata "github.com/hupe1980/vecgo/internal/metadata"
	"github.com/hupe1980/vecgo/model"
)

// DefaultUniverseSize is the default maximum row count for QueryBitmap.
// 1M rows covers typical segments; larger segments will trigger reallocation.
const DefaultUniverseSize = 1 << 20 // 1M

// Searcher is a reusable execution context for vector search operations.
// It owns all scratch memory required for search, eliminating heap allocations
// in the steady state.
//
// Searcher is NOT thread-safe. It is intended to be owned by a single goroutine
// during a search operation.
type Searcher struct {
	// Visited tracks visited nodes during graph traversal.
	Visited *VisitedSet

	// Candidates is a min-heap for tracking the best candidates found so far.
	// Used for the result set (keeping top K).
	Candidates *PriorityQueue

	// ScratchCandidates is a max-heap for tracking candidates to explore.
	// Used for the exploration set (ef).
	ScratchCandidates *PriorityQueue

	// ScratchVec is a reusable buffer for vector operations (e.g. decompression).
	ScratchVec []float32

	// IOBuffer is a reusable buffer for disk I/O (DiskANN).
	IOBuffer []byte

	// FilterBitmap is a reusable bitmap for metadata filtering.
	FilterBitmap *imetadata.LocalBitmap

	// ScratchResults is a reusable buffer for collecting intermediate results (e.g. DiskANN beam search).
	ScratchResults []PriorityQueueItem

	// CandidateHeap is a reusable heap for model.Candidate (used by flat search).
	Heap *CandidateHeap

	// Scores is a reusable buffer for batch distance calculations.
	Scores []float32

	// ScratchMap is a reusable map for hybrid search scoring.
	ScratchMap map[model.ID]float32

	// Results is a reusable buffer for collecting final results before returning.
	Results []model.Candidate

	// CandidateBuffer is a reusable buffer for intermediate results (e.g. reranking).
	CandidateBuffer []InternalCandidate

	// ModelScratch is a reusable buffer for converting InternalCandidates to model.Candidates (e.g. for Rerank).
	ModelScratch []model.Candidate

	// ScratchIDs is a reusable buffer for RowIDs.
	ScratchIDs []uint32

	// ScratchBools is a reusable buffer for batch boolean results (e.g., tombstone filtering).
	ScratchBools []bool

	// ScratchForeignIDs is a reusable buffer for PrimaryKeys.
	ScratchForeignIDs []model.ID

	// ScratchCols is a reusable buffer for column names during fetch (max 3: vector, metadata, payload).
	ScratchCols [3]string

	// ParallelResults is a reusable buffer for collecting results from parallel segment searches.
	ParallelResults []InternalCandidate

	// ParallelSlices is a reusable buffer for slice headers pointing into ParallelResults.
	ParallelSlices [][]InternalCandidate

	// SegmentFilters is a reusable buffer for segment filters.
	SegmentFilters []any

	// SemChan is a reusable semaphore channel for parallel search.
	SemChan chan struct{}

	// ScratchVecBuf is a reusable buffer for batch vector fetching (FetchVectors).
	// Size = batchSize * dim (typically 64 * 1536 = 98,304 floats = 384KB).
	ScratchVecBuf []float32

	// BitmapBuilder is a zero-alloc bitmap builder for filter evaluation.
	// Collects rowIDs into slice, builds bitmap once via AddMany.
	BitmapBuilder *imetadata.BitmapBuilder

	// MatcherScratch is a pooled scratch for building FilterMatcher composites.
	// Eliminates closure allocations in the filter hot path.
	MatcherScratch *imetadata.MatcherScratch

	// QueryBitmap is a SIMD-friendly bitmap for query-time filter operations.
	// Supports O(1) block skipping, active-mask-driven AND/OR, and zero allocations.
	// This is the hot-path bitmap for filtered graph traversal.
	QueryBitmap *bitmap.QueryBitmap

	// OpsPerformed tracks the number of distance calculations or node visits.
	OpsPerformed int

	// FilterGateStats tracks filter effectiveness during graph traversal.
	// Updated atomically during search for query feedback.
	FilterGateStats FilterGateStats
}

// FilterGateStats tracks filter effectiveness during HNSW traversal.
// Used for query feedback and adaptive traversal decisions.
type FilterGateStats struct {
	// NodesVisited is the total number of graph nodes visited.
	NodesVisited int
	// NodesPassedFilter is the count of nodes that passed the filter.
	NodesPassedFilter int
	// NodesRejectedByFilter is the count rejected by filter (before distance).
	NodesRejectedByFilter int
	// ExpansionsSkipped is the count of graph expansions skipped due to filter gating.
	ExpansionsSkipped int
	// DistanceComputations is the number of distance computations performed.
	DistanceComputations int
	// DistanceShortCircuits is the count of distance computations short-circuited.
	DistanceShortCircuits int
	// CandidatesEvaluated is the number of candidates evaluated (pushed to heap).
	CandidatesEvaluated int
	// BruteForceSegments is the number of segments searched with brute-force.
	BruteForceSegments int
	// HNSWSegments is the number of segments searched with HNSW.
	HNSWSegments int
	// FilterTimeNanos is the cumulative time spent on filter evaluation.
	FilterTimeNanos int64
	// SearchTimeNanos is the cumulative time spent on search (distance computation + heap operations).
	SearchTimeNanos int64
}

// Reset clears the filter gate stats.
func (f *FilterGateStats) Reset() {
	f.NodesVisited = 0
	f.NodesPassedFilter = 0
	f.NodesRejectedByFilter = 0
	f.ExpansionsSkipped = 0
	f.DistanceComputations = 0
	f.DistanceShortCircuits = 0
	f.CandidatesEvaluated = 0
	f.BruteForceSegments = 0
	f.HNSWSegments = 0
	f.FilterTimeNanos = 0
	f.SearchTimeNanos = 0
}

// Add aggregates stats from another FilterGateStats instance.
// Used to merge stats from parallel searchers.
func (f *FilterGateStats) Add(other *FilterGateStats) {
	f.NodesVisited += other.NodesVisited
	f.NodesPassedFilter += other.NodesPassedFilter
	f.NodesRejectedByFilter += other.NodesRejectedByFilter
	f.ExpansionsSkipped += other.ExpansionsSkipped
	f.DistanceComputations += other.DistanceComputations
	f.DistanceShortCircuits += other.DistanceShortCircuits
	f.CandidatesEvaluated += other.CandidatesEvaluated
	f.BruteForceSegments += other.BruteForceSegments
	f.HNSWSegments += other.HNSWSegments
	f.FilterTimeNanos += other.FilterTimeNanos
	f.SearchTimeNanos += other.SearchTimeNanos
}

// FilterPassRate returns the filter pass rate (0.0-1.0).
func (f *FilterGateStats) FilterPassRate() float64 {
	if f.NodesVisited == 0 {
		return 1.0
	}
	return float64(f.NodesPassedFilter) / float64(f.NodesVisited)
}

// FilterIsSelective returns true if the filter is selective enough to benefit
// from predicate-aware traversal. After warmup (100 visits), if >50% of nodes
// pass the filter, predicate checking is a pessimization.
// Used for adaptive mode switching (like Snowflake/DuckDB/ClickHouse).
func (s *Searcher) FilterIsSelective() bool {
	// During warmup, assume filter is selective (conservative)
	if s.FilterGateStats.NodesVisited < 100 {
		return true
	}
	// Filter is selective if <50% of nodes pass
	return s.FilterGateStats.FilterPassRate() < 0.5
}

var searcherPool = sync.Pool{
	New: func() any {
		return NewSearcher(1024, 128) // Default initial capacity
	},
}

// NewSearcher creates a new searcher with the given initial capacities.
func NewSearcher(visitedCap, queueCap int) *Searcher {
	// Pre-allocate all scratch buffers to avoid allocations in hot path.
	// Default capacity covers typical k=10..100 searches.
	// Pre-size SemChan to GOMAXPROCS to avoid reallocation under burst load.
	// Pre-size ParallelResults for typical 16 segments * 100 k = 1600 results.
	//
	// PriorityQueue capacity: Default EF is 300, so pre-allocate for that.
	// This eliminates growslice allocations which dominate mid-selectivity search CPU.
	pqCap := max(queueCap, 512) // At least 512 to cover EF=300 + headroom
	return &Searcher{
		Visited:           NewVisitedSet(visitedCap),
		Candidates:        NewPriorityQueueWithCapacity(true, pqCap),  // MaxHeap for results
		ScratchCandidates: NewPriorityQueueWithCapacity(false, pqCap), // MinHeap for exploration
		FilterBitmap:      imetadata.NewLocalBitmap(),
		ScratchResults:    make([]PriorityQueueItem, 0, queueCap),
		Heap:              NewCandidateHeap(queueCap, false),
		Scores:            make([]float32, 256),
		ScratchMap:        make(map[model.ID]float32),
		// IOBuffer for DiskANN PQ/INT4/RaBitQ code reads (256 bytes covers most quantization formats)
		IOBuffer: make([]byte, 256),
		// Pre-allocate result/scratch buffers to reduce allocations
		Results:           make([]model.Candidate, 0, queueCap),
		CandidateBuffer:   make([]InternalCandidate, 0, queueCap),
		ModelScratch:      make([]model.Candidate, 0, queueCap),
		ScratchIDs:        make([]uint32, 0, 1024), // Larger default for filtered search
		ScratchBools:      make([]bool, 0, 1024),   // Pre-allocate for tombstone filtering
		ScratchForeignIDs: make([]model.ID, 0, queueCap),
		ParallelResults:   make([]InternalCandidate, 0, 1600), // 16 segments * 100 results
		ParallelSlices:    make([][]InternalCandidate, 0, 16),
		SegmentFilters:    make([]any, 0, 16),
		// Pre-size SemChan to GOMAXPROCS to avoid reallocation under burst
		SemChan: make(chan struct{}, 16), // Will grow if needed, but rarely
		// BitmapBuilder for zero-alloc filter evaluation
		BitmapBuilder: imetadata.NewBitmapBuilder(),
		// MatcherScratch for zero-alloc filter matcher building
		MatcherScratch: imetadata.GetMatcherScratch(),
		// QueryBitmap for SIMD-friendly filter operations (1M universe covers most segments)
		QueryBitmap: bitmap.New(DefaultUniverseSize),
		// ScratchVecBuf for batch vector fetching (64 batch * 512 dim = 32K floats covers most cases)
		ScratchVecBuf: make([]float32, 0, 64*512),
	}
}

// Get returns a Searcher from the pool.
func Get() *Searcher {
	s := searcherPool.Get().(*Searcher)
	s.Reset()
	return s
}

// Put returns a Searcher to the pool.
func Put(s *Searcher) {
	searcherPool.Put(s)
}

// Reset clears the searcher state for reuse.
func (s *Searcher) Reset() {
	s.Visited.Reset()
	s.Candidates.Reset()
	s.ScratchCandidates.Reset()
	s.FilterBitmap.Clear()
	s.ScratchResults = s.ScratchResults[:0]
	s.Heap.Reset(false)
	clear(s.ScratchMap)

	// Reset all slice buffers to zero length (preserving capacity)
	s.Results = s.Results[:0]
	s.CandidateBuffer = s.CandidateBuffer[:0]
	s.ModelScratch = s.ModelScratch[:0]
	s.ScratchIDs = s.ScratchIDs[:0]
	s.ScratchBools = s.ScratchBools[:0]
	s.ScratchForeignIDs = s.ScratchForeignIDs[:0]
	s.ParallelResults = s.ParallelResults[:0]
	s.ParallelSlices = s.ParallelSlices[:0]

	// SegmentFilters holds references to objects, clear to avoid memory leaks
	clear(s.SegmentFilters) // Go 1.21 generic clear for slices
	s.SegmentFilters = s.SegmentFilters[:0]

	// Reset BitmapBuilder for next query
	if s.BitmapBuilder != nil {
		s.BitmapBuilder.Reset(0)
	}

	// Reset QueryBitmap for next query (only clears active blocks - O(active) not O(universe))
	if s.QueryBitmap != nil {
		s.QueryBitmap.Clear()
	}

	s.OpsPerformed = 0
	s.FilterGateStats.Reset()
}

// EnsureQueryBitmapSize ensures the QueryBitmap can hold the given universe size.
// If the current bitmap is too small, a new one is allocated (rare case).
func (s *Searcher) EnsureQueryBitmapSize(universeSize uint32) {
	if s.QueryBitmap == nil || s.QueryBitmap.UniverseSize() < universeSize {
		s.QueryBitmap = bitmap.New(universeSize)
	}
}
