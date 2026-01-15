package searcher

import (
	"sync"

	imetadata "github.com/hupe1980/vecgo/internal/metadata"
	"github.com/hupe1980/vecgo/model"
)

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

	// OpsPerformed tracks the number of distance calculations or node visits.
	OpsPerformed int
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
	return &Searcher{
		Visited:           NewVisitedSet(visitedCap),
		Candidates:        NewPriorityQueue(true),  // MaxHeap for results (keep smallest)
		ScratchCandidates: NewPriorityQueue(false), // MinHeap for exploration (explore closest)
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
		ScratchIDs:        make([]uint32, 0, queueCap),
		ScratchForeignIDs: make([]model.ID, 0, queueCap),
		ParallelResults:   make([]InternalCandidate, 0, queueCap*16),
		ParallelSlices:    make([][]InternalCandidate, 0, 16),
		SegmentFilters:    make([]any, 0, 16),
		// BitmapBuilder for zero-alloc filter evaluation
		BitmapBuilder: imetadata.NewBitmapBuilder(),
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

	s.OpsPerformed = 0
}
