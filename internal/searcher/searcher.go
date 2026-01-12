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

	// ScratchScores is a reusable buffer for batch distance calculations.
	ScratchScores []float32

	// BQBuffer is a reusable buffer for Binary Quantization codes.
	BQBuffer []uint64

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

	// ScratchIDs is a reusable buffer for PrimaryKeys.
	ScratchForeignIDs []model.ID
	// ParallelResults is a reusable buffer for collecting results from parallel segment searches.
	ParallelResults []InternalCandidate

	// ParallelSlices is a reusable buffer for slice headers pointing into ParallelResults.
	ParallelSlices [][]InternalCandidate

	// SegmentFilters is a reusable buffer for segment filters.
	SegmentFilters []any
	// OpsPerformed tracks the number of distance calculations or node visits.
	OpsPerformed int
}

var searcherPool = sync.Pool{
	New: func() interface{} {
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
		// Pre-allocate result/scratch buffers to reduce allocations
		Results:           make([]model.Candidate, 0, queueCap),
		CandidateBuffer:   make([]InternalCandidate, 0, queueCap),
		ModelScratch:      make([]model.Candidate, 0, queueCap),
		ScratchIDs:        make([]uint32, 0, queueCap),
		ScratchForeignIDs: make([]model.ID, 0, queueCap),
		ParallelResults:   make([]InternalCandidate, 0, queueCap*16),
		ParallelSlices:    make([][]InternalCandidate, 0, 16),
		SegmentFilters:    make([]any, 0, 16),
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
	s.ParallelSlices = s.ParallelSlices[:0]
	// Note: We don't shrink ParallelResults or SegmentFilters capacity,
	// just let them be overwritten/re-sliced by the caller.
	// But resetting length to 0 is safer for avoiding stale references if they held pointers (but they are internal/any).
	// SegmentFilters holds references to objects, so we SHOULD clear it to avoid memory leaks (holding onto filters).
	clear(s.SegmentFilters) // Go 1.21 generic clear for slices
	s.SegmentFilters = s.SegmentFilters[:0]

	s.OpsPerformed = 0
}
