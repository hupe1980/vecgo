package searcher

import (
	"sync"

	"github.com/hupe1980/vecgo/metadata"
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
	FilterBitmap *metadata.LocalBitmap

	// ScratchResults is a reusable buffer for collecting intermediate results (e.g. DiskANN beam search).
	ScratchResults []PriorityQueueItem

	// BQBuffer is a reusable buffer for Binary Quantization codes.
	BQBuffer []uint64

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
	return &Searcher{
		Visited:           NewVisitedSet(visitedCap),
		Candidates:        NewPriorityQueue(true),  // MaxHeap for results (keep smallest)
		ScratchCandidates: NewPriorityQueue(false), // MinHeap for exploration (explore closest)
		FilterBitmap:      metadata.NewLocalBitmap(),
		ScratchResults:    make([]PriorityQueueItem, 0, queueCap),
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
	// Re-create heaps to clear them (faster than popping all)
	// Or just slice to 0 if we implement Reset on PriorityQueue
	s.Candidates = NewPriorityQueue(true)
	s.ScratchCandidates = NewPriorityQueue(false)
	s.FilterBitmap.Clear()
	s.ScratchResults = s.ScratchResults[:0]
	s.OpsPerformed = 0
}
