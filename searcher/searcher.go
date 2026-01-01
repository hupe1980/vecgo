package searcher

import (
	"sync"
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

	// OpsPerformed tracks the number of distance calculations or node visits.
	OpsPerformed int
}

var searcherPool = sync.Pool{
	New: func() interface{} {
		return NewSearcher(1024, 128) // Default initial capacity
	},
}

// AcquireSearcher retrieves a Searcher from the pool and prepares it for use.
// It ensures the Searcher has sufficient capacity for maxNodes and dim.
func AcquireSearcher(maxNodes int, dim int) *Searcher {
	s := searcherPool.Get().(*Searcher)
	s.Visited.EnsureCapacity(maxNodes)
	if cap(s.ScratchVec) < dim {
		s.ScratchVec = make([]float32, dim)
	} else {
		s.ScratchVec = s.ScratchVec[:dim]
	}
	s.OpsPerformed = 0
	return s
}

// ReleaseSearcher resets the Searcher and returns it to the pool.
func ReleaseSearcher(s *Searcher) {
	s.Reset()
	searcherPool.Put(s)
}

// NewSearcher creates a new Searcher with the given configuration.
// maxNodes is the maximum number of nodes in the index (for visited set sizing).
// dim is the vector dimension (for scratch vector sizing).
func NewSearcher(maxNodes int, dim int) *Searcher {
	return &Searcher{
		Visited:           NewVisitedSet(maxNodes),
		Candidates:        NewMax(128), // Max-heap for results (keeps K smallest, evicts largest)
		ScratchCandidates: NewMin(128), // Min-heap for exploration (explores closest first)
		ScratchVec:        make([]float32, dim),
		IOBuffer:          make([]byte, 4096), // Default page size
	}
}

// Reset clears the searcher state for reuse without freeing memory.
func (s *Searcher) Reset() {
	s.Visited.Reset()
	s.Candidates.Reset()
	s.ScratchCandidates.Reset()
	// ScratchVec and IOBuffer don't need clearing, just overwriting
	s.OpsPerformed = 0
}

// EnsureQueueCapacity ensures that the candidate queues have sufficient capacity.
// This should be called before search with the expected ef/k values.
func (s *Searcher) EnsureQueueCapacity(capacity int) {
	s.Candidates.EnsureCapacity(capacity)
	s.ScratchCandidates.EnsureCapacity(capacity)
}
