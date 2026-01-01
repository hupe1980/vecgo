package search

import (
	"github.com/hupe1980/vecgo/internal/queue"
	"github.com/hupe1980/vecgo/internal/visited"
)

// Searcher is a reusable execution context for vector search operations.
// It owns all scratch memory required for search, eliminating heap allocations
// in the steady state.
//
// Searcher is NOT thread-safe. It is intended to be owned by a single goroutine
// during a search operation.
type Searcher struct {
	// Visited tracks visited nodes during graph traversal.
	Visited *visited.VisitedSet

	// Candidates is a min-heap for tracking the best candidates found so far.
	// Used for the result set (keeping top K).
	Candidates *queue.PriorityQueue

	// ScratchCandidates is a max-heap for tracking candidates to explore.
	// Used for the exploration set (ef).
	ScratchCandidates *queue.PriorityQueue

	// ScratchVec is a reusable buffer for vector operations (e.g. decompression).
	ScratchVec []float32

	// IOBuffer is a reusable buffer for disk I/O (DiskANN).
	IOBuffer []byte

	// OpsPerformed tracks the number of distance calculations or node visits.
	OpsPerformed int
}

// NewSearcher creates a new Searcher with the given configuration.
// maxNodes is the maximum number of nodes in the index (for visited set sizing).
// dim is the vector dimension (for scratch vector sizing).
func NewSearcher(maxNodes int, dim int) *Searcher {
	return &Searcher{
		Visited:           visited.New(maxNodes),
		Candidates:        queue.NewMax(128), // Max-heap for results (keeps K smallest, evicts largest)
		ScratchCandidates: queue.NewMin(128), // Min-heap for exploration (explores closest first)
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
