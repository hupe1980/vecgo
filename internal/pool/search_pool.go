// Package pool provides object pools for zero-allocation search operations.
// Uses sync.Pool for automatic memory reuse and bitsets for efficient visited tracking.
package pool

import (
	"sync"

	"github.com/bits-and-blooms/bitset"
	"github.com/hupe1980/vecgo/internal/queue"
)

const (
	// DefaultMaxNodes is the default initial capacity for bitsets.
	// Set large enough to avoid reallocations for most use cases
	DefaultMaxNodes = 1000000

	// DefaultQueueCapacity is the default capacity for priority queues.
	DefaultQueueCapacity = 400

	// DefaultMaxDimensions is the default maximum vector dimensions.
	DefaultMaxDimensions = 2048

	// DefaultBatchSize is the default capacity for batch distance buffers.
	DefaultBatchSize = 128
)

// SearchContext contains pre-allocated buffers for HNSW search operations.
// All fields are reusable across multiple searches to eliminate allocations.
type SearchContext struct {
	Visited    *bitset.BitSet
	Candidates *queue.PriorityQueue
	Result     *queue.PriorityQueue
	TempVec    []float32

	// Batch distance computation buffers
	TempNeighborIDs []uint32    // Neighbor IDs for batch processing
	TempVectors     [][]float32 // Neighbor vectors for batch processing
	TempDistances   []float32   // Distance results for batch processing

	maxNodes uint32
}

// searchContextPool is the global pool of SearchContext objects.
var searchContextPool = sync.Pool{
	New: func() interface{} {
		return &SearchContext{
			Visited:         bitset.New(DefaultMaxNodes),
			Candidates:      queue.NewMin(DefaultQueueCapacity),
			Result:          queue.NewMax(DefaultQueueCapacity),
			TempVec:         make([]float32, 0, DefaultMaxDimensions),
			TempNeighborIDs: make([]uint32, 0, DefaultBatchSize),
			TempVectors:     make([][]float32, 0, DefaultBatchSize),
			TempDistances:   make([]float32, DefaultBatchSize),
			maxNodes:        DefaultMaxNodes,
		}
	},
}

// Get retrieves a SearchContext from the pool.
func Get() *SearchContext {
	ctx := searchContextPool.Get().(*SearchContext)
	ctx.Reset()
	return ctx
}

// Put returns a SearchContext to the pool for reuse.
func Put(ctx *SearchContext) {
	if ctx.Visited.Len() > DefaultMaxNodes*10 {
		ctx.Visited = bitset.New(DefaultMaxNodes)
		ctx.maxNodes = DefaultMaxNodes
	}
	searchContextPool.Put(ctx)
}

// Reset clears the SearchContext for reuse.
func (sc *SearchContext) Reset() {
	sc.Visited.ClearAll()
	sc.Candidates.Reset()
	sc.Result.Reset()
	sc.TempVec = sc.TempVec[:0]
	sc.TempNeighborIDs = sc.TempNeighborIDs[:0]
	sc.TempVectors = sc.TempVectors[:0]
	// TempDistances capacity is preserved (no need to clear values)
}

// EnsureVisitedCapacity ensures the visited bitset can track up to nodeID.
func (sc *SearchContext) EnsureVisitedCapacity(nodeID uint32) {
	if nodeID >= sc.maxNodes {
		newSize := max(nodeID+1, sc.maxNodes*2)
		newBitset := bitset.New(uint(newSize))
		for i := uint(0); i < sc.Visited.Len(); i++ {
			if sc.Visited.Test(i) {
				newBitset.Set(i)
			}
		}
		sc.Visited = newBitset
		sc.maxNodes = uint32(newSize)
	}
}

// MarkVisited marks a node as visited.
// Returns true if the node was already visited, false otherwise.
func (sc *SearchContext) MarkVisited(nodeID uint32) bool {
	sc.EnsureVisitedCapacity(nodeID)
	if sc.Visited.Test(uint(nodeID)) {
		return true
	}
	sc.Visited.Set(uint(nodeID))
	return false
}

// IsVisited checks if a node has been visited.
func (sc *SearchContext) IsVisited(nodeID uint32) bool {
	if nodeID >= sc.maxNodes {
		return false
	}
	return sc.Visited.Test(uint(nodeID))
}

// SearchContextStats returns statistics about the SearchContext.
type SearchContextStats struct {
	VisitedCapacity uint32
	VisitedCount    uint
	CandidatesCount int
	ResultCount     int
	TempVecCap      int
}

// Stats returns current statistics about this SearchContext.
func (sc *SearchContext) Stats() SearchContextStats {
	return SearchContextStats{
		VisitedCapacity: sc.maxNodes,
		VisitedCount:    sc.Visited.Count(),
		CandidatesCount: sc.Candidates.Len(),
		ResultCount:     sc.Result.Len(),
		TempVecCap:      cap(sc.TempVec),
	}
}
