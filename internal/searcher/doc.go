// Package searcher provides pooled search context for zero-allocation queries.
//
// # Overview
//
// The Searcher struct owns all reusable resources needed for vector search,
// eliminating heap allocations in the steady state by reusing scratch buffers.
//
// # Architecture
//
//	┌─────────────────────────────────────────────────────────────┐
//	│                       sync.Pool                             │
//	│  ┌─────────┐ ┌─────────┐ ┌─────────┐                       │
//	│  │Searcher │ │Searcher │ │Searcher │  ...                  │
//	│  └────┬────┘ └────┬────┘ └────┬────┘                       │
//	└───────┼───────────┼───────────┼─────────────────────────────┘
//	        │           │           │
//	        v           v           v
//	┌─────────────────────────────────────────────────────────────┐
//	│                    Searcher Components                      │
//	│                                                             │
//	│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
//	│  │ VisitedSet   │  │PriorityQueue │  │CandidateHeap │      │
//	│  │ (generation) │  │  (min/max)   │  │  (top-k)     │      │
//	│  └──────────────┘  └──────────────┘  └──────────────┘      │
//	│                                                             │
//	│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
//	│  │ ScratchVec   │  │  IOBuffer    │  │FilterBitmap  │      │
//	│  │(decompression)│  │ (disk I/O)  │  │ (metadata)   │      │
//	│  └──────────────┘  └──────────────┘  └──────────────┘      │
//	└─────────────────────────────────────────────────────────────┘
//
// # Components
//
// Priority Queues:
//   - PriorityQueue: Generic binary heap supporting min/max heap modes
//   - CandidateHeap: Specialized heap for InternalCandidate with tie-breaking
//
// Visited Tracking:
//   - VisitedSet: Generation-based visited tracking with O(1) reset
//
// Scratch Buffers:
//   - ScratchVec: Vector decompression buffer
//   - IOBuffer: Disk I/O buffer for DiskANN
//   - FilterBitmap: Metadata filtering bitmap
//   - Various result/candidate buffers for zero-allocation operation
//
// # Usage
//
//	// Get a searcher from the pool
//	s := searcher.Get()
//	defer searcher.Put(s)
//
//	// Use for graph traversal
//	s.Visited.EnsureCapacity(numNodes)
//	s.Candidates.Reset()
//	s.ScratchCandidates.Reset()
//
//	// Process candidates
//	for s.ScratchCandidates.Len() > 0 {
//	    item, _ := s.ScratchCandidates.PopItem()
//	    if s.Visited.Visited(item.Node) {
//	        continue
//	    }
//	    s.Visited.Visit(item.Node)
//	    // ... process node ...
//	}
//
// # Thread Safety
//
// Searcher instances are NOT thread-safe. Each goroutine performing a search
// should obtain its own Searcher from the pool. The pool itself (sync.Pool)
// is safe for concurrent access.
//
// # Memory Management
//
// The pool maintains Searcher instances between queries. Buffers grow as needed
// but never shrink, optimizing for peak workload performance. Reset() is called
// automatically when obtaining a Searcher via Get().
//
// # Performance Notes
//
//   - VisitedSet uses generation-based reset: O(1) instead of O(n)
//   - PriorityQueue is value-based: better cache locality, no interface overhead
//   - CandidateHeap has deterministic tie-breaking for reproducible results
//   - Bounded push operations avoid allocations for top-k collection
package searcher
