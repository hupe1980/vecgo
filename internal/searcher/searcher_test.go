package searcher

import (
	"testing"

	"github.com/hupe1980/vecgo/model"
)

func TestSearcher_Lifecycle(t *testing.T) {
	// 1. Get from pool (or create new)
	s := Get()

	// 2. Modify state
	s.Visited.Visit(1)

	s.Candidates.PushItem(PriorityQueueItem{Distance: 1.0, Node: 100})
	s.ScratchCandidates.PushItem(PriorityQueueItem{Distance: 2.0, Node: 200})

	s.Heap.Push(InternalCandidate{Score: 0.5, RowID: 5})

	s.ScratchMap[model.ID(123)] = 0.99
	s.OpsPerformed = 50

	// Add something to ScratchResults manually to test clearing
	s.ScratchResults = append(s.ScratchResults, PriorityQueueItem{Distance: 1.0, Node: 1})

	if s.Visited.Visited(1) == false {
		t.Fatal("Visited set should work")
	}

	// 3. Put back (which shouldn't necessarily reset, but Get should)
	Put(s)

	// 4. Get again -> Should be clean
	s2 := Get()

	// It MIGHT be the same specific instance or a new one, but if we assume single thread test, it's likely the same if the pool logic is simple.
	// However, sync.Pool doesn't guarantee which one you get.
	// But `Get()` calls `s.Reset()`.

	// Test Reset explicit call to ensure coverage
	s.Reset()

	if s.Visited.Visited(1) {
		t.Error("Visited not cleared")
	}
	if s.Candidates.Len() != 0 {
		t.Error("Candidates not cleared")
	}
	if s.ScratchCandidates.Len() != 0 {
		t.Error("ScratchCandidates not cleared")
	}
	if s.Heap.Len() != 0 {
		t.Error("Heap not cleared")
	}
	if len(s.ScratchMap) != 0 {
		t.Error("ScratchMap not cleared")
	}
	if s.OpsPerformed != 0 {
		t.Error("OpsPerformed not cleared")
	}
	if len(s.ScratchResults) != 0 {
		t.Error("ScratchResults not cleared")
	}

	// Cleanup
	Put(s2)
}

func TestNewSearcher(t *testing.T) {
	s := NewSearcher(10, 20)
	if s == nil {
		t.Fatal("NewSearcher returned nil")
	}
	if s.Visited == nil || s.Candidates == nil {
		t.Error("Components not initialized")
	}
	if cap(s.ScratchResults) != 20 {
		t.Errorf("ScratchResults cap mismatch: got %d, want 20", cap(s.ScratchResults))
	}
}

func TestSearcher_Reset_AllFields(t *testing.T) {
	s := NewSearcher(100, 50)

	// Populate all resetable fields
	s.Visited.Visit(1)
	s.Visited.Visit(50)
	s.Candidates.PushItem(PriorityQueueItem{Distance: 1.0, Node: 1})
	s.ScratchCandidates.PushItem(PriorityQueueItem{Distance: 2.0, Node: 2})
	s.Heap.Push(InternalCandidate{Score: 0.5, RowID: 5})
	s.ScratchMap[model.ID(123)] = 0.99
	s.ScratchResults = append(s.ScratchResults, PriorityQueueItem{Distance: 1.0, Node: 1})
	s.Results = append(s.Results, model.Candidate{Score: 1.0})
	s.CandidateBuffer = append(s.CandidateBuffer, InternalCandidate{Score: 2.0})
	s.ModelScratch = append(s.ModelScratch, model.Candidate{Score: 3.0})
	s.ScratchIDs = append(s.ScratchIDs, 100)
	s.ScratchForeignIDs = append(s.ScratchForeignIDs, model.ID(200))
	s.ParallelResults = append(s.ParallelResults, InternalCandidate{Score: 4.0})
	s.ParallelSlices = append(s.ParallelSlices, []InternalCandidate{{Score: 5.0}})
	s.SegmentFilters = append(s.SegmentFilters, "filter")
	s.OpsPerformed = 100

	// Reset
	s.Reset()

	// Verify all fields are cleared
	if s.Visited.Visited(1) || s.Visited.Visited(50) {
		t.Error("Visited not cleared")
	}
	if s.Candidates.Len() != 0 {
		t.Error("Candidates not cleared")
	}
	if s.ScratchCandidates.Len() != 0 {
		t.Error("ScratchCandidates not cleared")
	}
	if s.Heap.Len() != 0 {
		t.Error("Heap not cleared")
	}
	if len(s.ScratchMap) != 0 {
		t.Error("ScratchMap not cleared")
	}
	if len(s.ScratchResults) != 0 {
		t.Error("ScratchResults not cleared")
	}
	if len(s.Results) != 0 {
		t.Error("Results not cleared")
	}
	if len(s.CandidateBuffer) != 0 {
		t.Error("CandidateBuffer not cleared")
	}
	if len(s.ModelScratch) != 0 {
		t.Error("ModelScratch not cleared")
	}
	if len(s.ScratchIDs) != 0 {
		t.Error("ScratchIDs not cleared")
	}
	if len(s.ScratchForeignIDs) != 0 {
		t.Error("ScratchForeignIDs not cleared")
	}
	if len(s.ParallelResults) != 0 {
		t.Error("ParallelResults not cleared")
	}
	if len(s.ParallelSlices) != 0 {
		t.Error("ParallelSlices not cleared")
	}
	if len(s.SegmentFilters) != 0 {
		t.Error("SegmentFilters not cleared")
	}
	if s.OpsPerformed != 0 {
		t.Error("OpsPerformed not cleared")
	}

	// Verify capacities are preserved (no reallocation)
	if cap(s.ScratchResults) < 50 {
		t.Error("ScratchResults capacity shrunk")
	}
	if cap(s.Results) < 50 {
		t.Error("Results capacity shrunk")
	}
}

func TestSearcher_PoolConcurrency(t *testing.T) {
	// Test that pool works correctly under concurrent access
	const goroutines = 10
	const iterations = 100

	done := make(chan bool, goroutines)

	for g := 0; g < goroutines; g++ {
		go func(id int) {
			for i := 0; i < iterations; i++ {
				s := Get()

				// Verify clean state
				if s.Candidates.Len() != 0 {
					t.Errorf("goroutine %d, iter %d: Candidates not clean", id, i)
				}
				if s.OpsPerformed != 0 {
					t.Errorf("goroutine %d, iter %d: OpsPerformed not clean", id, i)
				}

				// Use the searcher
				s.Visited.Visit(model.RowID(id*1000 + i))
				s.Candidates.PushItem(PriorityQueueItem{Distance: float32(i), Node: model.RowID(i)})
				s.OpsPerformed = id*1000 + i

				Put(s)
			}
			done <- true
		}(g)
	}

	// Wait for all goroutines
	for i := 0; i < goroutines; i++ {
		<-done
	}
}
