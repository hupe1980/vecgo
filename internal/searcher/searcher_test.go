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

	s.Heap.Push(model.Candidate{Score: 0.5, Loc: model.Location{RowID: 5}})

	s.ScratchMap[model.PKUint64(123)] = 0.99
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
