package searcher

import (
	"math/rand"
	"testing"
	"time"
)

func TestCandidateHeap(t *testing.T) {
	t.Run("Ascending (L2)", func(t *testing.T) {
		h := NewCandidateHeap(10, false) // false = ascending (min score is best)
		if h.Descending() {
			t.Error("expected ascending")
		}

		// Push random scores
		rng := rand.New(rand.NewSource(time.Now().UnixNano()))
		for i := 0; i < 100; i++ {
			h.Push(InternalCandidate{
				Score:     rng.Float32(),
				SegmentID: 1,
				RowID:     uint32(i),
			})
		}

		if h.Len() != 100 {
			t.Errorf("expected len 100, got %d", h.Len())
		}

		// Pop should return worst elements first (max score in ascending mode)
		// Because the heap keeps the "worst" element at the top.

		prev := h.Pop()
		for h.Len() > 0 {
			curr := h.Pop()
			if curr.Score > prev.Score {
				t.Errorf("expected worst-first pop order (decreasing score for L2), got %v then %v", prev.Score, curr.Score)
			}
			prev = curr
		}
	})

	t.Run("Descending (Cosine/Dot)", func(t *testing.T) {
		h := NewCandidateHeap(10, true) // true = descending (max score is best)

		h.Push(InternalCandidate{Score: 0.1})
		h.Push(InternalCandidate{Score: 0.9})
		h.Push(InternalCandidate{Score: 0.5})

		// Worst element for Dot is smallest score.
		// So heap top should be 0.1
		top := h.Candidates[0]
		if top.Score != 0.1 {
			t.Errorf("expected top to be worst (0.1), got %v", top.Score)
		}

		popped := h.Pop()
		if popped.Score != 0.1 {
			t.Errorf("expected pop to be 0.1, got %v", popped.Score)
		}

		popped = h.Pop()
		if popped.Score != 0.5 {
			t.Errorf("expected pop to be 0.5, got %v", popped.Score)
		}

		popped = h.Pop()
		if popped.Score != 0.9 {
			t.Errorf("expected pop to be 0.9, got %v", popped.Score)
		}
	})

	t.Run("TieBreaking", func(t *testing.T) {
		h := NewCandidateHeap(10, false)

		// Same score, different locations
		c1 := InternalCandidate{Score: 1.0, SegmentID: 1, RowID: 10}
		c2 := InternalCandidate{Score: 1.0, SegmentID: 1, RowID: 20}

		// CandidateBetter says c1 is better than c2 because RowID 10 < 20.
		// CandidateWorse says c2 is worse than c1.
		// Heap top is WORST. So c2 should be at top if both are in.

		h.Push(c1)
		h.Push(c2)

		if h.Candidates[0].RowID != 20 {
			t.Errorf("expected worse candidate (row 20) at top, got %v", h.Candidates[0].RowID)
		}
	})

	t.Run("ReplaceTop", func(t *testing.T) {
		h := NewCandidateHeap(10, true) // Descending (Max best). Worst is Min.

		h.Push(InternalCandidate{Score: 0.1}) // Worst
		h.Push(InternalCandidate{Score: 0.5})
		h.Push(InternalCandidate{Score: 0.9})

		// Create a new candidate better than worst (0.2 > 0.1)
		h.ReplaceTop(InternalCandidate{Score: 0.2})

		// New worst should be 0.2
		if h.Candidates[0].Score != 0.2 {
			t.Errorf("expected new worst 0.2, got %v", h.Candidates[0].Score)
		}

		// Now replace with something very good (1.0)
		h.ReplaceTop(InternalCandidate{Score: 1.0})

		// Now worst should be 0.5
		if h.Candidates[0].Score != 0.5 {
			t.Errorf("expected new worst 0.5, got %v", h.Candidates[0].Score)
		}
	})

	t.Run("GetCandidates", func(t *testing.T) {
		h := NewCandidateHeap(10, false)
		h.Push(InternalCandidate{Score: 10})
		h.Push(InternalCandidate{Score: 5})
		h.Push(InternalCandidate{Score: 20})

		h.Reset(true)

		h.Push(InternalCandidate{Score: 0.1})
		h.Push(InternalCandidate{Score: 0.9})
		h.Push(InternalCandidate{Score: 0.5})

		cands := h.GetCandidates()
		if len(cands) != 3 {
			t.Errorf("expected 3 candidates, got %d", len(cands))
		}

		// In descending mode (Max score is best), the heap keeps the "worst" (min score) at the top
		// so it can be evicted.
		// So h.Candidates[0] should be 0.1.
		// GetCandidates returns the raw heap slice.
		if cands[0].Score != 0.1 {
			t.Errorf("expected heap top (worst=0.1) at index 0, got %v", cands[0].Score)
		}
	})

	t.Run("Determinism", func(t *testing.T) {
		// Verify strict determinism across many operations
		makeCands := func() []InternalCandidate {
			rng := rand.New(rand.NewSource(42))
			cands := make([]InternalCandidate, 1000)
			for i := range cands {
				cands[i] = InternalCandidate{
					Score:     float32(rng.Intn(10)), // Many duplicates
					SegmentID: uint32(rng.Intn(5)),
					RowID:     uint32(rng.Intn(100)),
				}
			}
			return cands
		}

		runHeap := func() []InternalCandidate {
			h := NewCandidateHeap(1000, false)
			input := makeCands()
			for _, c := range input {
				h.Push(c)
			}
			res := make([]InternalCandidate, 0)
			for h.Len() > 0 {
				res = append(res, h.Pop())
			}
			return res
		}

		res1 := runHeap()
		res2 := runHeap()

		if len(res1) != len(res2) {
			t.Fatal("length mismatch")
		}
		for i := range res1 {
			c1 := res1[i]
			c2 := res2[i]
			if c1.Score != c2.Score || c1.SegmentID != c2.SegmentID || c1.RowID != c2.RowID {
				t.Fatalf("mismatch at %d: %v vs %v", i, c1, c2)
			}
		}
	})

	t.Run("InternalCandidateBetter", func(t *testing.T) {
		// Test the standalone comparison function
		c1 := InternalCandidate{Score: 10}
		c2 := InternalCandidate{Score: 20}

		// L2 (Ascending): smaller is better
		if !InternalCandidateBetter(c1, c2, false) {
			t.Error("10 better than 20 for L2")
		}
		if InternalCandidateBetter(c2, c1, false) {
			t.Error("20 not better than 10 for L2")
		}

		// Dot (Descending): larger is better
		if InternalCandidateBetter(c1, c2, true) {
			t.Error("10 not better than 20 for Dot")
		}
		if !InternalCandidateBetter(c2, c1, true) {
			t.Error("20 better than 10 for Dot")
		}

		// Tie-breaking
		c3 := InternalCandidate{Score: 10, SegmentID: 1, RowID: 5}
		c4 := InternalCandidate{Score: 10, SegmentID: 1, RowID: 10} // Higher RowID is worse (tie-break uses ID ascending for determinism)

		// Better = smaller ID
		if !InternalCandidateBetter(c3, c4, false) {
			t.Error("same score, smaller ID should be better")
		}

		// Tie-breaking (SegmentID)
		c5 := InternalCandidate{Score: 10, SegmentID: 1, RowID: 10}
		c6 := InternalCandidate{Score: 10, SegmentID: 2, RowID: 5} // Higher SegmentID is worse

		if !InternalCandidateBetter(c5, c6, false) {
			t.Error("same score, smaller SegmentID should be better")
		}
	})
}
