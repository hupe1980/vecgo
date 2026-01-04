package searcher

import (
	"github.com/hupe1980/vecgo/model"
)

// CandidateHeap is a heap of model.Candidate.
// It is used for collecting top-k results.
type CandidateHeap struct {
	Candidates []model.Candidate
	descending bool // true if we want largest scores (Dot/Cosine), false for smallest (L2)
}

// CandidateBetter reports whether a is better than b under the metric direction.
// Tie-breaker is (SegmentID, RowID) ascending for determinism.
func CandidateBetter(a, b model.Candidate, descending bool) bool {
	if a.Score != b.Score {
		if descending {
			return a.Score > b.Score
		}
		return a.Score < b.Score
	}
	if a.Loc.SegmentID != b.Loc.SegmentID {
		return a.Loc.SegmentID < b.Loc.SegmentID
	}
	return a.Loc.RowID < b.Loc.RowID
}

// CandidateWorse reports whether a is worse than b under the metric direction.
// Tie-breaker is (SegmentID, RowID) descending for determinism (so the heap evicts larger locations first).
func CandidateWorse(a, b model.Candidate, descending bool) bool {
	if a.Score != b.Score {
		if descending {
			return a.Score < b.Score
		}
		return a.Score > b.Score
	}
	if a.Loc.SegmentID != b.Loc.SegmentID {
		return a.Loc.SegmentID > b.Loc.SegmentID
	}
	return a.Loc.RowID > b.Loc.RowID
}

// NewCandidateHeap creates a new CandidateHeap.
func NewCandidateHeap(capacity int, descending bool) *CandidateHeap {
	return &CandidateHeap{
		Candidates: make([]model.Candidate, 0, capacity),
		descending: descending,
	}
}

// Reset clears the heap for reuse.
func (h *CandidateHeap) Reset(descending bool) {
	h.Candidates = h.Candidates[:0]
	h.descending = descending
}

// Descending returns true if the heap is configured for descending scores (Dot/Cosine).
func (h *CandidateHeap) Descending() bool {
	return h.descending
}

func (h *CandidateHeap) Len() int { return len(h.Candidates) }

func (h *CandidateHeap) Swap(i, j int) {
	h.Candidates[i], h.Candidates[j] = h.Candidates[j], h.Candidates[i]
}

func (h *CandidateHeap) Less(i, j int) bool {
	// The heap is ordered by "worst first" so the top element is the eviction candidate.
	return CandidateWorse(h.Candidates[i], h.Candidates[j], h.descending)
}

func (h *CandidateHeap) Push(x model.Candidate) {
	h.Candidates = append(h.Candidates, x)
	h.up(h.Len() - 1)
}

func (h *CandidateHeap) Pop() model.Candidate {
	n := h.Len() - 1
	h.Swap(0, n)
	h.down(0, n)
	x := h.Candidates[n]
	h.Candidates = h.Candidates[0:n]
	return x
}

func (h *CandidateHeap) ReplaceTop(x model.Candidate) {
	h.Candidates[0] = x
	h.down(0, h.Len())
}

func (h *CandidateHeap) up(j int) {
	for {
		i := (j - 1) / 2 // parent
		if i == j || !h.Less(j, i) {
			break
		}
		h.Swap(i, j)
		j = i
	}
}

func (h *CandidateHeap) down(i0, n int) {
	i := i0
	for {
		j1 := 2*i + 1
		if j1 >= n || j1 < 0 { // j1 < 0 after int overflow
			break
		}
		j := j1 // left child
		if j2 := j1 + 1; j2 < n && h.Less(j2, j1) {
			j = j2 // = 2*i + 2  // right child
		}
		if !h.Less(j, i) {
			break
		}
		h.Swap(i, j)
		i = j
	}
}

// Candidates returns the underlying slice.
func (h *CandidateHeap) GetCandidates() []model.Candidate {
	return h.Candidates
}
