package searcher

// CandidateHeap is a heap of InternalCandidate.
// It is used for collecting top-k results.
type CandidateHeap struct {
	Candidates []InternalCandidate
	descending bool // true if we want largest scores (Dot/Cosine), false for smallest (L2)
}

// InternalCandidateBetter reports whether a is better than b under the metric direction.
// Tie-breaker is (SegmentID, RowID) ascending for determinism.
func InternalCandidateBetter(a, b InternalCandidate, descending bool) bool {
	if a.Score != b.Score {
		if descending {
			return a.Score > b.Score
		}
		return a.Score < b.Score
	}
	if a.SegmentID != b.SegmentID {
		return a.SegmentID < b.SegmentID
	}
	return a.RowID < b.RowID
}

// InternalCandidateWorse reports whether a is worse than b under the metric direction.
// Tie-breaker is (SegmentID, RowID) descending for determinism (so the heap evicts larger locations first).
func InternalCandidateWorse(a, b InternalCandidate, descending bool) bool {
	if a.Score != b.Score {
		if descending {
			return a.Score < b.Score
		}
		return a.Score > b.Score
	}
	if a.SegmentID != b.SegmentID {
		return a.SegmentID > b.SegmentID
	}
	return a.RowID > b.RowID
}

// NewCandidateHeap creates a new CandidateHeap.
func NewCandidateHeap(capacity int, descending bool) *CandidateHeap {
	return &CandidateHeap{
		Candidates: make([]InternalCandidate, 0, capacity),
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
	return InternalCandidateWorse(h.Candidates[i], h.Candidates[j], h.descending)
}

func (h *CandidateHeap) Push(x InternalCandidate) {
	h.Candidates = append(h.Candidates, x)
	h.up(h.Len() - 1)
}

func (h *CandidateHeap) Pop() InternalCandidate {
	n := h.Len() - 1
	h.Swap(0, n)
	h.down(0, n)
	x := h.Candidates[n]
	h.Candidates = h.Candidates[0:n]
	return x
}

// Peek returns the top element without removing it.
// Panics if the heap is empty - caller should check Len() > 0.
func (h *CandidateHeap) Peek() InternalCandidate {
	return h.Candidates[0]
}

// TryPeek returns the top element and true, or zero value and false if empty.
func (h *CandidateHeap) TryPeek() (InternalCandidate, bool) {
	if h.Len() == 0 {
		return InternalCandidate{}, false
	}
	return h.Candidates[0], true
}

// ReplaceTop replaces the top element and restores heap invariant.
// Panics if the heap is empty - caller should check Len() > 0.
func (h *CandidateHeap) ReplaceTop(x InternalCandidate) {
	h.Candidates[0] = x
	h.down(0, h.Len())
}

// TryReplaceTop replaces the top element if heap is non-empty.
// Returns true if replacement occurred, false if heap was empty.
func (h *CandidateHeap) TryReplaceTop(x InternalCandidate) bool {
	if h.Len() == 0 {
		return false
	}
	h.Candidates[0] = x
	h.down(0, h.Len())
	return true
}

// up moves element at j up the heap. Optimized with inline comparison and single final write.
// 4-ary heap: parent = (j-1)/4 instead of (j-1)/2
func (h *CandidateHeap) up(j int) {
	item := h.Candidates[j]
	for j > 0 {
		i := (j - 1) / heapArity // parent in 4-ary heap
		if !InternalCandidateWorse(item, h.Candidates[i], h.descending) {
			break
		}
		h.Candidates[j] = h.Candidates[i]
		j = i
	}
	h.Candidates[j] = item
}

// down moves element at i0 down the heap. Optimized with inline moves and single final write.
// 4-ary heap: first child = 4*i+1, up to 4 children to compare.
func (h *CandidateHeap) down(i0, n int) {
	i := i0
	item := h.Candidates[i]
	for {
		firstChild := heapArity*i + 1
		if firstChild >= n {
			break
		}

		// Find the best child among up to 4 children
		best := firstChild
		lastChild := firstChild + heapArity
		if lastChild > n {
			lastChild = n
		}
		for c := firstChild + 1; c < lastChild; c++ {
			if InternalCandidateWorse(h.Candidates[c], h.Candidates[best], h.descending) {
				best = c
			}
		}

		if !InternalCandidateWorse(h.Candidates[best], item, h.descending) {
			break
		}
		h.Candidates[i] = h.Candidates[best]
		i = best
	}
	h.Candidates[i] = item
}

// Candidates returns the underlying slice.
func (h *CandidateHeap) GetCandidates() []InternalCandidate {
	return h.Candidates
}
