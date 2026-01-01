package index

import (
	"iter"
)

// MergeSearchResults merges two sorted lists of SearchResult into a single sorted list of size k.
// Both input lists must be sorted by distance (ascending).
func MergeSearchResults(a, b []SearchResult, k int) []SearchResult {
	if len(a) == 0 {
		if len(b) > k {
			return b[:k]
		}
		return b
	}
	if len(b) == 0 {
		if len(a) > k {
			return a[:k]
		}
		return a
	}

	// Use a min-heap if k is large, or simple merge if k is small.
	// Since k is typically small (10-100), a simple merge is efficient.

	result := make([]SearchResult, 0, k)
	i, j := 0, 0

	for len(result) < k && (i < len(a) || j < len(b)) {
		if i < len(a) && j < len(b) {
			if a[i].Distance < b[j].Distance {
				result = append(result, a[i])
				i++
			} else {
				result = append(result, b[j])
				j++
			}
		} else if i < len(a) {
			result = append(result, a[i])
			i++
		} else {
			result = append(result, b[j])
			j++
		}
	}

	return result
}

// MergeNSearchResults merges multiple sorted lists of SearchResult into a single sorted list of size k.
// All input lists must be sorted by distance (ascending).
func MergeNSearchResults(k int, lists ...[]SearchResult) []SearchResult {
	res := make([]SearchResult, 0, k)
	MergeNSearchResultsInto(&res, k, lists...)
	return res
}

// MergeNSearchResultsInto merges multiple sorted lists of SearchResult into the provided buffer.
// The buffer is cleared before merging.
func MergeNSearchResultsInto(dst *[]SearchResult, k int, lists ...[]SearchResult) {
	*dst = (*dst)[:0]

	// Filter out empty lists
	// We use a small fixed-size array to avoid allocation for common cases (up to 8 lists)
	var activeListsBuf [8][]SearchResult
	var activeLists [][]SearchResult
	if len(lists) <= 8 {
		activeLists = activeListsBuf[:0]
	} else {
		activeLists = make([][]SearchResult, 0, len(lists))
	}

	for _, l := range lists {
		if len(l) > 0 {
			activeLists = append(activeLists, l)
		}
	}

	if len(activeLists) == 0 {
		return
	}
	if len(activeLists) == 1 {
		l := activeLists[0]
		if len(l) > k {
			l = l[:k]
		}
		*dst = append(*dst, l...)
		return
	}

	// For 2 lists, use optimized merge
	if len(activeLists) == 2 {
		mergeSearchResultsInto(dst, activeLists[0], activeLists[1], k)
		return
	}

	// Use a specialized min-heap for N-way merge to avoid interface boxing
	// We reuse a small buffer for the heap if possible, but since it's small (N lists),
	// a slice allocation is acceptable or we could use a pool.
	// Given N is usually small (shards), this is fine.
	h := make(mergeHeap, 0, len(activeLists))

	// Initialize heap with first element from each list
	for i, list := range activeLists {
		h = append(h, mergeItem{
			res:     list[0],
			listIdx: i,
			elemIdx: 0,
		})
	}
	h.init()

	for len(h) > 0 && len(*dst) < k {
		// Peek root
		root := h[0]
		*dst = append(*dst, root.res)

		// Get next element from the same list
		nextIdx := root.elemIdx + 1
		if nextIdx < len(activeLists[root.listIdx]) {
			// Optimization: Replace root with next element and fix down
			// This avoids a pop (swap+resize) followed by a push (append+up)
			h[0] = mergeItem{
				res:     activeLists[root.listIdx][nextIdx],
				listIdx: root.listIdx,
				elemIdx: nextIdx,
			}
			h.down(0, len(h))
		} else {
			// List exhausted, remove root
			h.pop()
		}
	}
}

func mergeSearchResultsInto(dst *[]SearchResult, a, b []SearchResult, k int) {
	i, j := 0, 0
	for len(*dst) < k && (i < len(a) || j < len(b)) {
		if i < len(a) && j < len(b) {
			if a[i].Distance < b[j].Distance {
				*dst = append(*dst, a[i])
				i++
			} else {
				*dst = append(*dst, b[j])
				j++
			}
		} else if i < len(a) {
			*dst = append(*dst, a[i])
			i++
		} else {
			*dst = append(*dst, b[j])
			j++
		}
	}
}

type mergeItem struct {
	res     SearchResult
	listIdx int
	elemIdx int
}

// mergeHeap is a specialized min-heap for mergeItem to avoid interface boxing.
type mergeHeap []mergeItem

func (h mergeHeap) less(i, j int) bool { return h[i].res.Distance < h[j].res.Distance }
func (h mergeHeap) swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h mergeHeap) init() {
	n := len(h)
	for i := n/2 - 1; i >= 0; i-- {
		h.down(i, n)
	}
}

func (h *mergeHeap) pop() {
	n := len(*h) - 1
	(*h)[0] = (*h)[n]
	*h = (*h)[:n]
	if n > 0 {
		(*h).down(0, n)
	}
}

func (h mergeHeap) down(i0, n int) {
	i := i0
	for {
		j1 := 2*i + 1
		if j1 >= n || j1 < 0 { // j1 < 0 after int overflow
			break
		}
		j := j1 // left child
		if j2 := j1 + 1; j2 < n && h.less(j2, j1) {
			j = j2 // = 2*i + 2  // right child
		}
		if !h.less(j, i) {
			break
		}
		h.swap(i, j)
		i = j
	}
}

// SliceToStream converts a slice of SearchResult to an iterator.
func SliceToStream(results []SearchResult) iter.Seq2[SearchResult, error] {
	return func(yield func(SearchResult, error) bool) {
		for _, res := range results {
			if !yield(res, nil) {
				return
			}
		}
	}
}

// MergeSearchStreams merges two sorted streams of SearchResult.
func MergeSearchStreams(seq1, seq2 iter.Seq2[SearchResult, error]) iter.Seq2[SearchResult, error] {
	return func(yield func(SearchResult, error) bool) {
		next1, stop1 := iter.Pull2(seq1)
		defer stop1()
		next2, stop2 := iter.Pull2(seq2)
		defer stop2()

		v1, err1, ok1 := next1()
		v2, err2, ok2 := next2()

		for ok1 || ok2 {
			if err1 != nil {
				yield(SearchResult{}, err1)
				return
			}
			if err2 != nil {
				yield(SearchResult{}, err2)
				return
			}

			if !ok1 {
				if !yield(v2, nil) {
					return
				}
				v2, err2, ok2 = next2()
				continue
			}
			if !ok2 {
				if !yield(v1, nil) {
					return
				}
				v1, err1, ok1 = next1()
				continue
			}

			if v1.Distance < v2.Distance {
				if !yield(v1, nil) {
					return
				}
				v1, err1, ok1 = next1()
			} else {
				if !yield(v2, nil) {
					return
				}
				v2, err2, ok2 = next2()
			}
		}
	}
}
