package index

import (
	"container/heap"
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
	// Filter out empty lists
	activeLists := make([][]SearchResult, 0, len(lists))
	for _, l := range lists {
		if len(l) > 0 {
			activeLists = append(activeLists, l)
		}
	}

	if len(activeLists) == 0 {
		return nil
	}
	if len(activeLists) == 1 {
		if len(activeLists[0]) > k {
			return activeLists[0][:k]
		}
		return activeLists[0]
	}
	if len(activeLists) == 2 {
		return MergeSearchResults(activeLists[0], activeLists[1], k)
	}

	// Use a min-heap for N-way merge
	h := &mergeHeap{}
	heap.Init(h)

	// Initialize heap with first element from each list
	for i, list := range activeLists {
		heap.Push(h, mergeItem{
			res:     list[0],
			listIdx: i,
			elemIdx: 0,
		})
	}

	result := make([]SearchResult, 0, k)
	for h.Len() > 0 && len(result) < k {
		item := heap.Pop(h).(mergeItem)
		result = append(result, item.res)

		// Push next element from the same list
		if item.elemIdx+1 < len(activeLists[item.listIdx]) {
			heap.Push(h, mergeItem{
				res:     activeLists[item.listIdx][item.elemIdx+1],
				listIdx: item.listIdx,
				elemIdx: item.elemIdx + 1,
			})
		}
	}

	return result
}

type mergeItem struct {
	res     SearchResult
	listIdx int
	elemIdx int
}

type mergeHeap []mergeItem

func (h mergeHeap) Len() int           { return len(h) }
func (h mergeHeap) Less(i, j int) bool { return h[i].res.Distance < h[j].res.Distance }
func (h mergeHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *mergeHeap) Push(x any) {
	*h = append(*h, x.(mergeItem))
}

func (h *mergeHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
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
