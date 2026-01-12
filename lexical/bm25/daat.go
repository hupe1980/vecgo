package bm25

import (
	"github.com/hupe1980/vecgo/model"
)

// SearchDAAT performs a Document-At-A-Time search.
// This avoids allocating a large score map and improves cache locality.
func (idx *MemoryIndex) SearchDAAT(text string, k int) ([]model.Candidate, error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if idx.docCount == 0 {
		return nil, nil
	}

	// 1. Create Iterators
	iterators := idx.iteratorPool.Get().([]termIterator)
	iterators = iterators[:0]
	defer func() {
		idx.iteratorPool.Put(iterators)
	}()

	idx.forEachToken(text, func(t string) {
		postings, ok := idx.inverted[t]
		if !ok || len(postings) == 0 {
			return
		}
		// Compute IDF once per term
		df := len(postings)
		idf := idx.computeIDF(df)
		iterators = append(iterators, termIterator{postings: postings, idx: 0, idf: idf})
	})

	if len(iterators) == 0 {
		return nil, nil
	}

	avgDL := float64(idx.totalLength) / float64(idx.docCount)

	// Precompute BM25 constants for this query
	k1_plus_1 := k1 + 1
	k1_1b := k1 * (1 - b)
	k1_b_avgDL := k1 * b / avgDL

	h := idx.heapPool.Get().(*candidateHeap)
	*h = (*h)[:0]
	defer idx.heapPool.Put(h)

	// 2. DAAT Loop
	// We need to find the minimum docID across all iterators.
	// A min-heap of iterators could be used, but for short queries, linear scan is fine.

	for {
		// Find min docID
		minDoc := ^uint32(0)
		for i := range iterators {
			doc := iterators[i].doc()
			if doc < minDoc {
				minDoc = doc
			}
		}

		// If minDoc is max uint32, we are done
		if minDoc == ^uint32(0) {
			break
		}

		// Score document
		var score float64
		docLen := float64(idx.docLengths[minDoc])

		for i := range iterators {
			it := &iterators[i]
			if it.doc() == minDoc {
				tf := float64(it.count())

				// BM25 formula
				num := tf * k1_plus_1
				denom := tf + k1_1b + k1_b_avgDL*docLen
				score += it.idf * (num / denom)

				it.next()
			}
		}

		// Update Top-K Heap
		if score > 0 {
			pk := idx.docIDToPK[minDoc]
			if len(*h) < k {
				h.push(model.Candidate{ID: pk, Score: float32(score)})
			} else if float32(score) > (*h)[0].Score {
				(*h)[0] = model.Candidate{ID: pk, Score: float32(score)}
				h.down(0, len(*h))
			}
		}
	}

	// 3. Convert to Result Slice
	candidates := make([]model.Candidate, len(*h))
	for i := len(*h) - 1; i >= 0; i-- {
		candidates[i] = h.pop()
	}

	return candidates, nil
}
