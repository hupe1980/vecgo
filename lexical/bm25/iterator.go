package bm25

// termIterator allows iterating over a posting list for a specific term.
type termIterator struct {
	postings []posting
	idx      int
	idf      float64
}

// doc returns the current docID. Returns max uint32 if exhausted.
func (it *termIterator) doc() uint32 {
	if it.idx >= len(it.postings) {
		return ^uint32(0) // Max uint32
	}
	return it.postings[it.idx].docID
}

// count returns the term frequency in the current document.
func (it *termIterator) count() uint32 {
	if it.idx >= len(it.postings) {
		return 0
	}
	return it.postings[it.idx].count
}

// next advances to the next posting.
func (it *termIterator) next() {
	it.idx++
}
