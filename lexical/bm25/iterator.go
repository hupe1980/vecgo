package bm25

// termIterator allows iterating over a posting list for a specific term.
type termIterator struct {
	postings []posting
	idx      int
	idf      float64
}

// newTermIterator creates a new iterator for the given postings.
func newTermIterator(postings []posting, idf float64) *termIterator {
	return &termIterator{
		postings: postings,
		idx:      0,
		idf:      idf,
	}
}

// doc returns the current docID. Returns max uint32 if exhausted.
func (it *termIterator) doc() uint32 {
	if it.idx >= len(it.postings) {
		return ^uint32(0) // Max uint32
	}
	return it.postings[it.idx].docID
}

// count returns the term frequency in the current document.
func (it *termIterator) count() int {
	if it.idx >= len(it.postings) {
		return 0
	}
	return it.postings[it.idx].count
}

// next advances to the next posting.
func (it *termIterator) next() {
	it.idx++
}

// advance moves to the first posting with docID >= target.
func (it *termIterator) advance(target uint32) {
	// Simple linear scan for now.
	// Optimization: Use binary search or skipping if postings are large.
	for it.doc() < target {
		it.next()
	}
}
