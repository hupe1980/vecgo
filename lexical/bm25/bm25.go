package bm25

import (
	"context"
	"math"
	"sync"
	"unicode"

	"github.com/hupe1980/vecgo/lexical"
	"github.com/hupe1980/vecgo/model"
)

const (
	k1 = 1.2
	b  = 0.75
)

type posting struct {
	docID uint32
	count uint32
}

// Pointer wrapper types for sync.Pool to avoid allocation on Put (SA6002).
type tokenBuf struct{ buf []byte }
type termIterators struct{ iters []termIterator }
type tfMap struct{ m map[string]int }

// MemoryIndex is a simple in-memory BM25 index using DAAT (Document-At-A-Time) scoring.
type MemoryIndex struct {
	mu sync.RWMutex

	// Mappings
	pkToDocID map[model.ID]uint32
	docIDToPK []model.ID
	freeIDs   []uint32 // Reusable IDs from deleted docs

	// Dense Data
	docLengths []uint32 // Indexed by docID

	// Inverted Index
	inverted map[string][]posting

	// Reverse index: docID -> terms (for efficient delete)
	docTerms [][]string

	totalLength int64
	docCount    int

	iteratorPool sync.Pool
	heapPool     sync.Pool
	tokenBufPool sync.Pool
	tfPool       sync.Pool // Pool for term-frequency maps
}

// New creates a new MemoryIndex.
func New() *MemoryIndex {
	return &MemoryIndex{
		pkToDocID:  make(map[model.ID]uint32),
		docIDToPK:  make([]model.ID, 0),
		docLengths: make([]uint32, 0),
		docTerms:   make([][]string, 0),
		inverted:   make(map[string][]posting),
		iteratorPool: sync.Pool{
			New: func() any {
				return &termIterators{iters: make([]termIterator, 0, 8)}
			},
		},
		heapPool: sync.Pool{
			New: func() any {
				h := make(candidateHeap, 0, 64)
				return &h
			},
		},
		tokenBufPool: sync.Pool{
			New: func() any {
				return &tokenBuf{buf: make([]byte, 0, 64)}
			},
		},
		tfPool: sync.Pool{
			New: func() any {
				return &tfMap{m: make(map[string]int, 32)}
			},
		},
	}
}

// Ensure MemoryIndex implements lexical.Index
var _ lexical.Index = (*MemoryIndex)(nil)

// forEachToken iterates over whitespace-separated tokens in text.
// Uses in-place lowercasing to avoid allocation.
func (idx *MemoryIndex) forEachToken(text string, fn func(token string)) {
	wrap := idx.tokenBufPool.Get().(*tokenBuf)
	buf := wrap.buf[:0]
	defer func() { wrap.buf = buf; idx.tokenBufPool.Put(wrap) }()

	start := -1
	for i := 0; i < len(text); i++ {
		c := text[i]
		if c <= ' ' { // Fast ASCII whitespace check
			if start >= 0 {
				// Emit token
				buf = buf[:0]
				for j := start; j < i; j++ {
					ch := text[j]
					if ch >= 'A' && ch <= 'Z' {
						ch += 32 // ASCII lowercase
					}
					buf = append(buf, ch)
				}
				fn(string(buf))
				start = -1
			}
		} else if start < 0 {
			start = i
		}
	}
	if start >= 0 {
		buf = buf[:0]
		for j := start; j < len(text); j++ {
			ch := text[j]
			if ch >= 'A' && ch <= 'Z' {
				ch += 32
			}
			buf = append(buf, ch)
		}
		fn(string(buf))
	}
}

// forEachTokenUnicode handles non-ASCII text properly.
func (idx *MemoryIndex) forEachTokenUnicode(text string, fn func(token string)) {
	wrap := idx.tokenBufPool.Get().(*tokenBuf)
	buf := wrap.buf[:0]
	defer func() { wrap.buf = buf; idx.tokenBufPool.Put(wrap) }()

	start := -1
	for i, r := range text {
		if unicode.IsSpace(r) {
			if start >= 0 {
				buf = buf[:0]
				for _, c := range text[start:i] {
					buf = append(buf, string(unicode.ToLower(c))...)
				}
				fn(string(buf))
				start = -1
			}
		} else if start < 0 {
			start = i
		}
	}
	if start >= 0 {
		buf = buf[:0]
		for _, c := range text[start:] {
			buf = append(buf, string(unicode.ToLower(c))...)
		}
		fn(string(buf))
	}
}

// isASCII returns true if text contains only ASCII characters.
func isASCII(text string) bool {
	for i := 0; i < len(text); i++ {
		if text[i] >= 128 {
			return false
		}
	}
	return true
}

// tokenize calls fn for each token, choosing fast ASCII or Unicode path.
func (idx *MemoryIndex) tokenize(text string, fn func(token string)) {
	if isASCII(text) {
		idx.forEachToken(text, fn)
	} else {
		idx.forEachTokenUnicode(text, fn)
	}
}

func (idx *MemoryIndex) Add(id model.ID, text string) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	// If exists, delete first
	if _, ok := idx.pkToDocID[id]; ok {
		idx.deleteLocked(id)
	}

	// Allocate DocID
	var docID uint32
	if len(idx.freeIDs) > 0 {
		docID = idx.freeIDs[len(idx.freeIDs)-1]
		idx.freeIDs = idx.freeIDs[:len(idx.freeIDs)-1]
		idx.pkToDocID[id] = docID
		idx.docIDToPK[docID] = id
	} else {
		docID = uint32(len(idx.docIDToPK))
		idx.pkToDocID[id] = docID
		idx.docIDToPK = append(idx.docIDToPK, id)
		idx.docLengths = append(idx.docLengths, 0)
		idx.docTerms = append(idx.docTerms, nil)
	}

	// Use pooled tf map to avoid allocation per Add call
	tfWrap := idx.tfPool.Get().(*tfMap)
	tf := tfWrap.m
	clear(tf) // Reset for reuse
	defer func() { idx.tfPool.Put(tfWrap) }()

	length := 0
	idx.tokenize(text, func(t string) {
		length++
		tf[t]++
	})

	idx.docLengths[docID] = uint32(length)
	idx.totalLength += int64(length)
	idx.docCount++

	// Store terms for efficient delete
	terms := make([]string, 0, len(tf))
	for t, count := range tf {
		terms = append(terms, t)
		idx.inverted[t] = append(idx.inverted[t], posting{docID: docID, count: uint32(count)})
	}
	idx.docTerms[docID] = terms

	return nil
}

func (idx *MemoryIndex) Delete(id model.ID) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	idx.deleteLocked(id)
	return nil
}

func (idx *MemoryIndex) deleteLocked(id model.ID) {
	docID, ok := idx.pkToDocID[id]
	if !ok {
		return
	}

	length := idx.docLengths[docID]
	terms := idx.docTerms[docID]

	// Remove from inverted index - only scan terms this doc has
	for _, t := range terms {
		postings := idx.inverted[t]
		for i, p := range postings {
			if p.docID == docID {
				// Swap-remove for O(1) removal
				postings[i] = postings[len(postings)-1]
				idx.inverted[t] = postings[:len(postings)-1]
				break
			}
		}
		// Clean up empty posting lists
		if len(idx.inverted[t]) == 0 {
			delete(idx.inverted, t)
		}
	}

	// Update stats
	idx.totalLength -= int64(length)
	idx.docCount--

	// Clear dense data
	idx.docLengths[docID] = 0
	idx.docIDToPK[docID] = 0
	idx.docTerms[docID] = nil

	// Remove mapping
	delete(idx.pkToDocID, id)

	// Add to free list
	idx.freeIDs = append(idx.freeIDs, docID)
}

// Search performs a keyword search using DAAT scoring.
// Implements lexical.Index interface.
func (idx *MemoryIndex) Search(ctx context.Context, text string, k int) ([]model.Candidate, error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if idx.docCount == 0 {
		return nil, nil
	}

	// Check context before starting
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	// 1. Create Iterators
	wrap := idx.iteratorPool.Get().(*termIterators)
	iterators := wrap.iters[:0]
	defer func() { wrap.iters = iterators; idx.iteratorPool.Put(wrap) }()

	idx.tokenize(text, func(t string) {
		postings, ok := idx.inverted[t]
		if !ok || len(postings) == 0 {
			return
		}
		df := len(postings)
		idf := idx.computeIDF(df)
		iterators = append(iterators, termIterator{postings: postings, idx: 0, idf: idf})
	})

	if len(iterators) == 0 {
		return nil, nil
	}

	avgDL := float64(idx.totalLength) / float64(idx.docCount)
	k1_plus_1 := k1 + 1
	k1_1b := k1 * (1 - b)
	k1_b_avgDL := k1 * b / avgDL

	h := idx.heapPool.Get().(*candidateHeap)
	*h = (*h)[:0]
	defer idx.heapPool.Put(h)

	// 2. DAAT Loop
	iterations := 0
	for {
		// Periodic context check
		iterations++
		if iterations&0xFF == 0 { // Every 256 iterations
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}

		// Find min docID
		minDoc := ^uint32(0)
		for i := range iterators {
			doc := iterators[i].doc()
			if doc < minDoc {
				minDoc = doc
			}
		}

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

	// 3. Convert to Result Slice (sorted descending)
	candidates := make([]model.Candidate, len(*h))
	for i := len(*h) - 1; i >= 0; i-- {
		candidates[i] = h.pop()
	}

	return candidates, nil
}

func (idx *MemoryIndex) computeIDF(df int) float64 {
	N := float64(idx.docCount)
	n := float64(df)
	return math.Log(1 + (N-n+0.5)/(n+0.5))
}

func (idx *MemoryIndex) Close() error {
	return nil
}

// candidateHeap is a min-heap of candidates by score.
type candidateHeap []model.Candidate

func (h *candidateHeap) push(x model.Candidate) {
	*h = append(*h, x)
	h.up(len(*h) - 1)
}

func (h *candidateHeap) pop() model.Candidate {
	old := *h
	n := len(old)
	root := old[0]
	(*h)[0] = old[n-1]
	*h = old[0 : n-1]
	if len(*h) > 0 {
		h.down(0, len(*h))
	}
	return root
}

func (h *candidateHeap) up(j int) {
	for {
		i := (j - 1) / 2
		if i == j || (*h)[j].Score >= (*h)[i].Score {
			break
		}
		(*h)[j], (*h)[i] = (*h)[i], (*h)[j]
		j = i
	}
}

func (h *candidateHeap) down(i0, n int) {
	i := i0
	for {
		j1 := 2*i + 1
		if j1 >= n || j1 < 0 {
			break
		}
		j := j1
		if j2 := j1 + 1; j2 < n && (*h)[j2].Score < (*h)[j1].Score {
			j = j2
		}
		if (*h)[j].Score >= (*h)[i].Score {
			break
		}
		(*h)[i], (*h)[j] = (*h)[j], (*h)[i]
		i = j
	}
}
