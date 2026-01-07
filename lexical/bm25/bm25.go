package bm25

import (
	"math"
	"strings"
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

// MemoryIndex is a simple in-memory BM25 index.
type MemoryIndex struct {
	mu sync.RWMutex

	// Mappings
	pkToDocID map[model.PK]uint32
	docIDToPK []model.PK
	freeIDs   []uint32 // Reusable IDs from deleted docs

	// Dense Data
	docLengths []uint32 // Indexed by docID

	// Inverted Index
	inverted map[string][]posting

	totalLength int64
	docCount    int

	scorePool    sync.Pool
	iteratorPool sync.Pool
	heapPool     sync.Pool
}

// New creates a new MemoryIndex.
func New() *MemoryIndex {
	return &MemoryIndex{
		pkToDocID:  make(map[model.PK]uint32),
		docIDToPK:  make([]model.PK, 0),
		docLengths: make([]uint32, 0),
		inverted:   make(map[string][]posting),
		scorePool: sync.Pool{
			New: func() interface{} {
				return make([]float32, 1024)
			},
		},
		iteratorPool: sync.Pool{
			New: func() any {
				return make([]termIterator, 0, 8)
			},
		},
		heapPool: sync.Pool{
			New: func() any {
				h := make(candidateHeap, 0, 16)
				return &h
			},
		},
	}
}

// Ensure MemoryIndex implements lexical.Index
var _ lexical.Index = (*MemoryIndex)(nil)

func (idx *MemoryIndex) forEachToken(text string, fn func(token string)) {
	start := -1
	for i, r := range text {
		if unicode.IsSpace(r) {
			if start >= 0 {
				fn(strings.ToLower(text[start:i]))
				start = -1
			}
		} else {
			if start < 0 {
				start = i
			}
		}
	}
	if start >= 0 {
		fn(strings.ToLower(text[start:]))
	}
}

func (idx *MemoryIndex) Add(pk model.PK, text string) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	// If exists, delete first
	if _, ok := idx.pkToDocID[pk]; ok {
		idx.deleteLocked(pk)
	}

	// Allocate DocID
	var docID uint32
	if len(idx.freeIDs) > 0 {
		docID = idx.freeIDs[len(idx.freeIDs)-1]
		idx.freeIDs = idx.freeIDs[:len(idx.freeIDs)-1]
		// Update mappings
		idx.pkToDocID[pk] = docID
		idx.docIDToPK[docID] = pk
	} else {
		docID = uint32(len(idx.docIDToPK))
		idx.pkToDocID[pk] = docID
		idx.docIDToPK = append(idx.docIDToPK, pk)
		idx.docLengths = append(idx.docLengths, 0)
	}

	length := 0
	tf := make(map[string]int)
	idx.forEachToken(text, func(t string) {
		length++
		tf[t]++
	})

	idx.docLengths[docID] = uint32(length)
	idx.totalLength += int64(length)
	idx.docCount++

	for t, count := range tf {
		idx.inverted[t] = append(idx.inverted[t], posting{docID: docID, count: uint32(count)})
	}

	return nil
}

func (idx *MemoryIndex) Delete(pk model.PK) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	return idx.deleteLocked(pk)
}

func (idx *MemoryIndex) deleteLocked(pk model.PK) error {
	docID, ok := idx.pkToDocID[pk]
	if !ok {
		return nil
	}

	length := idx.docLengths[docID]

	// Remove from inverted index (Slow!)
	for t := range idx.inverted {
		postings := idx.inverted[t]
		for i, p := range postings {
			if p.docID == docID {
				// Remove
				idx.inverted[t] = append(postings[:i], postings[i+1:]...)
				break
			}
		}
	}

	// Update stats
	idx.totalLength -= int64(length)
	idx.docCount--

	// Clear dense data
	idx.docLengths[docID] = 0
	idx.docIDToPK[docID] = model.PK{} // Sentinel for deleted

	// Remove mapping
	delete(idx.pkToDocID, pk)

	// Add to free list
	idx.freeIDs = append(idx.freeIDs, docID)

	return nil
}

func (idx *MemoryIndex) Search(text string, k int) ([]model.Candidate, error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if idx.docCount == 0 {
		return nil, nil
	}

	// Get score buffer
	maxDocID := len(idx.docIDToPK)
	scoreBuf := idx.scorePool.Get().([]float32)
	if cap(scoreBuf) < maxDocID {
		scoreBuf = make([]float32, maxDocID*2)
	} else {
		scoreBuf = scoreBuf[:maxDocID]
		// Zero out used range
		for i := range scoreBuf {
			scoreBuf[i] = 0
		}
	}
	defer idx.scorePool.Put(scoreBuf)

	avgDL := float64(idx.totalLength) / float64(idx.docCount)

	// Precompute BM25 constants for this query
	k1_plus_1 := k1 + 1
	k1_1b := k1 * (1 - b)
	k1_b_avgDL := k1 * b / avgDL

	// Accumulate scores
	idx.forEachToken(text, func(t string) {
		postings, ok := idx.inverted[t]
		if !ok {
			return
		}

		// IDF
		df := len(postings)
		idf := idx.computeIDF(df)

		for _, p := range postings {
			tf := float64(p.count)
			docLen := float64(idx.docLengths[p.docID])

			// BM25 formula
			num := tf * k1_plus_1
			denom := tf + k1_1b + k1_b_avgDL*docLen
			score := idf * (num / denom)

			scoreBuf[p.docID] += float32(score)
		}
	})

	// Use a min-heap to keep top k
	h := &candidateHeap{}

	// Iterate over scoreBuf to find top K
	for docID, score := range scoreBuf {
		if score <= 0 {
			continue
		}

		pk := idx.docIDToPK[docID]
		// Check if deleted (pk == 0 check if 0 is invalid PK, but 0 is valid PK)
		// However, if we deleted it, we removed it from inverted index, so score should be 0!
		// So score > 0 implies valid docID.

		if len(*h) < k {
			h.push(model.Candidate{PK: pk, Score: score})
		} else if score > (*h)[0].Score {
			(*h)[0] = model.Candidate{PK: pk, Score: score}
			h.down(0, len(*h))
		}
	}

	// Convert heap to slice (sorted descending)
	candidates := make([]model.Candidate, len(*h))
	for i := len(*h) - 1; i >= 0; i-- {
		candidates[i] = h.pop()
	}

	return candidates, nil
}

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
	h.down(0, n-1)
	return root
}

func (h *candidateHeap) up(j int) {
	for {
		i := (j - 1) / 2 // parent
		if i == j || !((*h)[j].Score < (*h)[i].Score) {
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
		if j1 >= n || j1 < 0 { // j1 < 0 after int overflow
			break
		}
		j := j1 // left child
		if j2 := j1 + 1; j2 < n && (*h)[j2].Score < (*h)[j1].Score {
			j = j2 // = 2*i + 2  // right child
		}
		if !((*h)[j].Score < (*h)[i].Score) {
			break
		}
		(*h)[i], (*h)[j] = (*h)[j], (*h)[i]
		i = j
	}
}

func (idx *MemoryIndex) computeIDF(df int) float64 {
	// IDF = log(1 + (N - n + 0.5) / (n + 0.5))
	N := float64(idx.docCount)
	n := float64(df)
	return math.Log(1 + (N-n+0.5)/(n+0.5))
}

func (idx *MemoryIndex) Close() error {
	return nil
}
