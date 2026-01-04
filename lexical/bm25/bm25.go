package bm25

import (
	"math"
	"strings"
	"sync"

	"github.com/hupe1980/vecgo/lexical"
	"github.com/hupe1980/vecgo/model"
)

const (
	k1 = 1.2
	b  = 0.75
)

type posting struct {
	pk    model.PrimaryKey
	count int
}

// MemoryIndex is a simple in-memory BM25 index.
type MemoryIndex struct {
	mu          sync.RWMutex
	inverted    map[string][]posting
	docLengths  map[model.PrimaryKey]int
	totalLength int64
	docCount    int
}

// New creates a new MemoryIndex.
func New() *MemoryIndex {
	return &MemoryIndex{
		inverted:   make(map[string][]posting),
		docLengths: make(map[model.PrimaryKey]int),
	}
}

// Ensure MemoryIndex implements lexical.Index
var _ lexical.Index = (*MemoryIndex)(nil)

func (idx *MemoryIndex) tokenize(text string) []string {
	// Very simple tokenizer: lowercase and split by whitespace
	return strings.Fields(strings.ToLower(text))
}

func (idx *MemoryIndex) Add(pk model.PrimaryKey, text string) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	// If exists, delete first (naive update)
	if _, ok := idx.docLengths[pk]; ok {
		idx.deleteLocked(pk)
	}

	tokens := idx.tokenize(text)
	length := len(tokens)

	idx.docLengths[pk] = length
	idx.totalLength += int64(length)
	idx.docCount++

	// Count term frequencies
	tf := make(map[string]int)
	for _, t := range tokens {
		tf[t]++
	}

	for t, count := range tf {
		idx.inverted[t] = append(idx.inverted[t], posting{pk: pk, count: count})
	}

	return nil
}

func (idx *MemoryIndex) Delete(pk model.PrimaryKey) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	return idx.deleteLocked(pk)
}

func (idx *MemoryIndex) deleteLocked(pk model.PrimaryKey) error {
	length, ok := idx.docLengths[pk]
	if !ok {
		return nil
	}

	// This is slow (O(terms * docs)), but fine for a reference implementation.
	for t := range idx.inverted {
		postings := idx.inverted[t]
		for i, p := range postings {
			if p.pk == pk {
				// Remove
				idx.inverted[t] = append(postings[:i], postings[i+1:]...)
				break
			}
		}
	}

	delete(idx.docLengths, pk)
	idx.totalLength -= int64(length)
	idx.docCount--
	return nil
}

func (idx *MemoryIndex) Search(text string) (map[model.PrimaryKey]float32, error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	tokens := idx.tokenize(text)
	scores := make(map[model.PrimaryKey]float32)

	if idx.docCount == 0 {
		return scores, nil
	}

	avgDL := float64(idx.totalLength) / float64(idx.docCount)

	for _, t := range tokens {
		postings, ok := idx.inverted[t]
		if !ok {
			continue
		}

		// IDF
		df := len(postings)
		idf := idx.computeIDF(df)

		for _, p := range postings {
			tf := float64(p.count)
			docLen := float64(idx.docLengths[p.pk])

			// BM25 formula
			num := tf * (k1 + 1)
			denom := tf + k1*(1-b+b*(docLen/avgDL))
			score := idf * (num / denom)

			scores[p.pk] += float32(score)
		}
	}

	return scores, nil
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
