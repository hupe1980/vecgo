package bm25

import (
	"fmt"
	"strings"
	"testing"

	"github.com/hupe1980/vecgo/model"
	"github.com/stretchr/testify/assert"
)

func TestMemoryIndex_Basic(t *testing.T) {
	idx := New()
	assert.NotNil(t, idx)

	// Add docs
	docs := []struct {
		id   uint64
		text string
	}{
		{1, "the quick brown fox"},
		{2, "jumped over the lazy dog"},
		{3, "quick brown dogs"},
		{4, "fox and dog"},
	}

	for _, d := range docs {
		err := idx.Add(model.ID(d.id), d.text)
		assert.NoError(t, err)
	}

	// Search
	results, err := idx.Search("fox", 10)
	assert.NoError(t, err)
	assert.NotEmpty(t, results)
	// Expect doc 1 and 4
	found := make(map[uint64]bool)
	for _, r := range results {
		pk := uint64(r.ID)
		found[pk] = true
		fmt.Printf("Doc %d Score: %f\n", pk, r.Score)
	}
	assert.True(t, found[1])
	assert.True(t, found[4])
}

func TestMemoryIndex_Delete(t *testing.T) {
	idx := New()
	idx.Add(model.ID(1), "test content")
	idx.Add(model.ID(2), "other content")

	res, _ := idx.Search("test", 10)
	assert.Len(t, res, 1)

	// Delete
	idx.Delete(model.ID(1))

	res, _ = idx.Search("test", 10)
	assert.Len(t, res, 0)

	// Add back
	idx.Add(model.ID(1), "test content again")
	res, _ = idx.Search("test", 10)
	assert.Len(t, res, 1)
}

func TestMemoryIndex_TypeOverflow(t *testing.T) {
	// Verify that uint32 optimized counters work
	idx := New()

	// Create a document with many repetitions
	var b strings.Builder
	for i := 0; i < 300; i++ {
		b.WriteString("word ")
	}
	idx.Add(model.ID(1), b.String())

	res, _ := idx.Search("word", 1)
	assert.Len(t, res, 1)
	assert.True(t, res[0].Score > 0)
}

func TestMemoryIndex_DAAT(t *testing.T) {
	idx := New()
	idx.Add(model.ID(1), "hello world")
	idx.Add(model.ID(2), "hello go")
	idx.Add(model.ID(3), "world")

	res, err := idx.SearchDAAT("hello", 10) // Should match doc 1 and 2
	assert.NoError(t, err)
	assert.Len(t, res, 2)
}

func TestMemoryIndex_Close(t *testing.T) {
	idx := New()
	assert.NoError(t, idx.Close())
}

func TestHeap_Logic(t *testing.T) {
	// Indirectly test heap via Search with k smaller than results
	idx := New()
	for i := 0; i < 20; i++ {
		idx.Add(model.ID(uint64(i)), "term")
	}

	// k=5, matches 20 docs. Heap will be full and pop/push will happen repeatedly.
	res, err := idx.Search("term", 5)
	assert.NoError(t, err)
	assert.Len(t, res, 5)
}
