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
		err := idx.Add(model.PKUint64(d.id), d.text)
		assert.NoError(t, err)
	}

	// Search
	results, err := idx.Search("fox", 10)
	assert.NoError(t, err)
	assert.NotEmpty(t, results)
	// Expect doc 1 and 4
	found := make(map[uint64]bool)
	for _, r := range results {
		pk, ok := r.PK.Uint64()
		if !ok {
			continue
		}
		found[pk] = true
		fmt.Printf("Doc %d Score: %f\n", pk, r.Score)
	}
	assert.True(t, found[1])
	assert.True(t, found[4])
}

func TestMemoryIndex_Delete(t *testing.T) {
	idx := New()
	idx.Add(model.PKUint64(1), "test content")
	idx.Add(model.PKUint64(2), "other content")

	res, _ := idx.Search("test", 10)
	assert.Len(t, res, 1)

	// Delete
	idx.Delete(model.PKUint64(1))

	res, _ = idx.Search("test", 10)
	assert.Len(t, res, 0)

	// Add back
	idx.Add(model.PKUint64(1), "test content again")
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
	idx.Add(model.PKUint64(1), b.String())

	res, _ := idx.Search("word", 1)
	assert.Len(t, res, 1)
	assert.True(t, res[0].Score > 0)
}
