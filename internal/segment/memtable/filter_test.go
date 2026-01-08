package memtable

import (
	"testing"

	"github.com/hupe1980/vecgo/internal/segment"
	"github.com/hupe1980/vecgo/metadata"
	"github.com/stretchr/testify/assert"
)

func TestMetadataFilterWrapper(t *testing.T) {
	// Setup Docs
	docs := []metadata.InternedDocument{
		metadata.Intern(metadata.Document{"color": metadata.String("red"), "price": metadata.Int(10)}),   // 0
		metadata.Intern(metadata.Document{"color": metadata.String("blue"), "price": metadata.Int(20)}),  // 1
		metadata.Intern(metadata.Document{"color": metadata.String("green"), "price": metadata.Int(30)}), // 2
		nil, // 3 (deleted/null)
	}

	// 1. Basic Filter: color == "blue"
	f1 := metadata.Filter{
		Key:      "color",
		Operator: metadata.OpEqual,
		Value:    metadata.String("blue"),
	}
	fs1 := metadata.NewFilterSet(f1)

	w1 := &metadataFilterWrapper{
		parent: nil,
		meta:   fs1,
		docs:   docs,
	}

	assert.False(t, w1.Matches(0))
	assert.True(t, w1.Matches(1))
	assert.False(t, w1.Matches(2))
	assert.False(t, w1.Matches(3))
	assert.False(t, w1.Matches(99)) // Out of bounds

	// 2. MatchesBatch
	ids := []uint32{0, 1, 2, 3, 99}
	out := make([]bool, len(ids))
	w1.MatchesBatch(ids, out)
	assert.Equal(t, []bool{false, true, false, false, false}, out)

	// 3. With Parent Filter
	// Parent allows 0, 1. Rejects 2.
	parent := &mockFilter{allowed: map[uint32]bool{0: true, 1: true}}

	w2 := &metadataFilterWrapper{
		parent: parent,
		meta:   fs1, // color == "blue"
		docs:   docs,
	}

	// id=0: Parent True, Meta False ("red" != "blue") -> False
	assert.False(t, w2.Matches(0))

	// id=1: Parent True, Meta True ("blue" == "blue") -> True
	assert.True(t, w2.Matches(1))

	// id=2: Parent False -> False (short circuit)
	assert.False(t, w2.Matches(2))

	// 4. Fallback methods
	assert.Nil(t, w1.AsBitmap())
	assert.True(t, w1.MatchesBlock(nil))
}

type mockFilter struct {
	allowed map[uint32]bool
}

func (m *mockFilter) Matches(id uint32) bool {
	return m.allowed[id]
}

func (m *mockFilter) MatchesBatch(ids []uint32, out []bool) {
	for i, id := range ids {
		out[i] = m.Matches(id)
	}
}

func (m *mockFilter) AsBitmap() segment.Bitmap {
	return nil
}

func (m *mockFilter) MatchesBlock(stats map[string]segment.FieldStats) bool {
	return true
}
