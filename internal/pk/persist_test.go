package pk

import (
	"bytes"
	"testing"

	"github.com/hupe1980/vecgo/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestIndex_SaveLoad(t *testing.T) {
	idx := New()

	// Populate some data
	// ID 1: Active
	idx.Upsert(1, model.Location{SegmentID: 1, RowID: 100}, 10)
	// ID 2: Deleted
	idx.Upsert(2, model.Location{SegmentID: 1, RowID: 200}, 20)
	idx.Delete(2, 21)
	// ID 100000: Far page
	idx.Upsert(100000, model.Location{SegmentID: 2, RowID: 300}, 30)

	var buf bytes.Buffer
	err := idx.Save(&buf)
	require.NoError(t, err)

	// Load into new index
	idx2 := New()
	err = idx2.Load(&buf)
	require.NoError(t, err)

	assert.Equal(t, idx.count.Load(), idx2.count.Load())

	// Check ID 1
	loc, exists := idx2.Get(1, 100) // LSN 100 > 10
	assert.True(t, exists)
	assert.Equal(t, model.Location{SegmentID: 1, RowID: 100}, loc)

	// Check ID 2 (Should be deleted)
	// Get at LSN 100 should return false (visible deletion)
	_, exists = idx2.Get(2, 100)
	assert.False(t, exists)

	// Check ID 2 (Time travel?)
	// Load only persists LATEST version. So time travel < 21 is lost?
	// Our Save implementation persists HEAD.
	// HEAD of 2 is tombstone at LSN 21.
	// If we query Get(2, 20), we compare 20 against HEAD LSN 21.
	// If 21 <= 20? No. So HEAD is NOT visible.
	// Then we go to next. But Save only saved HEAD. Next is nil.
	// So Get(2, 20) will return "not found" (implied empty history?).
	// This confirms Checkpoint truncates history.

	// Check ID 100000
	loc, exists = idx2.Get(100000, 100)
	assert.True(t, exists)
	assert.Equal(t, model.Location{SegmentID: 2, RowID: 300}, loc)
}

// Minimal check of Get to ensure test validity
func TestIndex_Get(t *testing.T) {
	idx := New()
	idx.Upsert(1, model.Location{SegmentID: 1, RowID: 100}, 10)

	loc, exists := idx.Get(1, 100)
	assert.True(t, exists)
	assert.Equal(t, model.Location{SegmentID: 1, RowID: 100}, loc)

	idx.Delete(1, 11)
	_, exists = idx.Get(1, 100)
	assert.False(t, exists)
}
