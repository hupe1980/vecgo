package pk

import (
	"bytes"
	"testing"

	"github.com/hupe1980/vecgo/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMemoryIndex(t *testing.T) {
	idx := NewMemoryIndex()

	// 1. Upsert
	err := idx.Upsert(1, model.Location{SegmentID: 10, RowID: 100})
	require.NoError(t, err)
	err = idx.Upsert(2, model.Location{SegmentID: 10, RowID: 101})
	require.NoError(t, err)

	// 2. Lookup
	loc, ok := idx.Lookup(1)
	assert.True(t, ok)
	assert.Equal(t, model.Location{SegmentID: 10, RowID: 100}, loc)

	loc, ok = idx.Lookup(3)
	assert.False(t, ok)

	// 3. Save/Load
	var buf bytes.Buffer
	err = idx.Save(&buf)
	require.NoError(t, err)

	idx2 := NewMemoryIndex()
	err = idx2.Load(&buf)
	require.NoError(t, err)

	loc, ok = idx2.Lookup(1)
	assert.True(t, ok)
	assert.Equal(t, model.Location{SegmentID: 10, RowID: 100}, loc)

	loc, ok = idx2.Lookup(2)
	assert.True(t, ok)
	assert.Equal(t, model.Location{SegmentID: 10, RowID: 101}, loc)

	// 4. Delete
	err = idx.Delete(1)
	require.NoError(t, err)
	_, ok = idx.Lookup(1)
	assert.False(t, ok)
}
