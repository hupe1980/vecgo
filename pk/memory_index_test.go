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
	err := idx.Upsert(model.PKUint64(1), model.Location{SegmentID: 10, RowID: 100})
	require.NoError(t, err)
	err = idx.Upsert(model.PKUint64(2), model.Location{SegmentID: 10, RowID: 101})
	require.NoError(t, err)

	// 2. Lookup
	loc, ok := idx.Lookup(model.PKUint64(1))
	assert.True(t, ok)
	assert.Equal(t, model.Location{SegmentID: 10, RowID: 100}, loc)

	loc, ok = idx.Lookup(model.PKUint64(3))
	assert.False(t, ok)

	// 3. Save/Load
	var buf bytes.Buffer
	err = idx.Save(&buf)
	require.NoError(t, err)

	idx2 := NewMemoryIndex()
	err = idx2.Load(&buf)
	require.NoError(t, err)

	loc, ok = idx2.Lookup(model.PKUint64(1))
	assert.True(t, ok)
	assert.Equal(t, model.Location{SegmentID: 10, RowID: 100}, loc)

	loc, ok = idx2.Lookup(model.PKUint64(2))
	assert.True(t, ok)
	assert.Equal(t, model.Location{SegmentID: 10, RowID: 101}, loc)

	// 4. Delete
	err = idx.Delete(model.PKUint64(1))
	require.NoError(t, err)
	_, ok = idx.Lookup(model.PKUint64(1))
	assert.False(t, ok)
}

func TestMemoryIndex_Persistence(t *testing.T) {
	t.Run("Mixed Types", func(t *testing.T) {
		idx := NewMemoryIndex()
		require.NoError(t, idx.Upsert(model.PKUint64(1), model.Location{SegmentID: 1, RowID: 1}))
		require.NoError(t, idx.Upsert(model.PKString("foo"), model.Location{SegmentID: 1, RowID: 2}))

		var buf bytes.Buffer
		require.NoError(t, idx.Save(&buf))

		idx2 := NewMemoryIndex()
		require.NoError(t, idx2.Load(&buf))

		loc, ok := idx2.Lookup(model.PKUint64(1))
		assert.True(t, ok)
		assert.Equal(t, model.Location{SegmentID: 1, RowID: 1}, loc)

		loc, ok = idx2.Lookup(model.PKString("foo"))
		assert.True(t, ok)
		assert.Equal(t, model.Location{SegmentID: 1, RowID: 2}, loc)
	})

	t.Run("Writer Error", func(t *testing.T) {
		idx := NewMemoryIndex()
		idx.Upsert(model.PKUint64(1), model.Location{})

		err := idx.Save(&failWriter{failAfter: 0})
		assert.Error(t, err)
	})

	t.Run("Reader Error", func(t *testing.T) {
		idx := NewMemoryIndex()
		idx.Upsert(model.PKUint64(1), model.Location{})
		var buf bytes.Buffer
		idx.Save(&buf)

		idx2 := NewMemoryIndex()
		// truncated buffer
		err := idx2.Load(bytes.NewReader(buf.Bytes()[:4]))
		assert.Error(t, err)
	})
}

type failWriter struct {
	failAfter int
	written   int
}

func (w *failWriter) Write(p []byte) (n int, err error) {
	if w.written >= w.failAfter {
		return 0, assert.AnError
	}
	n = len(p)
	w.written += n
	return n, nil
}
