package imetadata

import (
	"bytes"
	"context"
	"errors"
	"testing"

	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type errorWriter struct {
	err error
}

func (w *errorWriter) Write(p []byte) (n int, err error) {
	return 0, w.err
}

type errorReader struct {
	err error
}

func (r *errorReader) Read(p []byte) (n int, err error) {
	return 0, r.err
}

func TestUnifiedIndex_Serialization(t *testing.T) {
	ui := NewUnifiedIndex()
	ui.Set(1, metadata.Document{"tag": metadata.String("a"), "year": metadata.String("2023")})
	ui.Set(2, metadata.Document{"tag": metadata.String("b"), "year": metadata.String("2024")})
	ui.Set(3, metadata.Document{"tag": metadata.String("a")})

	var buf bytes.Buffer
	err := ui.WriteInvertedIndex(&buf)
	require.NoError(t, err)

	ui2 := NewUnifiedIndex()
	err = ui2.ReadInvertedIndex(&buf)
	require.NoError(t, err)

	// Check inverted index structure by compiling filters
	f1 := &metadata.FilterSet{Filters: []metadata.Filter{
		{Key: "tag", Value: metadata.String("a"), Operator: metadata.OpEqual},
	}}
	res1 := ui2.CompileFilter(f1)
	require.NotNil(t, res1)
	assert.True(t, res1.Contains(1))
	assert.True(t, res1.Contains(3))
	assert.False(t, res1.Contains(2))

	// Stats check
	stats := ui2.GetStats()
	assert.Greater(t, stats.FieldCount, 0)
	assert.Greater(t, stats.BitmapCount, 0)
}

func TestUnifiedIndex_Serialization_Errors(t *testing.T) {
	ui := NewUnifiedIndex()
	ui.Set(1, metadata.Document{"tag": metadata.String("a")})

	t.Run("WriteError", func(t *testing.T) {
		errWriter := &errorWriter{err: errors.New("write fail")}
		err := ui.WriteInvertedIndex(errWriter)
		assert.Error(t, err)
	})

	t.Run("ReadError", func(t *testing.T) {
		errReader := &errorReader{err: errors.New("read fail")}
		ui2 := NewUnifiedIndex()
		err := ui2.ReadInvertedIndex(errReader)
		assert.Error(t, err)
	})

	t.Run("ReadTruncated", func(t *testing.T) {
		var buf bytes.Buffer
		ui.WriteInvertedIndex(&buf)
		truncated := buf.Bytes()[:len(buf.Bytes())-1]

		ui2 := NewUnifiedIndex()
		err := ui2.ReadInvertedIndex(bytes.NewReader(truncated))
		assert.Error(t, err)
	})
}

func TestLocalBitmap_Extras(t *testing.T) {
	b := NewLocalBitmap()
	b.Add(1)
	b.Add(2)
	b.Add(3)

	assert.Equal(t, uint64(3), b.Cardinality())
	assert.False(t, b.IsEmpty())
	assert.Greater(t, b.GetSizeInBytes(), uint64(0))

	b2 := b.Clone()
	assert.Equal(t, uint64(3), b2.Cardinality())

	b.Clear()
	assert.True(t, b.IsEmpty())
	assert.False(t, b2.IsEmpty())

	count := 0
	for range b2.Iterator() {
		count++
	}
	assert.Equal(t, 3, count)
}

func TestFilterResult_CloneInto(t *testing.T) {
	t.Run("FilterRows_BasicClone", func(t *testing.T) {
		// Create a FilterResult with some rows
		original := RowsResult([]uint32{1, 5, 10, 15})

		// Clone into a new buffer
		buf := make([]uint32, 0, 16)
		cloned, newBuf := original.CloneInto(buf)

		// Verify cloned result
		assert.Equal(t, FilterRows, cloned.Mode())
		assert.Equal(t, 4, cloned.Cardinality())
		assert.True(t, cloned.Contains(1))
		assert.True(t, cloned.Contains(5))
		assert.True(t, cloned.Contains(10))
		assert.True(t, cloned.Contains(15))
		assert.False(t, cloned.Contains(2))

		// Verify buffer was extended
		assert.Equal(t, 4, len(newBuf))
	})

	t.Run("FilterRows_MultipleClones", func(t *testing.T) {
		// Simulate collecting multiple FilterResults into one buffer
		buf := make([]uint32, 0, 32)

		fr1 := RowsResult([]uint32{1, 2, 3})
		fr2 := RowsResult([]uint32{10, 20, 30})
		fr3 := RowsResult([]uint32{100, 200})

		cloned1, buf := fr1.CloneInto(buf)
		cloned2, buf := fr2.CloneInto(buf)
		cloned3, buf := fr3.CloneInto(buf)

		// Verify all clones are independent
		assert.Equal(t, 3, cloned1.Cardinality())
		assert.Equal(t, 3, cloned2.Cardinality())
		assert.Equal(t, 2, cloned3.Cardinality())

		// Verify no overlap
		assert.True(t, cloned1.Contains(1))
		assert.False(t, cloned1.Contains(10))
		assert.True(t, cloned2.Contains(10))
		assert.False(t, cloned2.Contains(100))
		assert.True(t, cloned3.Contains(100))

		// Buffer should contain all rows
		assert.Equal(t, 8, len(buf))
	})

	t.Run("FilterNone_CloneInto", func(t *testing.T) {
		original := FilterResult{mode: FilterNone}
		buf := make([]uint32, 0, 8)
		cloned, newBuf := original.CloneInto(buf)

		assert.True(t, cloned.IsEmpty())
		assert.Equal(t, 0, len(newBuf))
	})

	t.Run("FilterAll_CloneInto", func(t *testing.T) {
		original := FilterResult{mode: FilterAll}
		buf := make([]uint32, 0, 8)
		cloned, newBuf := original.CloneInto(buf)

		assert.True(t, cloned.IsAll())
		assert.Equal(t, 0, len(newBuf))
	})

	t.Run("EmptyRows_CloneInto", func(t *testing.T) {
		original := RowsResult([]uint32{})
		buf := make([]uint32, 0, 8)
		cloned, newBuf := original.CloneInto(buf)

		assert.True(t, cloned.IsEmpty())
		assert.Equal(t, 0, len(newBuf))
	})

	t.Run("CloneInto_NoAliasing", func(t *testing.T) {
		// Test that modifying the original doesn't affect the clone
		originalRows := []uint32{1, 2, 3, 4, 5}
		original := RowsResult(originalRows)

		buf := make([]uint32, 0, 16)
		cloned, _ := original.CloneInto(buf)

		// Modify original rows
		originalRows[0] = 999

		// Clone should be unaffected
		assert.True(t, cloned.Contains(1))
		assert.False(t, cloned.Contains(999))
	})
}

func TestUnifiedIndex_Helpers(t *testing.T) {
	ui := NewUnifiedIndex()
	ui.Set(1, metadata.Document{"tag": metadata.String("a")})

	// AddInvertedIndex coverage
	ui.AddInvertedIndex(2, metadata.Document{"tag": metadata.String("b")})

	// Verify 2 is in inverted but not documents
	doc, ok := ui.Get(context.Background(), 2)
	assert.False(t, ok)
	assert.Nil(t, doc)

	// Check inverted via filter
	f := &metadata.FilterSet{Filters: []metadata.Filter{
		{Key: "tag", Value: metadata.String("b"), Operator: metadata.OpEqual},
	}}
	res := ui.CompileFilter(f)
	assert.True(t, res.Contains(2))

	// ToMap
	m := ui.ToMap()
	assert.Len(t, m, 1) // Only doc 1 is in documents map

	// SetDocumentProvider coverage
	ui.SetDocumentProvider(func(_ context.Context, id model.RowID) (metadata.Document, bool) {
		if id == 99 {
			return metadata.Document{"magic": metadata.String("yes")}, true
		}
		return nil, false
	})

	doc99, ok := ui.Get(context.Background(), 99)
	assert.True(t, ok)
	assert.Equal(t, "yes", doc99["magic"].StringValue())

	// Delete
	ui.Delete(1)
	_, ok = ui.Get(context.Background(), 1)
	assert.False(t, ok)
	assert.Equal(t, 0, ui.Len())
}
