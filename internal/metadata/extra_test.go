package imetadata

import (
	"bytes"
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

func TestUnifiedIndex_Helpers(t *testing.T) {
	ui := NewUnifiedIndex()
	ui.Set(1, metadata.Document{"tag": metadata.String("a")})

	// AddInvertedIndex coverage
	ui.AddInvertedIndex(2, metadata.Document{"tag": metadata.String("b")})

	// Verify 2 is in inverted but not documents
	doc, ok := ui.Get(2)
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
	ui.SetDocumentProvider(func(id model.RowID) (metadata.Document, bool) {
		if id == 99 {
			return metadata.Document{"magic": metadata.String("yes")}, true
		}
		return nil, false
	})

	doc99, ok := ui.Get(99)
	assert.True(t, ok)
	assert.Equal(t, "yes", doc99["magic"].StringValue())

	// Delete
	ui.Delete(1)
	_, ok = ui.Get(1)
	assert.False(t, ok)
	assert.Equal(t, 0, ui.Len())
}
