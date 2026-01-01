package index

import (
	"bytes"
	"context"
	"encoding/binary"
	"io"
	"iter"
	"testing"

	"github.com/hupe1980/vecgo/core"
	"github.com/hupe1980/vecgo/persistence"
	"github.com/hupe1980/vecgo/searcher"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// MockIndex implements Index interface
type MockIndex struct{}

func (m *MockIndex) Insert(ctx context.Context, v []float32) (core.LocalID, error) { return 0, nil }
func (m *MockIndex) BatchInsert(ctx context.Context, vectors [][]float32) BatchInsertResult {
	return BatchInsertResult{}
}
func (m *MockIndex) Delete(ctx context.Context, id core.LocalID) error              { return nil }
func (m *MockIndex) Update(ctx context.Context, id core.LocalID, v []float32) error { return nil }
func (m *MockIndex) KNNSearch(ctx context.Context, q []float32, k int, opts *SearchOptions) ([]SearchResult, error) {
	return nil, nil
}
func (m *MockIndex) KNNSearchWithBuffer(ctx context.Context, q []float32, k int, opts *SearchOptions, buf *[]SearchResult) error {
	return nil
}
func (m *MockIndex) KNNSearchWithContext(ctx context.Context, s *searcher.Searcher, q []float32, k int, opts *SearchOptions) error {
	return nil
}
func (m *MockIndex) KNNSearchStream(ctx context.Context, q []float32, k int, opts *SearchOptions) iter.Seq2[SearchResult, error] {
	return func(yield func(SearchResult, error) bool) {}
}
func (m *MockIndex) BruteSearch(ctx context.Context, query []float32, k int, filter func(id core.LocalID) bool) ([]SearchResult, error) {
	return nil, nil
}
func (m *MockIndex) Stats() Stats   { return Stats{} }
func (m *MockIndex) Dimension() int { return 0 }

func TestBinaryRegistry(t *testing.T) {
	// Register a mock loader
	mockType := uint8(255)
	called := false
	loader := func(r io.Reader) (Index, error) {
		called = true
		return &MockIndex{}, nil
	}
	RegisterBinaryLoader(mockType, loader)

	// Prepare valid header
	// Magic: 0x56454330
	// Type: mockType at offset 8
	header := make([]byte, 64)
	binary.LittleEndian.PutUint32(header[0:4], 0x56454330)
	header[8] = mockType

	r := bytes.NewReader(header)
	idx, err := LoadBinaryIndex(r)
	require.NoError(t, err)
	assert.NotNil(t, idx)
	assert.True(t, called)

	// Test invalid magic
	headerInvalid := make([]byte, 64)
	binary.LittleEndian.PutUint32(headerInvalid[0:4], 0xDEADBEEF)
	rInvalid := bytes.NewReader(headerInvalid)
	_, err = LoadBinaryIndex(rInvalid)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "invalid magic number")

	// Test unknown type
	headerUnknown := make([]byte, 64)
	binary.LittleEndian.PutUint32(headerUnknown[0:4], 0x56454330)
	headerUnknown[8] = 254 // Unknown type
	rUnknown := bytes.NewReader(headerUnknown)
	_, err = LoadBinaryIndex(rUnknown)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "unknown index type")
}

func TestMmapRegistry(t *testing.T) {
	// Register a mock loader
	mockType := uint8(255)
	called := false
	loader := func(data []byte) (Index, int, error) {
		called = true
		return &MockIndex{}, len(data), nil
	}
	RegisterMmapBinaryLoader(mockType, loader)

	// Prepare valid header
	// Magic: persistence.MagicNumber
	// Type: mockType at offset 8
	headerSize := binary.Size(persistence.FileHeader{})
	header := make([]byte, headerSize)
	binary.LittleEndian.PutUint32(header[0:4], persistence.MagicNumber)
	header[8] = mockType

	idx, consumed, err := LoadBinaryIndexMmap(header)
	require.NoError(t, err)
	assert.NotNil(t, idx)
	assert.Equal(t, headerSize, consumed)
	assert.True(t, called)

	// Test invalid magic
	headerInvalid := make([]byte, headerSize)
	binary.LittleEndian.PutUint32(headerInvalid[0:4], 0xDEADBEEF)
	_, _, err = LoadBinaryIndexMmap(headerInvalid)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "invalid magic number")

	// Test unknown type
	headerUnknown := make([]byte, headerSize)
	binary.LittleEndian.PutUint32(headerUnknown[0:4], persistence.MagicNumber)
	headerUnknown[8] = 254 // Unknown type
	_, _, err = LoadBinaryIndexMmap(headerUnknown)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "unknown index type")

	// Test too small
	_, _, err = LoadBinaryIndexMmap([]byte{1, 2, 3})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "snapshot too small")
}
