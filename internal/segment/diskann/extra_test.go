package diskann

import (
	"bytes"
	"context"
	"io"
	"os"
	"testing"

	"github.com/hupe1980/vecgo/distance"
	"github.com/hupe1980/vecgo/internal/quantization"
	"github.com/hupe1980/vecgo/internal/searcher"
	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// memoryBlob implements both Blob and Mappable for testing
type memoryBlob struct {
	data []byte
}

func (m *memoryBlob) ReadAt(_ context.Context, p []byte, off int64) (n int, err error) {
	if off >= int64(len(m.data)) {
		return 0, io.EOF
	}
	n = copy(p, m.data[off:])
	if n < len(p) {
		return n, io.EOF
	}
	return n, nil
}

func (m *memoryBlob) Size() int64 {
	return int64(len(m.data))
}

func (m *memoryBlob) Close() error { return nil }

func (m *memoryBlob) Bytes() ([]byte, error) {
	return m.data, nil
}

func (m *memoryBlob) ReadRange(_ context.Context, off, length int64) (io.ReadCloser, error) {
	if off < 0 {
		return nil, io.EOF
	}
	if off >= int64(len(m.data)) {
		return io.NopCloser(bytes.NewReader(nil)), nil
	}
	if off+length > int64(len(m.data)) {
		length = int64(len(m.data)) - off
	}
	return io.NopCloser(bytes.NewReader(m.data[off : off+length])), nil
}

// nonMappableBlob implements Blob but NOT Mappable
type nonMappableBlob struct {
	data []byte
}

func (m *nonMappableBlob) ReadAt(_ context.Context, p []byte, off int64) (n int, err error) {
	if off >= int64(len(m.data)) {
		return 0, io.EOF
	}
	n = copy(p, m.data[off:])
	if n < len(p) {
		return n, io.EOF
	}
	return n, nil
}

func (m *nonMappableBlob) Size() int64 {
	return int64(len(m.data))
}

func (m *nonMappableBlob) Close() error { return nil }

func (m *nonMappableBlob) ReadRange(_ context.Context, off, length int64) (io.ReadCloser, error) {
	if off < 0 {
		return nil, io.EOF
	}
	if off >= int64(len(m.data)) {
		return io.NopCloser(bytes.NewReader(nil)), nil
	}
	if off+length > int64(len(m.data)) {
		length = int64(len(m.data)) - off
	}
	return io.NopCloser(bytes.NewReader(m.data[off : off+length])), nil
}

func TestOpen_Errors(t *testing.T) {
	header := FileHeader{
		Magic:    MagicNumber,
		Version:  Version,
		RowCount: 10,
		Dim:      128,
	}
	headerBytes := header.Encode()

	t.Run("FileTooSmall", func(t *testing.T) {
		blob := &memoryBlob{data: make([]byte, HeaderSize-1)}
		_, err := Open(context.Background(), blob)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "too small")
	})

	t.Run("HeaderDecodeFail", func(t *testing.T) {
		badData := make([]byte, HeaderSize)
		copy(badData, headerBytes)
		badData[0] = 0 // Bad magic
		blob := &memoryBlob{data: badData}
		_, err := Open(context.Background(), blob)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "magic")
	})

	t.Run("ChecksumMismatch", func(t *testing.T) {
		h := header
		h.Checksum = 0xDEADBEEF
		data := make([]byte, HeaderSize+100)
		copy(data, h.Encode())
		blob := &memoryBlob{data: data}
		_, err := Open(context.Background(), blob, WithVerifyChecksum(true))
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "checksum mismatch")
	})

	t.Run("VectorSectionOutOfBounds", func(t *testing.T) {
		h := header
		h.VectorOffset = 1000000
		blob := &memoryBlob{data: h.Encode()}
		_, err := Open(context.Background(), blob)
		assert.Error(t, err)
	})
}

func TestOpen_NonMappable(t *testing.T) {
	h := FileHeader{
		Magic:        MagicNumber,
		Version:      Version,
		RowCount:     0,
		Dim:          16,
		PKOffset:     HeaderSize,
		VectorOffset: HeaderSize,
	}
	data := h.Encode()

	blob := &nonMappableBlob{data: data}
	s, err := Open(context.Background(), blob)
	require.NoError(t, err)
	defer s.Close()
	assert.Equal(t, int64(len(data)), s.Size())
}

func TestSegment_Close(t *testing.T) {
	h := FileHeader{
		Magic:        MagicNumber,
		Version:      Version,
		RowCount:     0,
		Dim:          16,
		PKOffset:     HeaderSize,
		VectorOffset: HeaderSize,
	}
	blob := &memoryBlob{data: h.Encode()}
	s, err := Open(context.Background(), blob)
	require.NoError(t, err)
	err = s.Close()
	assert.NoError(t, err)
}

func TestNonMappable_Integration(t *testing.T) {
	f, err := os.CreateTemp("", "diskann_integration")
	require.NoError(t, err)
	defer os.Remove(f.Name())

	opts := Options{R: 4, L: 10}
	w := NewWriter(f, nil, 123, 2, distance.MetricL2, opts)
	require.NoError(t, w.Add(model.ID(1), []float32{1.0, 0.0}, nil, nil))
	require.NoError(t, w.Add(model.ID(2), []float32{0.0, 1.0}, nil, nil))
	require.NoError(t, w.Write(context.Background()))
	f.Close()

	content, err := os.ReadFile(f.Name())
	require.NoError(t, err)

	blob := &nonMappableBlob{data: content}
	s, err := Open(context.Background(), blob)
	require.NoError(t, err)
	defer s.Close()

	sc := searcher.Get()
	defer searcher.Put(sc)
	sc.Heap.Reset(false)

	err = s.Search(context.Background(), []float32{1.0, 0.0}, 2, nil, model.SearchOptions{}, sc)
	require.NoError(t, err)
	assert.GreaterOrEqual(t, sc.Heap.Len(), 1)

	vec, err := s.Get(context.Background(), 0)
	require.NoError(t, err)
	assert.Equal(t, float32(1.0), vec[0])
}

func TestExtras_Integration(t *testing.T) {
	f, err := os.CreateTemp("", "diskann_extras")
	require.NoError(t, err)
	defer os.Remove(f.Name())

	opts := Options{R: 4, L: 10}
	w := NewWriter(f, nil, 123, 2, distance.MetricL2, opts)

	md := metadata.Document{"key": metadata.String("value")}
	require.NoError(t, w.Add(model.ID(1), []float32{1.0, 0.0}, md, nil))
	require.NoError(t, w.Add(model.ID(2), []float32{0.0, 1.0}, nil, nil))
	require.NoError(t, w.Write(context.Background()))
	f.Close()

	content, err := os.ReadFile(f.Name())
	require.NoError(t, err)

	blob := &memoryBlob{data: content}
	s, err := Open(context.Background(), blob)
	require.NoError(t, err)
	defer s.Close()

	count := 0
	err = s.Iterate(context.Background(), func(rowID uint32, id model.ID, vec []float32, md metadata.Document, p []byte) error {
		count++
		if rowID == 0 {
			assert.Equal(t, float32(1.0), vec[0])
			require.NotNil(t, md)
			assert.Equal(t, "value", md["key"].StringValue())
		}
		if rowID == 1 {
			assert.Nil(t, md)
		}
		return nil
	})
	require.NoError(t, err)
	assert.Equal(t, 2, count)

	ids := make([]model.ID, 2)
	err = s.FetchIDs(context.Background(), []uint32{0, 1}, ids)
	require.NoError(t, err)
	assert.Equal(t, model.ID(1), ids[0])

	batch, err := s.Fetch(context.Background(), []uint32{0}, nil)
	require.NoError(t, err)
	assert.NotNil(t, batch)
}

func TestPQ_Integration(t *testing.T) {
	// t.Skip("PQ integration test disabled due to panic in quantization")
	f, err := os.CreateTemp("", "diskann_pq")
	require.NoError(t, err)
	defer os.Remove(f.Name())

	opts := Options{
		R:            4,
		L:            10,
		PQSubvectors: 2,
	}
	w := NewWriter(f, nil, 123, 4, distance.MetricL2, opts)

	for i := 0; i < 256; i++ {
		vec := []float32{float32(i), float32(i), float32(i), float32(i)}
		require.NoError(t, w.Add(model.ID(uint64(i)), vec, nil, nil))
	}
	require.NoError(t, w.Write(context.Background()))
	f.Close()

	content, err := os.ReadFile(f.Name())
	require.NoError(t, err)

	blob := &memoryBlob{data: content}
	s, err := Open(context.Background(), blob)
	require.NoError(t, err)
	defer s.Close()

	sc := searcher.Get()
	defer searcher.Put(sc)
	sc.Heap.Reset(false)

	err = s.Search(context.Background(), []float32{0, 0, 0, 0}, 2, nil, model.SearchOptions{}, sc)
	require.NoError(t, err)
	assert.Greater(t, sc.Heap.Len(), 0)
}

func TestRaBitQ_Integration(t *testing.T) {
	f, err := os.CreateTemp("", "diskann_rabitq")
	require.NoError(t, err)
	defer os.Remove(f.Name())

	opts := Options{
		R:                4,
		L:                10,
		QuantizationType: quantization.TypeRaBitQ,
	}
	w := NewWriter(f, nil, 123, 4, distance.MetricL2, opts)

	for i := 0; i < 256; i++ {
		vec := []float32{float32(i), float32(i), float32(i), float32(i)}
		require.NoError(t, w.Add(model.ID(uint64(i)), vec, nil, nil))
	}
	require.NoError(t, w.Write(context.Background()))
	f.Close()

	content, err := os.ReadFile(f.Name())
	require.NoError(t, err)

	blob := &memoryBlob{data: content}
	s, err := Open(context.Background(), blob)
	require.NoError(t, err)
	defer s.Close()

	sc := searcher.Get()
	defer searcher.Put(sc)
	sc.Heap.Reset(false)

	err = s.Search(context.Background(), []float32{0, 0, 0, 0}, 2, nil, model.SearchOptions{}, sc)
	require.NoError(t, err)
	assert.Greater(t, sc.Heap.Len(), 0)
}
