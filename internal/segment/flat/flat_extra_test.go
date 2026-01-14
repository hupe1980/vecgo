package flat

import (
	"bytes"
	"context"
	"io"
	"testing"

	"github.com/hupe1980/vecgo/blobstore"
	"github.com/hupe1980/vecgo/distance"
	"github.com/hupe1980/vecgo/internal/searcher"
	"github.com/hupe1980/vecgo/internal/segment"
	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type bytesBlob struct {
	data []byte
}

var _ blobstore.Blob = (*bytesBlob)(nil)
var _ blobstore.Mappable = (*bytesBlob)(nil)

func (b *bytesBlob) ReadAt(_ context.Context, p []byte, off int64) (int, error) {
	if off >= int64(len(b.data)) {
		return 0, io.EOF
	}
	n := copy(p, b.data[off:])
	if n < len(p) {
		return n, nil
	}
	return n, nil
}

func (b *bytesBlob) Size() int64 {
	return int64(len(b.data))
}

func (b *bytesBlob) Close() error {
	return nil
}

func (b *bytesBlob) ReadRange(_ context.Context, off, length int64) (io.ReadCloser, error) {
	if off+length > int64(len(b.data)) {
		return nil, io.EOF
	}
	return io.NopCloser(bytes.NewReader(b.data[off : off+length])), nil
}

func (b *bytesBlob) Bytes() ([]byte, error) {
	return b.data, nil
}

func TestFlatSegment_Extra_Coverage(t *testing.T) {
	ctx := context.Background()
	dim := 2
	buf := new(bytes.Buffer)
	payloadBuf := new(bytes.Buffer)

	// NewWriter(w, payloadW, segID, dim, metric, k, quantType) // 0=None for quantType
	w := NewWriter(buf, payloadBuf, model.SegmentID(1), dim, distance.MetricL2, 1, 0)

	// Add data
	id1 := model.ID(100)
	vec1 := []float32{1.0, 1.0}
	md1Map := map[string]any{"color": "red"}
	md1, err := metadata.FromMap(md1Map)
	require.NoError(t, err)
	payload1 := []byte("payload1")

	err = w.Add(id1, vec1, md1, payload1)
	require.NoError(t, err)

	err = w.Flush(context.Background())
	require.NoError(t, err)

	// Create Blobs
	mainBlob := &bytesBlob{data: buf.Bytes()}
	payloadBlob := &bytesBlob{data: payloadBuf.Bytes()}

	// Open Segment
	s, err := Open(context.Background(), mainBlob, WithPayloadBlob(payloadBlob), WithVerifyChecksum(true))
	require.NoError(t, err)
	defer s.Close()

	// 1. Fetch
	batch, err := s.Fetch(ctx, []uint32{0}, nil) // All columns
	require.NoError(t, err)
	require.Equal(t, 1, batch.RowCount())
	assert.Equal(t, id1, batch.ID(0))
	assert.Equal(t, vec1, batch.Vector(0))
	assert.Equal(t, payload1, batch.Payload(0))

	fetchedMD := batch.Metadata(0)
	v, ok := fetchedMD["color"]
	assert.True(t, ok)
	assert.Equal(t, "red", v.StringValue())

	// 2. FetchIDs
	dstIDs := make([]model.ID, 1)
	err = s.FetchIDs(ctx, []uint32{0}, dstIDs)
	require.NoError(t, err)
	assert.Equal(t, id1, dstIDs[0])

	// 3. Size
	assert.Greater(t, s.Size(), int64(0))

	// 4. Iterate
	iterCount := 0
	err = s.Iterate(context.Background(), func(rowID uint32, id model.ID, vec []float32, md metadata.Document, payload []byte) error {
		iterCount++
		assert.Equal(t, uint32(0), rowID)
		assert.Equal(t, id1, id)
		assert.Equal(t, vec1, vec)
		assert.Equal(t, payload1, payload)
		v := md["color"]
		assert.Equal(t, "red", v.StringValue())
		return nil
	})
	require.NoError(t, err)
	assert.Equal(t, 1, iterCount)

	// 5. Search
	k := 5
	h := searcher.NewCandidateHeap(k, false)
	srch := &searcher.Searcher{Heap: h}

	query := []float32{1.0, 0.9} // Close to 1.0, 1.0
	// Search(ctx, q, k, filter, opts, searcherCtx)
	err = s.Search(ctx, query, k, nil, model.SearchOptions{}, srch)
	require.NoError(t, err)
	require.Equal(t, 1, h.Len())
	top := h.Candidates[0]
	assert.Equal(t, uint32(1), top.SegmentID)
	assert.Equal(t, uint32(0), top.RowID)

	// 6. Filter test
	h2 := searcher.NewCandidateHeap(k, false)
	srch2 := &searcher.Searcher{Heap: h2}

	fs := metadata.NewFilterSet(metadata.Filter{
		Key:      "color",
		Operator: metadata.OpEqual,
		Value:    metadata.String("blue"),
	})

	err = s.Search(ctx, query, k, nil, model.SearchOptions{Filter: fs}, srch2)
	require.NoError(t, err)
	assert.Equal(t, 0, h2.Len())

	// 7. Advise
	err = s.Advise(segment.AccessSequential)
	require.NoError(t, err)
}

func TestFlatSegment_SQ8_Coverage(t *testing.T) {
	ctx := context.Background()
	dim := 4
	buf := new(bytes.Buffer)
	payloadBuf := new(bytes.Buffer)

	// SQ8 Quantization (type 1)
	w := NewWriter(buf, payloadBuf, model.SegmentID(2), dim, distance.MetricL2, 1, 1)

	// Add data
	id1 := model.ID(200)
	vec1 := []float32{1.1, 2.2, 3.3, 4.4}
	md1, _ := metadata.FromMap(map[string]any{"tag": "q"})

	err := w.Add(id1, vec1, md1, nil)
	require.NoError(t, err)

	err = w.Flush(context.Background())
	require.NoError(t, err)

	mainBlob := &bytesBlob{data: buf.Bytes()}
	payloadBlob := &bytesBlob{data: payloadBuf.Bytes()}

	s, err := Open(context.Background(), mainBlob, WithPayloadBlob(payloadBlob))
	require.NoError(t, err)
	defer s.Close()

	// Check QuantizationType
	assert.Equal(t, 1, s.QuantizationType())

	// Search
	k := 5
	h := searcher.NewCandidateHeap(k, false)
	srch := &searcher.Searcher{Heap: h}
	q := []float32{1.0, 2.0, 3.0, 4.0}

	err = s.Search(ctx, q, k, nil, model.SearchOptions{}, srch)
	require.NoError(t, err)
	require.Equal(t, 1, h.Len())
}

func TestFlatSegment_ErrorCases(t *testing.T) {
	ctx := context.Background()
	dim := 2
	buf := new(bytes.Buffer)
	payloadBuf := new(bytes.Buffer)

	w := NewWriter(buf, payloadBuf, model.SegmentID(3), dim, distance.MetricL2, 1, 0)
	// Add one item
	_ = w.Add(model.ID(1), []float32{1, 1}, nil, nil)
	_ = w.Flush(context.Background())

	mainBlob := &bytesBlob{data: buf.Bytes()}
	payloadBlob := &bytesBlob{data: payloadBuf.Bytes()}
	s, mdErr := Open(context.Background(), mainBlob, WithPayloadBlob(payloadBlob), WithBlockCache(nil)) // Cover WithBlockCache
	require.NoError(t, mdErr)
	defer s.Close()

	// Fetch OOB
	_, err := s.Fetch(ctx, []uint32{100}, nil)
	assert.Error(t, err)

	// FetchIDs OOB
	dst := make([]model.ID, 1)
	err = s.FetchIDs(ctx, []uint32{100}, dst)
	assert.Error(t, err)

	// FetchIDs len mismatch
	err = s.FetchIDs(ctx, []uint32{0}, make([]model.ID, 0))
	assert.Error(t, err)
}

func TestFlatSegment_Large_Coverage(t *testing.T) {
	ctx := context.Background()
	dim := 2
	buf := new(bytes.Buffer)
	payloadBuf := new(bytes.Buffer)

	// Create writer
	w := NewWriter(buf, payloadBuf, model.SegmentID(4), dim, distance.MetricL2, 1, 0)

	// Add 1100 items to ensure at least one block boundary (assuming 1024 default)
	// We need metadata on them to generate stats.
	for i := 0; i < 1100; i++ {
		id := model.ID(uint64(i))
		vec := []float32{float32(i), float32(i)}
		val := "small"
		if i >= 1000 {
			val = "large"
		}
		md, _ := metadata.FromMap(map[string]any{"size": val})
		_ = w.Add(id, vec, md, nil)
	}

	require.NoError(t, w.Flush(context.Background()))

	mainBlob := &bytesBlob{data: buf.Bytes()}
	payloadBlob := &bytesBlob{data: payloadBuf.Bytes()}
	s, err := Open(context.Background(), mainBlob, WithPayloadBlob(payloadBlob))
	require.NoError(t, err)
	defer s.Close()

	// Search with FilterSet that matches "large" (only last 100 items)
	// This should allow checking block stats for the first block (all "small") and skipping it?
	// Block 0: 0-1023 (all "small"). Filter "large". -> Should Skip Block 0.

	fs := metadata.NewFilterSet(metadata.Filter{
		Key:      "size",
		Operator: metadata.OpEqual,
		Value:    metadata.String("large"),
	})

	k := 10
	h := searcher.NewCandidateHeap(k, false)
	srch := &searcher.Searcher{Heap: h}
	query := []float32{1050.0, 1050.0} // Near large items

	err = s.Search(ctx, query, k, nil, model.SearchOptions{Filter: fs}, srch)
	require.NoError(t, err)
	assert.Greater(t, h.Len(), 0)

	// Check results are >= 1000
	for _, c := range h.Candidates {
		assert.GreaterOrEqual(t, int(c.RowID), 1000)
	}
}
