package diskann

import (
	"context"
	"io"
	"os"
	"path/filepath"
	"testing"

	"github.com/hupe1980/vecgo/blobstore"
	"github.com/hupe1980/vecgo/distance"
	"github.com/hupe1980/vecgo/internal/resource"
	"github.com/hupe1980/vecgo/internal/searcher"
	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestWriter(t *testing.T) {
	// Use a temp file because Writer requires Seeker to update header
	f, err := os.CreateTemp("", "diskann_test_*.bin")
	require.NoError(t, err)
	defer os.Remove(f.Name())
	defer f.Close()

	opts := Options{
		R:            4,
		L:            10,
		Alpha:        1.2,
		PQSubvectors: 0, // Disable PQ for simple test
	}

	w := NewWriter(f, nil, 123, 4, distance.MetricL2, opts)

	// Add some vectors
	vectors := [][]float32{
		{1, 0, 0, 0},
		{0, 1, 0, 0},
		{0, 0, 1, 0},
		{0, 0, 0, 1},
		{1, 1, 0, 0},
	}

	for i, v := range vectors {
		err := w.Add(model.ID(uint64(i)), v, nil, nil)
		require.NoError(t, err)
	}

	// Write
	err = w.Write(context.Background())
	require.NoError(t, err)

	// Verify content
	_, err = f.Seek(0, 0)
	require.NoError(t, err)

	data, err := io.ReadAll(f)
	require.NoError(t, err)

	require.Greater(t, len(data), HeaderSize)

	// Decode header
	h, err := DecodeHeader(data)
	require.NoError(t, err)

	assert.Equal(t, uint32(MagicNumber), h.Magic)
	assert.Equal(t, uint32(Version), h.Version)
	assert.Equal(t, uint64(123), h.SegmentID)
	assert.Equal(t, uint32(5), h.RowCount)
	assert.Equal(t, uint32(4), h.Dim)
	assert.Equal(t, uint32(4), h.MaxDegree)

	// Check offsets
	assert.Equal(t, uint64(HeaderSize), h.VectorOffset)

	// Check vectors
	// Vector size = 5 * 4 * 4 = 80 bytes
	expectedVectorSize := 5 * 4 * 4
	assert.Equal(t, uint64(HeaderSize+expectedVectorSize), h.GraphOffset)

	// Check graph
	// Graph size = 5 * 4 * 4 = 80 bytes
	expectedGraphSize := 5 * 4 * 4

	// Check PKs
	expectedPKSize := 5 * 8

	// Check Metadata
	expectedMetadataOffsetsSize := (5 + 1) * 8
	// Metadata blob size is 0 since we didn't add any metadata
	expectedMetadataSize := 0

	// Metadata Index size (at least 1 byte for count=0)
	expectedMetadataIndexSize := 1

	// Total size should be Header + Vectors + Graph + PKs + MetadataOffsets + Metadata + MetadataIndex
	assert.Equal(t, HeaderSize+expectedVectorSize+expectedGraphSize+expectedPKSize+expectedMetadataOffsetsSize+expectedMetadataSize+expectedMetadataIndexSize, len(data))

	// Test Reader
	st := blobstore.NewLocalStore(filepath.Dir(f.Name()))
	blob, err := st.Open(context.Background(), filepath.Base(f.Name()))
	require.NoError(t, err)
	s, err := Open(context.Background(), blob)
	require.NoError(t, err)
	defer s.Close()

	assert.Equal(t, uint32(5), s.header.RowCount)

	// Test Get
	vec, err := s.Get(context.Background(), 0)
	require.NoError(t, err)
	assert.Equal(t, vectors[0], vec)

	// Test Search
	// Query = {1, 0, 0, 0} (same as vector 0)
	sc := searcher.Get()
	defer searcher.Put(sc)
	sc.Heap.Reset(false)

	err = s.Search(context.Background(), []float32{1, 0, 0, 0}, 2, nil, model.SearchOptions{}, sc)
	require.NoError(t, err)
	assert.Equal(t, 2, sc.Heap.Len())

	var last model.Candidate
	for sc.Heap.Len() > 0 {
		last = sc.Heap.Pop().ToModel()
	}
	assert.Equal(t, model.RowID(0), last.Loc.RowID) // Should find itself
	assert.Equal(t, float32(0), last.Score)
}

func TestWriter_ResourceLimits(t *testing.T) {
	// Create controller with very low memory limit
	rc := resource.NewController(resource.Config{
		MemoryLimitBytes: 10, // 10 bytes is not enough for graph
	})

	opts := Options{
		R:                  4,
		L:                  10,
		ResourceController: rc,
	}

	w := NewWriter(io.Discard, nil, 1, 4, distance.MetricL2, opts)

	// Add vectors
	for i := 0; i < 10; i++ {
		w.Add(model.ID(uint64(i)), []float32{1, 0, 0, 0}, nil, nil)
	}

	// Write should fail due to memory limit (non-blocking fail-fast)
	err := w.Write(t.Context())
	assert.Error(t, err)
	assert.ErrorIs(t, err, resource.ErrMemoryLimitExceeded)
}

func TestMetadataPersistence(t *testing.T) {
	f, err := os.CreateTemp("", "diskann_metadata_test_*.bin")
	require.NoError(t, err)
	defer os.Remove(f.Name())
	defer f.Close()

	opts := Options{
		R:     4,
		L:     10,
		Alpha: 1.2,
	}

	w := NewWriter(f, nil, 123, 2, distance.MetricL2, opts)

	// Add vectors with metadata
	vectors := [][]float32{
		{1, 0},
		{0, 1},
	}

	md1 := map[string]any{
		"key1": "value1",
		"key2": 123.0, // Use float to match JSON unmarshal behavior
	}

	md2 := map[string]any{
		"tags": []any{"a", "b"},
	}

	doc1, err := metadata.FromMap(md1)
	require.NoError(t, err)
	err = w.Add(model.ID(0), vectors[0], doc1, nil)
	require.NoError(t, err)

	doc2, err := metadata.FromMap(md2)
	require.NoError(t, err)
	err = w.Add(model.ID(1), vectors[1], doc2, nil)
	require.NoError(t, err)

	// Flush
	err = w.Flush()
	require.NoError(t, err)

	// Open reader
	st := blobstore.NewLocalStore(filepath.Dir(f.Name()))
	blob, err := st.Open(context.Background(), filepath.Base(f.Name()))
	require.NoError(t, err)
	s, err := Open(context.Background(), blob)
	require.NoError(t, err)
	defer s.Close()

	// Fetch metadata
	batch, err := s.Fetch(context.Background(), []uint32{0, 1}, []string{"metadata"})
	require.NoError(t, err)
	require.Equal(t, 2, batch.RowCount())

	// Verify md1
	fetchedDoc1 := batch.Metadata(0)
	require.NotNil(t, fetchedDoc1)
	m1 := fetchedDoc1.ToMap()
	assert.Equal(t, "value1", m1["key1"])
	assert.Equal(t, 123.0, m1["key2"])

	// Verify md2
	fetchedDoc2 := batch.Metadata(1)
	require.NotNil(t, fetchedDoc2)
	m2 := fetchedDoc2.ToMap()
	// Arrays might be converted to something else or kept as []interface{}
	// metadata package might handle arrays specifically.
	// For now just check it exists.
	assert.NotNil(t, m2["tags"])

	// Iterate
	count := 0
	err = s.Iterate(context.Background(), func(rowID uint32, id model.ID, vec []float32, md metadata.Document, payload []byte) error {
		count++
		if rowID == 0 {
			require.NotNil(t, md)
			m := md.ToMap()
			assert.Equal(t, "value1", m["key1"])
		}
		if rowID == 1 {
			require.NotNil(t, md)
		}
		return nil
	})
	require.NoError(t, err)
	assert.Equal(t, 2, count)
}

func TestFilteredSearch(t *testing.T) {
	f, err := os.CreateTemp("", "diskann_filter_test_*.bin")
	require.NoError(t, err)
	defer os.Remove(f.Name())
	defer f.Close()

	opts := Options{
		R:     4,
		L:     10,
		Alpha: 1.2,
	}

	w := NewWriter(f, nil, 123, 2, distance.MetricL2, opts)

	// Add vectors with metadata
	// 0: {1, 0}, type: "a"
	// 1: {0, 1}, type: "b"
	// 2: {1, 1}, type: "a"

	vectors := [][]float32{
		{1, 0},
		{0, 1},
		{1, 1},
	}

	mds := []map[string]interface{}{
		{"type": "a"},
		{"type": "b"},
		{"type": "a"},
	}

	for i, v := range vectors {
		doc, err := metadata.FromMap(mds[i])
		require.NoError(t, err)
		err = w.Add(model.ID(uint64(i)), v, doc, nil)
		require.NoError(t, err)
	}

	// Build and Write
	err = w.Write(context.Background())
	require.NoError(t, err)

	// Open reader
	st := blobstore.NewLocalStore(filepath.Dir(f.Name()))
	blob, err := st.Open(context.Background(), filepath.Base(f.Name()))
	require.NoError(t, err)
	s, err := Open(context.Background(), blob)
	require.NoError(t, err)
	defer s.Close()

	// Search with filter type="a"
	// Should return 0 and 2.
	// Query {1, 0} should match 0 best.

	filter := &metadata.FilterSet{
		Filters: []metadata.Filter{
			{Key: "type", Operator: metadata.OpEqual, Value: metadata.String("a")},
		},
	}

	ctx := context.Background()
	q := []float32{1, 0}

	// We need a searcher context
	sc := searcher.Get()
	defer searcher.Put(sc)
	sc.Heap.Reset(false) // L2 = ascending

	searchOpts := model.SearchOptions{
		Filter: filter,
	}

	err = s.Search(ctx, q, 10, nil, searchOpts, sc)
	require.NoError(t, err)

	require.Equal(t, 2, sc.Heap.Len())

	// Verify results using IDs (not RowIDs, which change due to BFS reordering)
	c1 := sc.Heap.Pop()
	c2 := sc.Heap.Pop()

	id1, _ := s.GetID(context.Background(), c1.RowID)
	id2, _ := s.GetID(context.Background(), c2.RowID)
	ids := []uint64{uint64(id1), uint64(id2)}
	assert.Contains(t, ids, uint64(0), "ID=0 (type=a) should be in results")
	assert.Contains(t, ids, uint64(2), "ID=2 (type=a) should be in results")
	assert.NotContains(t, ids, uint64(1), "ID=1 (type=b) should not be in results")
}
