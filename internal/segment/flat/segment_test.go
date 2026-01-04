package flat

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/hupe1980/vecgo/blobstore"
	"github.com/hupe1980/vecgo/distance"
	"github.com/hupe1980/vecgo/model"
	"github.com/hupe1980/vecgo/searcher"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestFlatSegment(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "segment.bin")

	// 1. Write
	f, err := os.Create(path)
	require.NoError(t, err)

	w := NewWriter(f, nil, 1, 2, distance.MetricL2, 0, QuantizationNone)

	err = w.Add(1, []float32{1.0, 0.0}, nil, nil)
	require.NoError(t, err)
	err = w.Add(2, []float32{0.0, 1.0}, nil, nil)
	require.NoError(t, err)

	err = w.Flush()
	require.NoError(t, err)
	f.Close()

	// 2. Read
	st := blobstore.NewLocalStore(dir)
	blob, err := st.Open("segment.bin")
	require.NoError(t, err)

	seg, err := Open(blob)
	require.NoError(t, err)
	defer seg.Close()

	assert.Equal(t, uint32(2), seg.RowCount())

	// 3. Search
	ctx := context.Background()
	q := []float32{1.0, 0.0}

	s := searcher.Get()
	defer searcher.Put(s)
	s.Heap.Reset(false)

	err = seg.Search(ctx, q, 10, nil, model.SearchOptions{}, s)
	require.NoError(t, err)
	assert.Equal(t, 2, s.Heap.Len())

	var last model.Candidate
	for s.Heap.Len() > 0 {
		last = s.Heap.Pop()
	}
	assert.Equal(t, model.RowID(0), last.Loc.RowID) // Closest
}
