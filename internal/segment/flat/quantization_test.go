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

func TestQuantizedSegment(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "quantized.bin")

	f, err := os.Create(path)
	require.NoError(t, err)

	// Enable quantization
	w := NewWriter(f, nil, 1, 2, distance.MetricL2, 0, QuantizationSQ8)

	// Add some vectors
	vectors := [][]float32{
		{1.0, 0.0},
		{0.0, 1.0},
		{0.5, 0.5},
		{10.0, 10.0},
	}

	for i, vec := range vectors {
		err := w.Add(model.PrimaryKey(i), vec, nil, nil)
		require.NoError(t, err)
	}

	err = w.Flush()
	require.NoError(t, err)
	f.Close()

	// Open
	st := blobstore.NewLocalStore(dir)
	blob, err := st.Open("quantized.bin")
	require.NoError(t, err)
	seg, err := Open(blob)
	require.NoError(t, err)
	defer seg.Close()

	assert.NotNil(t, seg.sq)
	assert.NotEmpty(t, seg.codes)

	// Search
	q := []float32{0.9, 0.1}
	opts := model.SearchOptions{K: 2}

	s := searcher.Get()
	defer searcher.Put(s)
	s.Heap.Reset(false)

	err = seg.Search(context.Background(), q, 2, nil, opts, s)
	require.NoError(t, err)

	// Verify candidates have Approx=true
	cands := make([]model.Candidate, 0, s.Heap.Len())
	for s.Heap.Len() > 0 {
		cands = append(cands, s.Heap.Pop())
	}
	for _, c := range cands {
		assert.True(t, c.Approx)
	}

	// Rerank
	scored, err := seg.Rerank(context.Background(), q, cands, nil)
	require.NoError(t, err)

	// Verify scored rows match expected
	// Closest to (0.9, 0.1) should be (1.0, 0.0) -> ID 0
	// Second closest (0.5, 0.5) -> ID 2
	// (0.0, 1.0) -> ID 1
	// (10, 10) -> ID 3

	// Sort scored by score
	// (Assuming Rerank returns in order or we sort them)
	// Rerank implementation in segment.go just returns them in order of input candidates.
	// Search implementation returns all candidates (linear scan), not sorted.
	// So we need to find the best ones.

	bestID := -1
	bestScore := float32(1000.0)

	for _, s := range scored {
		if s.Score < bestScore {
			bestScore = s.Score
			bestID = int(s.Loc.RowID)
		}
	}

	assert.Equal(t, 0, bestID)
	assert.Less(t, bestScore, float32(0.1))
}
