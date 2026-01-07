package flat

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/hupe1980/vecgo/blobstore"
	"github.com/hupe1980/vecgo/distance"
	"github.com/hupe1980/vecgo/internal/searcher"
	"github.com/hupe1980/vecgo/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestPQSegment(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "pq.bin")

	f, err := os.Create(path)
	require.NoError(t, err)

	// Enable PQ
	// Dim 16, M=4 (4 subvectors of 4 dims)
	dim := 16
	w := NewWriter(f, nil, 1, dim, distance.MetricL2, 0, QuantizationPQ)
	w.SetPQConfig(4)

	// Add some vectors
	vectors := [][]float32{
		make([]float32, dim), // Zero vector
	}
	// Fill with some data
	// Add more random vectors to train PQ (need at least 256 for K=256, or less if K is adapted)
	// PQ implementation uses K=256 fixed in Writer.
	// So we need at least 256 vectors to train properly?
	// TrainKMeans handles fewer vectors by reducing K?
	// No, TrainKMeans returns error if n < k.
	// But PQ implementation in pq.go handles it?
	// pq.go: Train calls kmeans.
	// If we have fewer vectors than K, kmeans might fail or return fewer centroids.
	// Let's add 300 vectors.

	for i := 0; i < 300; i++ {
		vec := make([]float32, dim)
		for j := range vec {
			vec[j] = float32(i+j) * 0.01
		}
		vectors = append(vectors, vec)
	}

	// Add the zero vector at index 0
	// Actually I already added it.

	for i, vec := range vectors {
		err := w.Add(model.PKUint64(uint64(i)), vec, nil, nil)
		require.NoError(t, err)
	}

	err = w.Flush()
	require.NoError(t, err)
	f.Close()

	// Open segment
	st := blobstore.NewLocalStore(dir)
	blob, err := st.Open(context.Background(), "pq.bin")
	require.NoError(t, err)
	seg, err := Open(blob)
	require.NoError(t, err)
	defer seg.Close()

	// Search
	q := make([]float32, dim) // Zero query
	opts := model.SearchOptions{K: 5}

	s := searcher.Get()
	defer searcher.Put(s)
	s.Heap.Reset(false)

	err = seg.Search(context.Background(), q, 5, nil, opts, s)
	require.NoError(t, err)

	// Should find the zero vector (ID 0) first
	// Note: ID 0 is the zero vector.
	// Heap pops worst to best.
	var last model.Candidate
	for s.Heap.Len() > 0 {
		last = s.Heap.Pop()
	}
	assert.Equal(t, model.RowID(0), last.Loc.RowID)
	assert.Less(t, last.Score, float32(0.1)) // Should be very close to 0
}
