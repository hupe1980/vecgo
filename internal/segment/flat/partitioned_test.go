package flat

import (
	"context"
	"math/rand"
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

func TestPartitionedSegment(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "partitioned.bin")

	f, err := os.Create(path)
	require.NoError(t, err)

	// Force 4 partitions
	w := NewWriter(f, nil, 1, 2, distance.MetricL2, 4, QuantizationNone)

	// Generate clusters
	// Cluster 0: around (0,0)
	// Cluster 1: around (10,10)
	// Cluster 2: around (20,20)
	// Cluster 3: around (30,30)

	for i := 0; i < 100; i++ {
		cluster := i % 4
		base := float32(cluster * 10)
		vec := []float32{base + rand.Float32(), base + rand.Float32()}
		err := w.Add(model.PKUint64(uint64(i)), vec, nil, nil)
		require.NoError(t, err)
	}

	err = w.Flush()
	require.NoError(t, err)
	f.Close()

	// Open
	st := blobstore.NewLocalStore(dir)
	blob, err := st.Open("partitioned.bin")
	require.NoError(t, err)
	seg, err := Open(blob)
	require.NoError(t, err)
	defer seg.Close()

	assert.Equal(t, 4, seg.numPartitions)
	assert.Equal(t, 4*2, len(seg.centroids)) // 4 centroids * 2 dim

	// Search near (0,0) with nprobes=1
	// Should find vectors from Cluster 0
	q := []float32{0.1, 0.1}
	opts := model.SearchOptions{
		K:       10,
		NProbes: 1,
	}

	s := searcher.Get()
	defer searcher.Put(s)
	s.Heap.Reset(false)

	err = seg.Search(context.Background(), q, 10, nil, opts, s)
	require.NoError(t, err)

	// Verify candidates are from Cluster 0
	// Distance to (0,0) should be small (< 4.0)
	// Distance to (10,10) is > 100
	for s.Heap.Len() > 0 {
		c := s.Heap.Pop()
		assert.Less(t, c.Score, float32(8.0))
	}

	// Search near (30,30)
	q = []float32{30.1, 30.1}
	s.Heap.Reset(false)
	err = seg.Search(context.Background(), q, 10, nil, opts, s)
	require.NoError(t, err)
	for s.Heap.Len() > 0 {
		c := s.Heap.Pop()
		assert.Less(t, c.Score, float32(8.0))
	}
}
