package flat

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/hupe1980/vecgo/blobstore"
	"github.com/hupe1980/vecgo/distance"
	"github.com/hupe1980/vecgo/internal/searcher"
	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestBlockSkipping(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "block_skipping.bin")

	f, err := os.Create(path)
	require.NoError(t, err)

	// Use small block size for testing?
	// BlockSize is constant 1024.
	// So I need to write > 1024 rows to test skipping.
	// Let's write 2048 rows (2 blocks).
	// Block 0: price [0, 100]
	// Block 1: price [200, 300]

	w := NewWriter(f, nil, 1, 2, distance.MetricL2, 0, QuantizationNone)

	// Block 0
	for i := 0; i < 1024; i++ {
		md := metadata.Document{
			"price": metadata.Value{Kind: metadata.KindFloat, F64: float64(i % 100)},
		}
		err := w.Add(model.ID(uint64(i)), []float32{0.0, 0.0}, md, nil)
		require.NoError(t, err)
	}

	// Block 1
	for i := 1024; i < 2048; i++ {
		md := metadata.Document{
			"price": metadata.Value{Kind: metadata.KindFloat, F64: float64(200 + (i % 100))},
		}
		err := w.Add(model.ID(uint64(i)), []float32{1.0, 1.0}, md, nil) // Different vector to distinguish
		require.NoError(t, err)
	}

	err = w.Flush()
	require.NoError(t, err)
	f.Close()

	// Open
	st := blobstore.NewLocalStore(dir)
	blob, err := st.Open(context.Background(), "block_skipping.bin")
	require.NoError(t, err)
	seg, err := Open(blob)
	require.NoError(t, err)
	defer seg.Close()

	ctx := context.Background()
	q := []float32{0.0, 0.0}

	// Case 1: Filter price > 150. Should skip Block 0.
	// Block 0 max price is 99.
	// Block 1 min price is 200.
	// So only Block 1 matches.

	filter := &metadata.FilterSet{
		Filters: []metadata.Filter{
			{
				Key:      "price",
				Operator: metadata.OpGreaterThan,
				Value:    metadata.Value{Kind: metadata.KindFloat, F64: 150.0},
			},
		},
	}

	s := searcher.Get()
	defer searcher.Put(s)
	s.Heap.Reset(false)

	err = seg.Search(ctx, q, 10, nil, model.SearchOptions{Filter: filter}, s)
	require.NoError(t, err)

	// Should only find rows from Block 1 (ids >= 1024)
	for s.Heap.Len() > 0 {
		c := s.Heap.Pop()
		assert.GreaterOrEqual(t, int(c.RowID), 1024)
	}

	// Case 2: Filter price < 50. Should skip Block 1.
	filter2 := &metadata.FilterSet{
		Filters: []metadata.Filter{
			{
				Key:      "price",
				Operator: metadata.OpLessThan,
				Value:    metadata.Value{Kind: metadata.KindFloat, F64: 50.0},
			},
		},
	}

	s.Heap.Reset(false)
	err = seg.Search(ctx, q, 10, nil, model.SearchOptions{Filter: filter2}, s)
	require.NoError(t, err)

	// Should only find rows from Block 0 (ids < 1024)
	for s.Heap.Len() > 0 {
		c := s.Heap.Pop()
		assert.Less(t, int(c.RowID), 1024)
	}
}
