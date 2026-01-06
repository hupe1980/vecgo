package memtable

import (
	"context"
	"testing"

	"github.com/hupe1980/vecgo/distance"
	"github.com/hupe1980/vecgo/internal/searcher"
	"github.com/hupe1980/vecgo/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMemTable(t *testing.T) {
	mt, err := New(1, 2, distance.MetricL2, nil)
	require.NoError(t, err)
	defer mt.Close()

	// 1. Insert
	id1, err := mt.Insert(model.PKUint64(1), []float32{1.0, 0.0})
	require.NoError(t, err)
	assert.Equal(t, model.RowID(0), id1)

	id2, err := mt.Insert(model.PKUint64(2), []float32{0.0, 1.0})
	require.NoError(t, err)
	assert.Equal(t, model.RowID(1), id2)

	// 2. Search
	ctx := context.Background()
	q := []float32{1.0, 0.0}

	s := searcher.Get()
	defer searcher.Put(s)
	s.Heap.Reset(false)

	err = mt.Search(ctx, q, 10, nil, model.SearchOptions{}, s)
	require.NoError(t, err)
	assert.Equal(t, 2, s.Heap.Len())

	// 3. Delete
	mt.Delete(id1)

	s.Heap.Reset(false)
	err = mt.Search(ctx, q, 10, nil, model.SearchOptions{}, s)
	require.NoError(t, err)
	assert.Equal(t, 1, s.Heap.Len())
	assert.Equal(t, id2, s.Heap.Pop().Loc.RowID)
}
