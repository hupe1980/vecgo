package engine

import (
	"context"
	"testing"

	"github.com/hupe1980/vecgo/distance"
	"github.com/hupe1980/vecgo/model"
	"github.com/stretchr/testify/require"
)

func TestBatchDelete(t *testing.T) {
	dir := t.TempDir()
	e, err := OpenLocal(context.Background(), dir, WithDimension(2), WithMetric(distance.MetricL2))
	require.NoError(t, err)
	defer e.Close()

	// Insert 10 items
	ids := make([]model.ID, 10)
	for i := 0; i < 10; i++ {
		id, err := e.Insert(context.Background(), []float32{float32(i), float32(i)}, nil, nil)
		require.NoError(t, err)
		ids[i] = id
	}

	// Flush to create a segment (items 0-9 in segment)
	require.NoError(t, e.Commit(context.Background()))

	// Insert 5 more items (items 10-14 in MemTable)
	moreIDs := make([]model.ID, 5)
	for i := 0; i < 5; i++ {
		id, err := e.Insert(context.Background(), []float32{float32(10 + i), float32(10 + i)}, nil, nil)
		require.NoError(t, err)
		moreIDs[i] = id
	}

	// Batch Delete:
	// - 2 from Segment (0, 1)
	// - 2 from MemTable (10, 11)
	// - 1 non-existent (99)
	toDelete := []model.ID{
		ids[0],
		ids[1],
		moreIDs[0],
		moreIDs[1],
		model.ID(99),
	}

	require.NoError(t, e.BatchDelete(context.Background(), toDelete))

	// Verify deletions
	for _, id := range toDelete {
		if id == model.ID(99) {
			continue
		}
		_, err := e.Get(context.Background(), id)
		require.Error(t, err) // Should be not found
	}

	// Verify others still exist
	// Segment: 2-9
	for i := 2; i < 10; i++ {
		_, err := e.Get(context.Background(), ids[i])
		require.NoError(t, err)
	}
	// MemTable: 12-14
	for i := 12; i < 15; i++ {
		_, err := e.Get(context.Background(), moreIDs[i-10])
		require.NoError(t, err)
	}

	// Verify count via Scan
	count := 0
	for _ = range e.Scan(context.Background()) { // Loop body receives rec?
		count++
	}
	// Total inserted: 15. Deleted: 4. Remaining: 11.
	require.Equal(t, 11, count)
}
