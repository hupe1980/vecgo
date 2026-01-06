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
	e, err := Open(dir, 2, distance.MetricL2)
	require.NoError(t, err)
	defer e.Close()

	// Insert 10 items
	pks := make([]model.PK, 10)
	for i := 0; i < 10; i++ {
		pks[i] = model.PKUint64(uint64(i))
		require.NoError(t, e.Insert(pks[i], []float32{float32(i), float32(i)}, nil, nil))
	}

	// Flush to create a segment (items 0-9 in segment)
	require.NoError(t, e.Flush())

	// Insert 5 more items (items 10-14 in MemTable)
	morePKs := make([]model.PK, 5)
	for i := 0; i < 5; i++ {
		morePKs[i] = model.PKUint64(uint64(10 + i))
		require.NoError(t, e.Insert(morePKs[i], []float32{float32(10 + i), float32(10 + i)}, nil, nil))
	}

	// Batch Delete:
	// - 2 from Segment (0, 1)
	// - 2 from MemTable (10, 11)
	// - 1 non-existent (99)
	toDelete := []model.PK{
		model.PKUint64(0),
		model.PKUint64(1),
		model.PKUint64(10),
		model.PKUint64(11),
		model.PKUint64(99),
	}

	require.NoError(t, e.BatchDelete(toDelete))

	// Verify deletions
	for _, pk := range toDelete {
		if pk == model.PKUint64(99) {
			continue
		}
		_, err := e.Get(pk)
		require.Error(t, err) // Should be not found
	}

	// Verify others still exist
	// Segment: 2-9
	for i := 2; i < 10; i++ {
		_, err := e.Get(model.PKUint64(uint64(i)))
		require.NoError(t, err)
	}
	// MemTable: 12-14
	for i := 12; i < 15; i++ {
		_, err := e.Get(model.PKUint64(uint64(i)))
		require.NoError(t, err)
	}

	// Verify count via Scan
	count := 0
	for _, err := range e.Scan(context.Background()) {
		require.NoError(t, err)
		count++
	}
	// Total inserted: 15. Deleted: 4. Remaining: 11.
	require.Equal(t, 11, count)
}
