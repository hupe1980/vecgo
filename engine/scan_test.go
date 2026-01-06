package engine

import (
	"context"
	"testing"

	"github.com/hupe1980/vecgo/distance"
	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/model"
	"github.com/stretchr/testify/require"
)

func TestScan(t *testing.T) {
	dir := t.TempDir()
	e, err := Open(dir, 2, distance.MetricL2)
	require.NoError(t, err)
	defer e.Close()

	// Insert data
	// 1. MemTable
	require.NoError(t, e.Insert(model.PKUint64(1), []float32{1.0, 0.0}, map[string]any{"cat": "A"}, nil))
	require.NoError(t, e.Insert(model.PKUint64(2), []float32{0.0, 1.0}, map[string]any{"cat": "B"}, nil))

	// 2. Flush to create segment
	require.NoError(t, e.Flush())

	// 3. Insert more data (MemTable)
	require.NoError(t, e.Insert(model.PKUint64(3), []float32{1.0, 1.0}, map[string]any{"cat": "A"}, nil))

	// 4. Delete one
	require.NoError(t, e.Delete(model.PKUint64(2)))

	// Scan all
	count := 0
	foundPKs := make(map[uint64]bool)

	for rec, err := range e.Scan(context.Background()) {
		require.NoError(t, err)
		u64, ok := rec.PK.Uint64()
		require.True(t, ok)
		foundPKs[u64] = true
		count++
	}

	require.Equal(t, 2, count)
	require.True(t, foundPKs[1])
	require.True(t, foundPKs[3])
	require.False(t, foundPKs[2]) // Deleted

	// Scan with Filter (cat == "A")
	valA, _ := metadata.FromAny("A")
	filter := &metadata.Filter{
		Key:      "cat",
		Operator: metadata.OpEqual,
		Value:    valA,
	}

	countA := 0
	for rec, err := range e.Scan(context.Background(), WithScanFilter(filter)) {
		require.NoError(t, err)
		_, ok := rec.PK.Uint64()
		require.True(t, ok)
		require.Equal(t, "A", rec.Metadata["cat"])
		countA++
	}
	require.Equal(t, 2, countA) // 1 and 3 are both "A"

	// Scan with Filter (cat == "B") - should be empty because 2 was deleted
	valB, _ := metadata.FromAny("B")
	filterB := &metadata.Filter{
		Key:      "cat",
		Operator: metadata.OpEqual,
		Value:    valB,
	}

	countB := 0
	for _, err := range e.Scan(context.Background(), WithScanFilter(filterB)) {
		require.NoError(t, err)
		countB++
	}
	require.Equal(t, 0, countB)
}
