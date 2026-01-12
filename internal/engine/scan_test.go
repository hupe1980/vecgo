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
	id1, err := e.Insert([]float32{1.0, 0.0}, metadata.Document{"cat": metadata.String("A")}, nil)
	require.NoError(t, err)
	id2, err := e.Insert([]float32{0.0, 1.0}, metadata.Document{"cat": metadata.String("B")}, nil)
	require.NoError(t, err)

	// 2. Flush to create segment
	require.NoError(t, e.Flush())

	// 3. Insert more data (MemTable)
	id3, err := e.Insert([]float32{1.0, 1.0}, metadata.Document{"cat": metadata.String("A")}, nil)
	require.NoError(t, err)

	// 4. Delete one
	require.NoError(t, e.Delete(id2))

	// Scan all
	count := 0
	foundIDs := make(map[model.ID]bool)

	for rec, err := range e.Scan(context.Background()) {
		require.NoError(t, err)
		foundIDs[rec.ID] = true
		count++
	}

	require.Equal(t, 2, count)
	require.True(t, foundIDs[id1])
	require.True(t, foundIDs[id3])
	require.False(t, foundIDs[id2]) // Deleted

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
		require.Equal(t, "A", rec.Metadata["cat"].StringValue())
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
