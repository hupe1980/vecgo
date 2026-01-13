package engine

import (
	"context"
	"testing"

	"github.com/hupe1980/vecgo/distance"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestBatchOperations(t *testing.T) {
	dir := t.TempDir()
	e, err := Open(dir, 2, distance.MetricL2)
	require.NoError(t, err)
	defer e.Close()

	// 1. BatchInsert
	vectors := [][]float32{
		{1.0, 0.0},
		{0.0, 1.0},
		{1.0, 1.0},
	}
	ids, err := e.BatchInsert(context.Background(), vectors, nil, nil)
	require.NoError(t, err)
	require.Len(t, ids, 3)

	// Verify with Get
	rec, err := e.Get(ids[0])
	require.NoError(t, err)
	assert.Equal(t, []float32{1.0, 0.0}, rec.Vector)
	assert.Equal(t, ids[0], rec.ID)

	// 2. BatchSearch
	ctx := context.Background()
	queries := [][]float32{
		{1.0, 0.0}, // Should find 1 first
		{0.0, 1.0}, // Should find 2 first
	}
	results, err := e.BatchSearch(ctx, queries, 2)
	require.NoError(t, err)
	require.Len(t, results, 2)

	assert.Equal(t, ids[0], results[0][0].ID)
	assert.Equal(t, ids[1], results[1][0].ID)

	// 3. SearchThreshold
	// Distance L2:
	// 1: {1,0} -> {1,0} dist 0
	// 2: {1,0} -> {0,1} dist sqrt(2) ~= 1.414
	// 3: {1,0} -> {1,1} dist 1

	// Threshold 0.5 should only return 1
	res, err := e.SearchThreshold(ctx, []float32{1.0, 0.0}, 0.5, 10)
	require.NoError(t, err)
	assert.Len(t, res, 1)
	assert.Equal(t, ids[0], res[0].ID)

	// Threshold 1.1 should return 1 and 3
	res, err = e.SearchThreshold(ctx, []float32{1.0, 0.0}, 1.1, 10)
	require.NoError(t, err)
	assert.Len(t, res, 2)
	// Order might vary if scores are equal, but here scores are 0 and 1.
	// Search returns top-k sorted.
	assert.Equal(t, ids[0], res[0].ID)
	assert.Equal(t, ids[2], res[1].ID)
}
