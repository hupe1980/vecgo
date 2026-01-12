package engine

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/hupe1980/vecgo/distance"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestTombstonePersistence(t *testing.T) {
	dir := t.TempDir()
	ctx := context.Background()

	// 1. Open Engine
	e, err := Open(dir, 2, distance.MetricL2)
	require.NoError(t, err)

	// 2. Insert data
	id1, err := e.Insert([]float32{1.0, 0.0}, nil, nil)
	require.NoError(t, err)
	id2, err := e.Insert([]float32{0.0, 1.0}, nil, nil)
	require.NoError(t, err)

	// 3. Flush to create L1 segment (Segment 0)
	require.NoError(t, e.Flush())

	// 4. Delete PK 1 (which is in Segment 0)
	require.NoError(t, e.Delete(id1))

	// Verify deletion in memory
	cands, err := e.Search(ctx, []float32{1.0, 0.0}, 10)
	require.NoError(t, err)
	assert.Len(t, cands, 1)
	assert.Equal(t, id2, cands[0].ID)

	// 5. Close Engine (should persist tombstones)
	require.NoError(t, e.Close())

	// Verify .tomb file exists
	// Segment ID might vary due to allocation strategy. Find valid tomb file.
	matches, err := filepath.Glob(filepath.Join(dir, "segment_*.tomb"))
	require.NoError(t, err)
	require.NotEmpty(t, matches, "tombstone file should exist")

	// 6. Reopen Engine
	e2, err := Open(dir, 2, distance.MetricL2)
	require.NoError(t, err)
	defer e2.Close()

	// 7. Verify deletion persists
	cands2, err := e2.Search(ctx, []float32{1.0, 0.0}, 10)
	require.NoError(t, err)
	assert.Len(t, cands2, 1)
	assert.Equal(t, id2, cands2[0].ID)
}
