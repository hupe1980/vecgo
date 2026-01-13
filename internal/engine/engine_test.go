package engine

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/hupe1980/vecgo/distance"
	"github.com/hupe1980/vecgo/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestEngine(t *testing.T) {
	dir := t.TempDir()

	// 1. Open Engine
	e, err := Open(dir, 2, distance.MetricL2)
	require.NoError(t, err)

	// 2. Insert
	id1, err := e.Insert(context.Background(), []float32{1.0, 0.0}, nil, nil)
	require.NoError(t, err)
	id2, err := e.Insert(context.Background(), []float32{0.0, 1.0}, nil, nil)
	require.NoError(t, err)

	// 3. Search
	ctx := context.Background()
	cands, err := e.Search(ctx, []float32{1.0, 0.0}, 10)
	require.NoError(t, err)
	assert.Len(t, cands, 2)

	// 4. Close
	err = e.Commit(context.Background())
	require.NoError(t, err)
	err = e.Close()
	require.NoError(t, err)

	// 5. Reopen (Persistence Check)
	e2, err := Open(dir, 2, distance.MetricL2)
	require.NoError(t, err)
	defer e2.Close()

	cands2, err := e2.Search(ctx, []float32{1.0, 0.0}, 10)
	require.NoError(t, err)
	assert.Len(t, cands2, 2)

	// 6. Update (Delete + Insert)
	// For Auto-Inc ID, Update is just another Insert with a new ID,
	// unless we explicitly delete the old one. This test logic changes.
	// If we want to simulate "update", we delete and insert.
	err = e2.Delete(context.Background(), id1)
	require.NoError(t, err)
	id1New, err := e2.Insert(context.Background(), []float32{0.0, 0.0}, nil, nil) // Move 1 to origin
	require.NoError(t, err)
	_ = id1New

	cands3, err := e2.Search(ctx, []float32{0.0, 0.0}, 10)
	require.NoError(t, err)
	assert.Len(t, cands3, 2)

	// 7. Commit
	err = e2.Commit(context.Background())
	require.NoError(t, err)

	// 8. Search after Flush
	cands4, err := e2.Search(ctx, []float32{0.0, 0.0}, 10)
	require.NoError(t, err)
	assert.Len(t, cands4, 2)

	// 9. Restart after Flush (verify manifest and segments loaded)
	err = e2.Close()
	require.NoError(t, err)

	e3, err := Open(dir, 2, distance.MetricL2)
	require.NoError(t, err)
	defer e3.Close()

	cands5, err := e3.Search(ctx, []float32{0.0, 0.0}, 10)
	require.NoError(t, err)
	assert.Len(t, cands5, 2)

	// 10. Delete
	err = e3.Delete(context.Background(), id2)
	require.NoError(t, err)

	cands6, err := e3.Search(ctx, []float32{0.0, 1.0}, 10)
	require.NoError(t, err)
	assert.Len(t, cands6, 1)

	// 11. Compact
	// Insert 3 and Commit to create Seg 3
	_, err = e3.Insert(context.Background(), []float32{1.0, 1.0}, nil, nil)
	require.NoError(t, err)
	err = e3.Commit(context.Background())
	require.NoError(t, err)

	// Get current segments to compact
	snap := e3.current.Load()
	var segmentsToCompact []model.SegmentID
	for id := range snap.segments {
		segmentsToCompact = append(segmentsToCompact, id)
	}
	require.Len(t, segmentsToCompact, 3)

	err = e3.Compact(segmentsToCompact, 1)
	require.NoError(t, err)

	// Verify Search
	cands7, err := e3.Search(ctx, []float32{0.0, 0.0}, 10)
	require.NoError(t, err)
	// Should find 1 and 3.
	assert.Len(t, cands7, 2)

	// Verify old segments are gone
	for _, id := range segmentsToCompact {
		filename := filepath.Join(dir, fmt.Sprintf("segment_%d.bin", id))
		_, err := os.Stat(filename)
		assert.True(t, os.IsNotExist(err), "segment %d should be deleted", id)

		payloadFilename := filepath.Join(dir, fmt.Sprintf("segment_%d.payload", id))
		_, err = os.Stat(payloadFilename)
		assert.True(t, os.IsNotExist(err), "segment %d payload should be deleted", id)
	}
}
