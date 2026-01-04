package engine

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/hupe1980/vecgo/distance"
	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func getPK(t *testing.T, e *Engine, loc model.Location) uint64 {
	snap := e.current.Load()
	var foundPK uint64
	var found bool

	// Helper to stop iteration
	stopErr := errors.New("stop")

	if loc.SegmentID == snap.active.ID() {
		err := snap.active.Iterate(func(rowID uint32, pk model.PrimaryKey, vec []float32, md metadata.Document, payload []byte) error {
			if rowID == uint32(loc.RowID) {
				foundPK = uint64(pk)
				found = true
				return stopErr
			}
			return nil
		})
		if err != nil && err != stopErr {
			require.NoError(t, err)
		}
	} else {
		seg, ok := snap.segments[loc.SegmentID]
		require.True(t, ok, "Segment %d not found", loc.SegmentID)
		err := seg.Iterate(func(rowID uint32, pk model.PrimaryKey, vec []float32, md metadata.Document, payload []byte) error {
			if rowID == uint32(loc.RowID) {
				foundPK = uint64(pk)
				found = true
				return stopErr
			}
			return nil
		})
		if err != nil && err != stopErr {
			require.NoError(t, err)
		}
	}
	require.True(t, found, "PK not found for location %v", loc)
	return foundPK
}

func TestTombstonePersistence(t *testing.T) {
	dir := t.TempDir()
	ctx := context.Background()

	// 1. Open Engine
	e, err := Open(dir, 2, distance.MetricL2)
	require.NoError(t, err)

	// 2. Insert data
	require.NoError(t, e.Insert(1, []float32{1.0, 0.0}, nil, nil))
	require.NoError(t, e.Insert(2, []float32{0.0, 1.0}, nil, nil))

	// 3. Flush to create L1 segment (Segment 0)
	require.NoError(t, e.Flush())

	// 4. Delete PK 1 (which is in Segment 0)
	require.NoError(t, e.Delete(1))

	// Verify deletion in memory
	cands, err := e.Search(ctx, []float32{1.0, 0.0}, 10)
	require.NoError(t, err)
	assert.Len(t, cands, 1)
	pk := getPK(t, e, cands[0].Loc)
	assert.Equal(t, uint64(2), pk)

	// 5. Close Engine (should persist tombstones)
	require.NoError(t, e.Close())

	// Verify .tomb file exists
	// Segment ID should be 0 (first segment)
	tombPath := filepath.Join(dir, "segment_0.tomb")
	_, err = os.Stat(tombPath)
	require.NoError(t, err, "tombstone file should exist")

	// 6. Reopen Engine
	e2, err := Open(dir, 2, distance.MetricL2)
	require.NoError(t, err)
	defer e2.Close()

	// 7. Verify deletion persists
	cands2, err := e2.Search(ctx, []float32{1.0, 0.0}, 10)
	require.NoError(t, err)
	assert.Len(t, cands2, 1)
	pk2 := getPK(t, e2, cands2[0].Loc)
	assert.Equal(t, uint64(2), pk2)
}
