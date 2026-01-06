package engine_test

import (
	"fmt"
	"testing"

	"github.com/hupe1980/vecgo/distance"
	"github.com/hupe1980/vecgo/engine"
	"github.com/hupe1980/vecgo/internal/fs"
	"github.com/hupe1980/vecgo/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestEngine_Flush_DiskFull(t *testing.T) {
	// Setup FaultyFS with no initial limit
	baseFS := fs.LocalFS{}
	faultyFS := fs.NewFaultyFS(baseFS)
	faultyFS.Err = fmt.Errorf("fake disk full")

	dir := t.TempDir()

	opts := []engine.Option{
		engine.WithFileSystem(faultyFS),
	}

	e, err := engine.Open(dir, 128, distance.MetricL2, opts...)
	require.NoError(t, err)
	defer e.Close()

	// Insert some data to generate WAL traffic and fill memtable
	vec := make([]float32, 128)
	for i := 0; i < 100; i++ {
		err := e.Insert(model.PKString(fmt.Sprintf("id-%d", i)), vec, nil, nil)
		require.NoError(t, err)
	}

	// Capture current written bytes
	writtenBefore := faultyFS.GetWritten()
	t.Logf("Bytes written before flush (WAL): %d", writtenBefore)

	// Set limit to trigger failure during Flush
	// Flush writes segment file + payload + manifest update.
	// We want it to fail partway through.
	// Let's set it to fail after a small amount of additional bytes.
	faultyFS.SetLimit(writtenBefore + 1024) // Allow 1KB more, then fail

	// Trigger Flush
	err = e.Flush()

	// Expect error
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "fake disk full")

	t.Logf("Flush failed as expected: %v", err)

	// Reset limit to allow cleanup
	faultyFS.SetLimit(0)
}

func TestEngine_Compaction_DiskFull(t *testing.T) {
	// Setup FaultyFS
	baseFS := fs.LocalFS{}
	faultyFS := fs.NewFaultyFS(baseFS)
	faultyFS.Err = fmt.Errorf("fake disk full during compaction")

	dir := t.TempDir()

	// Disable auto flush/compaction to control them manually
	opts := []engine.Option{
		engine.WithFileSystem(faultyFS),
		engine.WithFlushConfig(engine.FlushConfig{MaxMemTableSize: 1024 * 1024 * 10}), // Large memtable
	}

	e, err := engine.Open(dir, 128, distance.MetricL2, opts...)
	require.NoError(t, err)
	defer e.Close()

	// 1. Create two segments
	vec := make([]float32, 128)

	// Segment 1
	for i := 0; i < 50; i++ {
		e.Insert(model.PKString(fmt.Sprintf("seg1-%d", i)), vec, nil, nil)
	}
	require.NoError(t, e.Flush())

	// Segment 2
	for i := 0; i < 50; i++ {
		e.Insert(model.PKString(fmt.Sprintf("seg2-%d", i)), vec, nil, nil)
	}
	require.NoError(t, e.Flush())

	// Measure written so far
	writtenBefore := faultyFS.GetWritten()
	t.Logf("Bytes written before compaction: %d", writtenBefore)

	// Set limit to fail during compaction
	faultyFS.SetLimit(writtenBefore + 500) // Fail quickly

	// Trigger Compaction
	infos := e.SegmentInfo()
	var segIDs []model.SegmentID
	for _, info := range infos {
		segIDs = append(segIDs, info.ID)
	}
	require.Len(t, segIDs, 2)

	err = e.Compact(segIDs)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "fake disk full")

	t.Logf("Compaction failed as expected: %v", err)

	// Reset limit
	faultyFS.SetLimit(0)
}
