package engine_test

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/hupe1980/vecgo/distance"
	"github.com/hupe1980/vecgo/engine"
	"github.com/hupe1980/vecgo/internal/fs"
	"github.com/hupe1980/vecgo/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRecovery_TruncatedWAL(t *testing.T) {
	dir := t.TempDir()

	// 1. Initial Insert
	{
		e, err := engine.Open(dir, 2, distance.MetricL2)
		require.NoError(t, err)

		err = e.Insert(model.PKUint64(1), []float32{1.0, 1.0}, nil, nil)
		require.NoError(t, err)
		err = e.Insert(model.PKUint64(2), []float32{2.0, 2.0}, nil, nil)
		require.NoError(t, err)

		require.NoError(t, e.Close())
	}

	// 2. Corrupt the WAL (append garbage)
	// Find WAL file. Assuming ID 1 or similar.
	matches, err := filepath.Glob(filepath.Join(dir, "wal_*.log"))
	require.NoError(t, err)
	if len(matches) == 0 {
		matches, err = filepath.Glob(filepath.Join(dir, "wal.log"))
		require.NoError(t, err)
	}
	require.NotEmpty(t, matches, "WAL file not found")
	walPath := matches[0]

	f, err := os.OpenFile(walPath, os.O_APPEND|os.O_WRONLY, 0644)
	require.NoError(t, err)
	_, err = f.Write([]byte("garbage_data_at_the_end_of_wal"))
	require.NoError(t, err)
	f.Close()

	// 3. Reopen and Verify
	{
		e, err := engine.Open(dir, 2, distance.MetricL2)
		// Should not fail, just truncate
		require.NoError(t, err, "Engine should recover from truncated WAL")
		defer e.Close()

		// Verify data exists
		res, err := e.Get(model.PKUint64(1))
		require.NoError(t, err)
		assert.NotNil(t, res)

		res, err = e.Get(model.PKUint64(2))
		require.NoError(t, err)
		assert.NotNil(t, res)
	}
}

func TestRecovery_CrashDuringFlush_ManifestNotUpdated(t *testing.T) {
	// Scenario: Segment file is written, but Manifest update fails (crash).
	// On restart, the orphan segment should be ignored, and data should be recovered from WAL.

	// We use FaultyFS to simulate this, but simpler:
	// 1. Insert data.
	// 2. Manually create a "fake" segment file that implies a flush happened but manifest didn't know.
	// 3. Reopen.
	// Actually, if we just create a file, the engine cleans it up.
	// To verify *recovery*, we must ensure the WAL wasn't deleted either.
	// WAL deletion usually happens *after* manifest update. So if manifest update crashes, WAL exists.

	dir := t.TempDir()

	// 1. Insert data
	{
		e, err := engine.Open(dir, 2, distance.MetricL2)
		require.NoError(t, err)

		// Insert enough to verify
		err = e.Insert(model.PKUint64(1), []float32{1.0, 1.0}, nil, nil)
		require.NoError(t, err)

		require.NoError(t, e.Close())
	}

	// 2. Create an orphan segment file
	orphanPath := filepath.Join(dir, "segment_9999.seg")
	err := os.WriteFile(orphanPath, []byte("garbage_segment_content"), 0644)
	require.NoError(t, err)

	// 3. Reopen
	{
		e, err := engine.Open(dir, 2, distance.MetricL2)
		require.NoError(t, err)
		defer e.Close()

		// Verify data is still there (from WAL)
		res, err := e.Get(model.PKUint64(1))
		require.NoError(t, err)
		assert.NotNil(t, res)

		// Verify orphan is gone (cleanup)
		_, err = os.Stat(orphanPath)
		assert.True(t, os.IsNotExist(err), "Orphan segment should be deleted")
	}
}

func TestRecovery_FaultyFS_FlushCrash(t *testing.T) {
	// A more realistic test using FaultyFS to crash exactly during flush.
	// This ensures our atomicity logic (write temp, fsync, rename) works.

	dir := t.TempDir()

	realFS := fs.LocalFS{}
	faultyFS := fs.NewFaultyFS(realFS)

	// We need to find the byte limit where Flush writes the segment but fails strictly before manifest update.
	// This is hard to predict exactly without introspection.
	// Instead, we can simulate "Write to Manifest fails".
	// FaultyFS usually fails on Write. Manifest is written via `manifest.tmp` then renamed.

	opts := []engine.Option{
		engine.WithFileSystem(faultyFS),
	}

	// 1. Run until flush
	{
		e, err := engine.Open(dir, 2, distance.MetricL2, opts...)
		require.NoError(t, err)

		e.Insert(model.PKUint64(1), []float32{1.0, 1.0}, nil, nil)

		// Calculate stats to guess where to fail?
		// Or easier: Run a flush, check size, then run again with limit.
		// Let's rely on the fact that if we fail ANYWHERE during flush, the system must recover.

		// Set a limit that will likely let some writes pass but fail eventually.
		// Flush writes: Segment (~KB), Payload (~KB), Tombstones, Manifest.
		// If we fail during segment write -> trash temp file.
		// If we fail during manifest write -> trash temp manifest.
		// In all cases, on reopen, we should rely on WAL.

		faultyFS.SetLimit(100) // Very low limit, should fail almost immediately during flush write
		faultyFS.Err = fmt.Errorf("injected IO error")

		err = e.Flush()
		require.Error(t, err) // Flush failed

		// Don't close cleanly, simulating crash (Close might try to flush)
		// Accessing private fields to stop workers would be ideal but can't here.
		// We just assume process dies.
		// But in unit test, e is still open. We can't really "kill" it without Close().
		// If we Close(), and it tries to flush again?
		// We just drop 'e'.
	}

	// Scan dir to see what mess we left
	// Likely some temp files.

	// 2. Recovery
	// Reset FaultyFS or use a clean one
	faultyFS.SetLimit(0) // No limit

	{
		e, err := engine.Open(dir, 2, distance.MetricL2, opts...)
		require.NoError(t, err, "Should recover after failed flush")
		defer e.Close()

		// Data must be there from WAL
		res, err := e.Get(model.PKUint64(1))
		require.NoError(t, err)
		assert.NotNil(t, res)
	}
}
