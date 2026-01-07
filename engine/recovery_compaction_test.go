package engine_test

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/hupe1980/vecgo/distance"
	"github.com/hupe1980/vecgo/engine"
	"github.com/hupe1980/vecgo/internal/fs"
	"github.com/hupe1980/vecgo/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// CrashFS wraps a FileSystem and injects errors based on filename patterns.
type CrashFS struct {
	fs.FileSystem
	CrashOnPattern string
	crashed        bool
}

func (c *CrashFS) OpenFile(name string, flag int, perm os.FileMode) (fs.File, error) {
	if c.CrashOnPattern != "" && strings.Contains(filepath.Base(name), c.CrashOnPattern) && (flag&os.O_CREATE != 0 || flag&os.O_WRONLY != 0) {
		c.crashed = true
		return nil, fmt.Errorf("injected crash on file: %s", name)
	}
	// Also fail on Rename if pattern matches
	return c.FileSystem.OpenFile(name, flag, perm)
}

func (c *CrashFS) Rename(oldpath, newpath string) error {
	if c.CrashOnPattern != "" && (strings.Contains(filepath.Base(newpath), c.CrashOnPattern) || strings.Contains(filepath.Base(oldpath), c.CrashOnPattern)) {
		c.crashed = true
		return fmt.Errorf("injected crash on rename: %s -> %s", oldpath, newpath)
	}
	return c.FileSystem.Rename(oldpath, newpath)
}

// Ensure CrashFS satisfies interface
var _ fs.FileSystem = &CrashFS{}

func TestRecovery_CrashDuringCompaction_ManifestUpdate(t *testing.T) {
	dir := t.TempDir()

	// Use manual fs for first setup
	realFS := fs.LocalFS{}

	// 1. Setup: Create 2 flushable segments
	{
		e, err := engine.Open(dir, 2, distance.MetricL2)
		require.NoError(t, err)

		// Seg 1
		e.Insert(model.PKUint64(1), []float32{1, 1}, nil, nil)
		e.Flush() // Created segment 1

		// Seg 2
		e.Insert(model.PKUint64(2), []float32{2, 2}, nil, nil)
		e.Flush() // Created segment 2

		e.Close()
	}

	// 2. Crash during manifest update (compaction commit)
	crashFS := &CrashFS{FileSystem: realFS, CrashOnPattern: "MANIFEST"}
	// Manifest is updated by renaming "manifest.tmp" to "manifest.json" (or similar).
	// Failing on "manifest" should catch the temp file creation or the rename.

	{
		opts := []engine.Option{engine.WithFileSystem(crashFS)}
		e, err := engine.Open(dir, 2, distance.MetricL2, opts...)
		require.NoError(t, err)

		// Trigger Compaction
		// We use IDs 0 and 1.
		err = e.Compact([]model.SegmentID{0, 1})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "injected crash")

		e.Close()
	}

	// 3. Recovery: Should see old segments 1 and 2
	{
		e, err := engine.Open(dir, 2, distance.MetricL2)
		require.NoError(t, err)
		defer e.Close()

		v1, err := e.Get(model.PKUint64(1))
		require.NoError(t, err)
		assert.NotNil(t, v1)

		v2, err := e.Get(model.PKUint64(2))
		require.NoError(t, err)
		assert.NotNil(t, v2)
	}
}

func TestRecovery_CrashDuringCompaction_SegmentWrite(t *testing.T) {
	dir := t.TempDir()
	realFS := fs.LocalFS{}

	// 1. Setup
	{
		e, err := engine.Open(dir, 2, distance.MetricL2)
		require.NoError(t, err)
		e.Insert(model.PKUint64(1), []float32{1, 1}, nil, nil)
		e.Flush()
		e.Insert(model.PKUint64(2), []float32{2, 2}, nil, nil)
		e.Flush()
		e.Close()
	}

	// 2. Crash during merged segment write
	// Merged segment usually has a new ID.
	crashFS := &CrashFS{FileSystem: realFS, CrashOnPattern: ".bin"}

	{
		opts := []engine.Option{engine.WithFileSystem(crashFS)}
		e, err := engine.Open(dir, 2, distance.MetricL2, opts...)
		require.NoError(t, err)

		// The new segment write (CREATE/WRITE) should fail.
		// Old segments open (READ) should pass.
		// CrashFS logic: OpenFile checks flag&O_CREATE || flag&O_WRONLY.

		err = e.Compact([]model.SegmentID{0, 1})
		require.Error(t, err)

		e.Close()
	}

	// 3. Recovery
	{
		e, err := engine.Open(dir, 2, distance.MetricL2)
		require.NoError(t, err)
		defer e.Close()

		v1, err := e.Get(model.PKUint64(1))
		require.NoError(t, err)
		assert.NotNil(t, v1)
	}
}
