package engine

import (
	"path/filepath"
	"testing"

	"github.com/hupe1980/vecgo/distance"
	"github.com/hupe1980/vecgo/internal/manifest"
	"github.com/hupe1980/vecgo/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestWALRotation(t *testing.T) {
	dir := t.TempDir()
	dim := 4

	e, err := Open(dir, dim, distance.MetricL2)
	require.NoError(t, err)

	// Initial state: WALID should be 0, file should be wal.log
	assert.FileExists(t, filepath.Join(dir, "wal.log"))
	assert.NoFileExists(t, filepath.Join(dir, "wal_1.log"))

	// Insert some data
	err = e.Insert(model.PKUint64(1), []float32{1, 2, 3, 4}, nil, nil)
	require.NoError(t, err)

	// Force Flush
	err = e.Flush()
	require.NoError(t, err)

	// Check rotation
	// wal.log should be gone
	assert.NoFileExists(t, filepath.Join(dir, "wal.log"))
	// wal_1.log should exist
	assert.FileExists(t, filepath.Join(dir, "wal_1.log"))

	// Check Manifest
	mStore := manifest.NewStore(nil, dir)
	m, err := mStore.Load()
	require.NoError(t, err)
	assert.Equal(t, uint64(1), m.WALID)

	// Insert more data (goes to wal_1.log)
	err = e.Insert(model.PKUint64(2), []float32{5, 6, 7, 8}, nil, nil)
	require.NoError(t, err)

	// Flush again
	err = e.Flush()
	require.NoError(t, err)

	// Check rotation
	// wal_1.log should be gone
	assert.NoFileExists(t, filepath.Join(dir, "wal_1.log"))
	// wal_2.log should exist
	assert.FileExists(t, filepath.Join(dir, "wal_2.log"))

	// Check Manifest
	m, err = mStore.Load()
	require.NoError(t, err)
	assert.Equal(t, uint64(2), m.WALID)

	e.Close()

	// Reopen and verify
	e2, err := Open(dir, dim, distance.MetricL2)
	require.NoError(t, err)
	defer e2.Close()

	// Should be able to read both vectors
	// PK 1 is in segment (flushed)
	// PK 2 is in segment (flushed)
	// Wait, both flushed.

	// Let's insert one more and NOT flush, to verify it's in WAL
	err = e2.Insert(model.PKUint64(3), []float32{9, 10, 11, 12}, nil, nil)
	require.NoError(t, err)

	// Close e2
	e2.Close()

	// Reopen e3
	e3, err := Open(dir, dim, distance.MetricL2)
	require.NoError(t, err)
	defer e3.Close()

	// Verify PK 3 exists (recovered from wal_2.log)
	_, err = e3.Get(model.PKUint64(3))
	assert.NoError(t, err)
}
