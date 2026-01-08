package manifest

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/hupe1980/vecgo/internal/fs"
	"github.com/hupe1980/vecgo/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestManifestVersioning(t *testing.T) {
	dir := t.TempDir()
	store := NewStore(nil, dir)

	// 1. Save a manifest
	m := &Manifest{
		ID: 0,
	}
	err := store.Save(m)
	require.NoError(t, err)

	// 2. Load it
	loaded, err := store.Load()
	require.NoError(t, err)
	assert.Equal(t, CurrentVersion, loaded.Version)

	// 3. Corrupt version
	// Find the manifest file
	currentContent, err := os.ReadFile(filepath.Join(dir, CurrentFileName))
	require.NoError(t, err)
	manifestPath := filepath.Join(dir, string(currentContent))

	// Read JSON
	data, err := os.ReadFile(manifestPath)
	require.NoError(t, err)

	var raw map[string]interface{}
	err = json.Unmarshal(data, &raw)
	require.NoError(t, err)

	// Modify version
	raw["version"] = 999
	newData, err := json.Marshal(raw)
	require.NoError(t, err)

	// Write back
	err = os.WriteFile(manifestPath, newData, 0644)
	require.NoError(t, err)

	// 4. Load again - should fail
	_, err = store.Load()
	assert.Error(t, err)
	assert.ErrorIs(t, err, ErrIncompatibleVersion)
}

func TestStore(t *testing.T) {
	dir := t.TempDir()

	fsys := fs.LocalFS{}
	store := NewStore(fsys, dir)

	// 1. Load on empty -> default manifest
	m, err := store.Load()
	require.NoError(t, err)
	assert.Equal(t, uint64(0), m.ID)
	assert.Equal(t, CurrentVersion, m.Version)

	// 2. Save (increments ID)
	m.NextSegmentID = 100
	err = store.Save(m)
	require.NoError(t, err)
	assert.Equal(t, uint64(1), m.ID) // Incremented

	// 3. Load updated
	m2, err := store.Load()
	require.NoError(t, err)
	assert.Equal(t, uint64(1), m2.ID)
	assert.Equal(t, model.SegmentID(100), m2.NextSegmentID)

	// 4. Verify file structure
	matches, _ := filepath.Glob(filepath.Join(dir, "MANIFEST-*"))
	assert.GreaterOrEqual(t, len(matches), 1)

	// 5. Save another one
	err = store.Save(m)
	require.NoError(t, err)
	assert.Equal(t, uint64(2), m.ID)
	
	m3, err := store.Load()
	require.NoError(t, err)
	assert.Equal(t, uint64(2), m3.ID)
}

func TestStore_LoadErrors(t *testing.T) {
	dir := t.TempDir()
	fsys := fs.LocalFS{}
	store := NewStore(fsys, dir)

	// Create a corrupted CURRENT file
	currentPath := filepath.Join(dir, "CURRENT")
	err := os.WriteFile(currentPath, []byte("MANIFEST-999999.json"), 0644)
	require.NoError(t, err)

	_, err = store.Load()
	assert.Error(t, err) // Should fail to read the manifest file
}

type MockFS struct {
	fs.LocalFS
	FailOpenFile bool
	FailRename   bool
}

func (m MockFS) OpenFile(name string, flag int, perm os.FileMode) (fs.File, error) {
	if m.FailOpenFile {
		return nil, assert.AnError
	}
	return m.LocalFS.OpenFile(name, flag, perm)
}

func (m MockFS) Rename(oldpath, newpath string) error {
	if m.FailRename {
		return assert.AnError
	}
	return m.LocalFS.Rename(oldpath, newpath)
}

func TestStore_SaveErrors(t *testing.T) {
	dir := t.TempDir()
	
	t.Run("OpenFile Error", func(t *testing.T) {
		fsys := MockFS{FailOpenFile: true}
		store := NewStore(fsys, dir)
		err := store.Save(&Manifest{})
		assert.Error(t, err)
	})

	t.Run("Rename Error", func(t *testing.T) {
		fsys := MockFS{FailRename: true}
		store := NewStore(fsys, dir)
		err := store.Save(&Manifest{})
		assert.Error(t, err)
	})
}

func TestStore_SaveErrors_Write(t *testing.T) {
	dir := t.TempDir()
	
	// Use FaultyFS to fail writes
	fsys := fs.NewFaultyFS(fs.LocalFS{})
	fsys.SetLimit(10) // Small limit to fail early
	
	store := NewStore(fsys, dir)
	err := store.Save(&Manifest{ID: 1})
	assert.Error(t, err)
}
