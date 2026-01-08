package fs

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestLocalFS(t *testing.T) {
	tmp := t.TempDir()
	lfs := LocalFS{}

	// Test MkdirAll
	dir := filepath.Join(tmp, "subdir")
	assert.NoError(t, lfs.MkdirAll(dir, 0755))

	// Test OpenFile (Create)
	fpath := filepath.Join(dir, "test.txt")
	f, err := lfs.OpenFile(fpath, os.O_CREATE|os.O_RDWR, 0644)
	require.NoError(t, err)

	// Write
	_, err = f.Write([]byte("hello"))
	assert.NoError(t, err)

	// Sync
	assert.NoError(t, f.Sync())

	// Stat via File
	info, err := f.Stat()
	assert.NoError(t, err)
	assert.Equal(t, int64(5), info.Size())

	assert.NoError(t, f.Close())

	// Stat via FS
	info2, err := lfs.Stat(fpath)
	assert.NoError(t, err)
	assert.Equal(t, int64(5), info2.Size())

	// ReadDir
	entries, err := lfs.ReadDir(dir)
	assert.NoError(t, err)
	assert.Len(t, entries, 1)

	// Rename
	newPath := filepath.Join(dir, "renamed.txt")
	assert.NoError(t, lfs.Rename(fpath, newPath))

	// Truncate
	assert.NoError(t, lfs.Truncate(newPath, 3))
	info3, err := lfs.Stat(newPath)
	assert.NoError(t, err)
	assert.Equal(t, int64(3), info3.Size())

	// Remove
	assert.NoError(t, lfs.Remove(newPath))
	_, err = lfs.Stat(newPath)
	assert.True(t, os.IsNotExist(err))
}

func TestFaultyFS(t *testing.T) {
	tmp := t.TempDir()
	lfs := LocalFS{}
	ffs := NewFaultyFS(lfs)

	ffs.SetLimit(5) // Fail after 5 bytes

	fpath := filepath.Join(tmp, "faulty.txt")
	f, err := ffs.OpenFile(fpath, os.O_CREATE|os.O_RDWR, 0644)
	require.NoError(t, err)

	// Write 5 bytes - OK
	n, err := f.Write([]byte("hello"))
	assert.NoError(t, err)
	assert.Equal(t, 5, n)

	// Write 1 byte - Fail
	n, err = f.Write([]byte("!"))
	assert.Error(t, err)
	assert.Equal(t, 0, n) // Or logic might differ?

	// Verify delegation
	assert.Equal(t, int64(5), ffs.GetWritten())

	f.Close()

	// Verify other methods delegate
	assert.NoError(t, ffs.Rename(fpath, fpath+".renamed"))
	_, err = ffs.Stat(fpath + ".renamed")
	assert.NoError(t, err)
}

func TestFaultyFS_Delegation(t *testing.T) {
	tmp := t.TempDir()
	lfs := LocalFS{}
	ffs := NewFaultyFS(lfs)

	// MkdirAll
	dir := filepath.Join(tmp, "subdir")
	assert.NoError(t, ffs.MkdirAll(dir, 0755))

	// Truncate
	fpath := filepath.Join(dir, "test.txt")
	f, _ := lfs.OpenFile(fpath, os.O_CREATE, 0644)
	f.Close()
	assert.NoError(t, ffs.Truncate(fpath, 10))

	// Remove
	assert.NoError(t, ffs.Remove(fpath))

	// ReadDir
	_, err := ffs.ReadDir(dir)
	assert.NoError(t, err)
}
