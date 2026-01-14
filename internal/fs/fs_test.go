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

	// WriteAt
	_, err = f.WriteAt([]byte("X"), 2)
	assert.NoError(t, err)

	// Sync
	assert.NoError(t, f.Sync())

	// Stat via File
	info, err := f.Stat()
	assert.NoError(t, err)
	assert.Equal(t, int64(5), info.Size())

	// ReadAt to verify WriteAt worked
	buf := make([]byte, 5)
	_, err = f.ReadAt(buf, 0)
	assert.NoError(t, err)
	assert.Equal(t, []byte("heXlo"), buf)

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

func TestFaultyFS_GlobalLimit(t *testing.T) {
	tmp := t.TempDir()
	ffs := NewFaultyFS(nil)

	ffs.SetLimit(5) // Fail after 5 bytes globally

	fpath := filepath.Join(tmp, "faulty.txt")
	f, err := ffs.OpenFile(fpath, os.O_CREATE|os.O_RDWR, 0644)
	require.NoError(t, err)

	// Write 5 bytes - OK
	n, err := f.Write([]byte("hello"))
	assert.NoError(t, err)
	assert.Equal(t, 5, n)

	// Write 1 byte - Fail (exceeds global limit)
	n, err = f.Write([]byte("!"))
	assert.Error(t, err)
	assert.Equal(t, 0, n)

	// Verify counter
	assert.Equal(t, int64(5), ffs.GetWritten())

	f.Close()

	// Verify other methods delegate correctly
	assert.NoError(t, ffs.Rename(fpath, fpath+".renamed"))
	_, err = ffs.Stat(fpath + ".renamed")
	assert.NoError(t, err)
}

func TestFaultyFS_WriteAt(t *testing.T) {
	tmp := t.TempDir()
	ffs := NewFaultyFS(nil)

	ffs.SetLimit(10)

	fpath := filepath.Join(tmp, "writeat.txt")
	f, err := ffs.OpenFile(fpath, os.O_CREATE|os.O_RDWR, 0644)
	require.NoError(t, err)
	defer f.Close()

	// WriteAt 5 bytes - OK
	n, err := f.WriteAt([]byte("hello"), 0)
	assert.NoError(t, err)
	assert.Equal(t, 5, n)

	// WriteAt 5 more bytes - OK
	n, err = f.WriteAt([]byte("world"), 5)
	assert.NoError(t, err)
	assert.Equal(t, 5, n)

	// WriteAt 1 more byte - Fail (exceeds global limit)
	n, err = f.WriteAt([]byte("!"), 10)
	assert.Error(t, err)
	assert.Equal(t, 0, n)

	assert.Equal(t, int64(10), ffs.GetWritten())
}

func TestFaultyFS_PerFileRule(t *testing.T) {
	tmp := t.TempDir()
	ffs := NewFaultyFS(nil)

	// Rule: fail after 3 bytes for files matching "small"
	ffs.AddRule("small", Fault{FailAfterBytes: 3})

	// File matching rule
	smallPath := filepath.Join(tmp, "small.txt")
	f1, err := ffs.OpenFile(smallPath, os.O_CREATE|os.O_RDWR, 0644)
	require.NoError(t, err)

	_, err = f1.Write([]byte("abc")) // 3 bytes - OK
	assert.NoError(t, err)
	_, err = f1.Write([]byte("d")) // 4th byte - Fail
	assert.Error(t, err)
	f1.Close()

	// File NOT matching rule - no limit
	bigPath := filepath.Join(tmp, "big.txt")
	f2, err := ffs.OpenFile(bigPath, os.O_CREATE|os.O_RDWR, 0644)
	require.NoError(t, err)

	_, err = f2.Write([]byte("abcdefghij")) // 10 bytes - OK (no global limit set)
	assert.NoError(t, err)
	f2.Close()
}

func TestFaultyFS_SyncClose(t *testing.T) {
	tmp := t.TempDir()
	ffs := NewFaultyFS(nil)

	customErr := assert.AnError
	ffs.AddRule("fail", Fault{
		FailAfterBytes: -1, // No byte limit
		FailOnSync:     true,
		FailOnClose:    true,
		Err:            customErr,
	})

	fpath := filepath.Join(tmp, "fail.txt")
	f, err := ffs.OpenFile(fpath, os.O_CREATE|os.O_RDWR, 0644)
	require.NoError(t, err)

	// Write should work
	_, err = f.Write([]byte("test"))
	assert.NoError(t, err)

	// Sync should fail with custom error
	err = f.Sync()
	assert.Equal(t, customErr, err)

	// Close should fail with custom error (but still close file)
	err = f.Close()
	assert.Equal(t, customErr, err)
}

func TestFaultyFS_Reset(t *testing.T) {
	ffs := NewFaultyFS(nil)

	ffs.SetLimit(100)
	ffs.AddRule("test", Fault{FailOnSync: true})

	// Simulate some writes
	tmp := t.TempDir()
	f, _ := ffs.OpenFile(filepath.Join(tmp, "x.txt"), os.O_CREATE|os.O_RDWR, 0644)
	f.Write([]byte("data"))
	f.Close()

	assert.Equal(t, int64(4), ffs.GetWritten())

	// Reset
	ffs.Reset()

	assert.Equal(t, int64(0), ffs.GetWritten())

	// Rules should be cleared - "test" pattern should no longer apply
	f2, _ := ffs.OpenFile(filepath.Join(tmp, "test.txt"), os.O_CREATE|os.O_RDWR, 0644)
	err := f2.Sync() // Should NOT fail since rules were reset
	assert.NoError(t, err)
	f2.Close()
}

func TestFaultyFS_Delegation(t *testing.T) {
	tmp := t.TempDir()
	ffs := NewFaultyFS(nil)

	// MkdirAll
	dir := filepath.Join(tmp, "subdir")
	assert.NoError(t, ffs.MkdirAll(dir, 0755))

	// Truncate
	fpath := filepath.Join(dir, "test.txt")
	f, _ := ffs.OpenFile(fpath, os.O_CREATE, 0644)
	f.Close()
	assert.NoError(t, ffs.Truncate(fpath, 10))

	// Stat
	info, err := ffs.Stat(fpath)
	assert.NoError(t, err)
	assert.Equal(t, int64(10), info.Size())

	// ReadDir
	entries, err := ffs.ReadDir(dir)
	assert.NoError(t, err)
	assert.Len(t, entries, 1)

	// Remove
	assert.NoError(t, ffs.Remove(fpath))
}
