package blobstore

import (
	"bytes"
	"context"
	"io"
	"os"
	"path/filepath"
	"sort"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestLocalBlobStore_Lifecycle(t *testing.T) {
	tmpDir := t.TempDir()
	store := NewLocalStore(tmpDir)

	ctx := context.Background()

	// 1. Create a blob
	blobName := "data-001.bin"
	data := []byte("hello world, this is a test blob for vecgo")

	w, err := store.Create(ctx, blobName)
	require.NoError(t, err)

	n, err := w.Write(data)
	require.NoError(t, err)
	require.Equal(t, len(data), n)

	err = w.Close()
	require.NoError(t, err)

	// Verify file exists on disk
	expectedPath := filepath.Join(tmpDir, blobName)
	_, err = os.Stat(expectedPath)
	require.NoError(t, err)

	// 2. Open and ReadAt
	blob, err := store.Open(ctx, blobName)
	require.NoError(t, err)
	defer blob.Close()

	require.Equal(t, int64(len(data)), blob.Size())

	buf := make([]byte, 5)
	n, err = blob.ReadAt(ctx, buf, 6) // "world"
	require.NoError(t, err)
	require.Equal(t, 5, n)
	require.Equal(t, "world", string(buf))

	// 3. ReadRange
	// Read "this" (offset 13, length 4)
	rangeReader, err := blob.ReadRange(ctx, 13, 4)
	require.NoError(t, err)
	defer rangeReader.Close()

	rangeContent, err := io.ReadAll(rangeReader)
	require.NoError(t, err)
	require.Equal(t, "this", string(rangeContent))

	// 4. List
	// Create another file to test listing
	blobName2 := "data-002.bin"
	w2, err := store.Create(ctx, blobName2)
	require.NoError(t, err)
	w2.Close()

	blobs, err := store.List(ctx, "")
	require.NoError(t, err)

	// Sort for deterministic assertion
	var names []string
	for _, b := range blobs {
		names = append(names, b)
	}
	sort.Strings(names)

	require.Equal(t, []string{blobName, blobName2}, names)

	// 5. Delete
	err = store.Delete(ctx, blobName)
	require.NoError(t, err)

	// Verify deletion
	blobsAfter, err := store.List(ctx, "")
	require.NoError(t, err)
	require.Equal(t, []string{blobName2}, blobsAfter)

	_, err = store.Open(ctx, blobName)
	require.Error(t, err) // Should fail now
}

func TestLocalBlobStore_ReadRange_Boundaries(t *testing.T) {
	tmpDir := t.TempDir()
	store := NewLocalStore(tmpDir)
	ctx := context.Background()

	blobName := "boundary.bin"
	data := []byte("0123456789")
	w, _ := store.Create(ctx, blobName)
	w.Write(data)
	w.Close()

	blob, err := store.Open(ctx, blobName)
	require.NoError(t, err)
	defer blob.Close()

	// Case 1: Read full range
	r, err := blob.ReadRange(ctx, 0, 10)
	require.NoError(t, err)
	content, _ := io.ReadAll(r)
	r.Close()
	require.True(t, bytes.Equal(data, content))

	// Case 2: Read past end
	r, err = blob.ReadRange(ctx, 8, 5) // Request 5 bytes starting at 8 (only 2 available: 8, 9)
	require.NoError(t, err)
	content, err = io.ReadAll(r)
	require.NoError(t, err)
	require.Equal(t, "89", string(content))
	r.Close()

	// Case 3: Offset past EOF
	r, err = blob.ReadRange(ctx, 20, 5)
	require.ErrorIs(t, err, io.EOF)
	if r != nil {
		r.Close()
	}
}
