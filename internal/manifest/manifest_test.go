package manifest

import (
	"context"
	"encoding/binary"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/hupe1980/vecgo/blobstore"
	"github.com/hupe1980/vecgo/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestManifestVersioning(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	store := NewStore(blobstore.NewLocalStore(dir))

	// 1. Save a manifest
	m := &Manifest{
		ID: 0,
	}
	err := store.Save(ctx, m)
	require.NoError(t, err)

	// 2. Load it
	loaded, err := store.Load(ctx)
	require.NoError(t, err)
	assert.Equal(t, CurrentVersion, loaded.Version)

	// 3. Corrupt version
	// Find the manifest file
	currentContent, err := os.ReadFile(filepath.Join(dir, CurrentFileName))
	require.NoError(t, err)
	manifestPath := filepath.Join(dir, string(currentContent))

	// Read Binary
	data, err := os.ReadFile(manifestPath)
	require.NoError(t, err)

	// Verify it's binary by checking magic
	require.Greater(t, len(data), 4)
	magic := binary.LittleEndian.Uint32(data[0:4])
	require.Equal(t, uint32(binaryMagic), magic)

	// Modify version (bytes 4-8) - Assuming header structure
	// Magic(4) + Version(4)
	binary.LittleEndian.PutUint32(data[4:8], 999)

	// Write back
	err = os.WriteFile(manifestPath, data, 0644)
	require.NoError(t, err)

	// 4. Load again - should fail
	_, err = store.Load(ctx)
	assert.Error(t, err)
	// Error comes from ReadBinary validation
	assert.Contains(t, err.Error(), "unsupported version")
}

func TestStore(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()

	store := NewStore(blobstore.NewLocalStore(dir))

	// 1. Load on empty -> ErrNotFound (no CURRENT file)
	_, err := store.Load(ctx)
	require.ErrorIs(t, err, ErrNotFound)

	// 2. Create and Save a manifest
	m := New(128, "L2")
	m.NextSegmentID = 100
	err = store.Save(ctx, m)
	require.NoError(t, err)
	assert.Equal(t, uint64(1), m.ID) // Incremented

	// 3. Load updated
	m2, err := store.Load(ctx)
	require.NoError(t, err)
	assert.Equal(t, uint64(1), m2.ID)
	assert.Equal(t, model.SegmentID(100), m2.NextSegmentID)

	// 4. Verify file structure
	matches, _ := filepath.Glob(filepath.Join(dir, "MANIFEST-*"))
	assert.GreaterOrEqual(t, len(matches), 1)

	// 5. Save another one
	err = store.Save(ctx, m)
	require.NoError(t, err)
	assert.Equal(t, uint64(2), m.ID)

	m3, err := store.Load(ctx)
	require.NoError(t, err)
	assert.Equal(t, uint64(2), m3.ID)
}

func TestStore_LoadErrors(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	store := NewStore(blobstore.NewLocalStore(dir))

	// Create a corrupted CURRENT file
	currentPath := filepath.Join(dir, "CURRENT")
	err := os.WriteFile(currentPath, []byte("MANIFEST-999999.json"), 0644)
	require.NoError(t, err)

	_, err = store.Load(ctx)
	assert.Error(t, err) // Should fail to read the manifest file
}

type MockBlobStore struct {
	PutFunc func(ctx context.Context, name string, data []byte) error
	blobstore.BlobStore
}

func (m MockBlobStore) Put(ctx context.Context, name string, data []byte) error {
	if m.PutFunc != nil {
		return m.PutFunc(ctx, name, data)
	}
	return nil
}

func (m MockBlobStore) Open(ctx context.Context, name string) (blobstore.Blob, error) {
	return nil, blobstore.ErrNotFound
}

func (m MockBlobStore) Create(ctx context.Context, name string) (blobstore.WritableBlob, error) {
	return nil, nil
}
func (m MockBlobStore) Delete(ctx context.Context, name string) error             { return nil }
func (m MockBlobStore) List(ctx context.Context, prefix string) ([]string, error) { return nil, nil }

func TestStore_SaveErrors(t *testing.T) {
	ctx := context.Background()
	// Test failure during Put
	mockStore := MockBlobStore{
		PutFunc: func(ctx context.Context, name string, data []byte) error {
			return errors.New("simulated put error")
		},
	}

	store := NewStore(mockStore)
	err := store.Save(ctx, &Manifest{})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "simulated put error")
}
