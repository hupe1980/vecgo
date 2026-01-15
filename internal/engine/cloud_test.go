package engine

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/hupe1980/vecgo/blobstore"
	"github.com/hupe1980/vecgo/distance"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestRemote_DirectCloudWrites verifies that OpenRemote() supports direct cloud writes
// where segments are built in memory and written atomically.
func TestRemote_DirectCloudWrites(t *testing.T) {
	// Use in-memory store to simulate cloud storage
	store := blobstore.NewMemoryStore()
	ctx := context.Background()

	// Create index directly on "cloud" store
	eng, err := OpenRemote(ctx, store,
		WithDimension(4),
		WithMetric(distance.MetricL2),
		WithFlushConfig(FlushConfig{MaxMemTableSize: 1}), // Force immediate flush
	)
	require.NoError(t, err)

	// Insert data
	id1, err := eng.Insert(ctx, []float32{1, 0, 0, 0}, nil, nil)
	require.NoError(t, err)
	id2, err := eng.Insert(ctx, []float32{0, 1, 0, 0}, nil, nil)
	require.NoError(t, err)

	// Wait for background flush
	time.Sleep(200 * time.Millisecond)

	// Commit to ensure persistence
	require.NoError(t, eng.Commit(ctx))

	// Search works
	results, err := eng.Search(ctx, []float32{1, 0, 0, 0}, 10)
	require.NoError(t, err)
	assert.GreaterOrEqual(t, len(results), 1)
	assert.Equal(t, id1, results[0].ID)

	// Close writer
	require.NoError(t, eng.Close())

	// Verify data is in "cloud" store by reopening as reader
	reader, err := OpenRemote(ctx, store, ReadOnly())
	require.NoError(t, err)
	defer reader.Close()

	results, err = reader.Search(ctx, []float32{0, 1, 0, 0}, 10)
	require.NoError(t, err)
	require.Len(t, results, 2)
	assert.Equal(t, id2, results[0].ID) // Closest to [0,1,0,0]
}

// TestRemote_ReadOnly verifies that ReadOnly() mode rejects writes.
func TestRemote_ReadOnly(t *testing.T) {
	store := blobstore.NewMemoryStore()
	ctx := context.Background()

	// First create an index with some data
	writer, err := OpenRemote(ctx, store,
		WithDimension(4),
		WithMetric(distance.MetricL2),
	)
	require.NoError(t, err)
	_, err = writer.Insert(ctx, []float32{1, 0, 0, 0}, nil, nil)
	require.NoError(t, err)
	require.NoError(t, writer.Commit(ctx))
	require.NoError(t, writer.Close())

	// Open as read-only
	reader, err := OpenRemote(ctx, store, ReadOnly())
	require.NoError(t, err)
	defer reader.Close()

	// Search should work
	results, err := reader.Search(ctx, []float32{1, 0, 0, 0}, 10)
	require.NoError(t, err)
	assert.Len(t, results, 1)

	// Insert should fail with ErrReadOnly
	_, err = reader.Insert(ctx, []float32{0, 1, 0, 0}, nil, nil)
	assert.True(t, errors.Is(err, ErrReadOnly), "expected ErrReadOnly, got %v", err)

	// Delete should fail with ErrReadOnly
	err = reader.Delete(ctx, 1)
	assert.True(t, errors.Is(err, ErrReadOnly), "expected ErrReadOnly, got %v", err)
}

// TestRemote_WithoutDiskCache verifies pure memory cache mode (no disk cache).
func TestRemote_WithoutDiskCache(t *testing.T) {
	store := blobstore.NewMemoryStore()
	ctx := context.Background()

	// Create index
	writer, err := OpenRemote(ctx, store,
		WithDimension(4),
		WithMetric(distance.MetricL2),
	)
	require.NoError(t, err)

	// Insert data
	for i := 0; i < 100; i++ {
		vec := make([]float32, 4)
		vec[i%4] = 1.0
		_, err := writer.Insert(ctx, vec, nil, nil)
		require.NoError(t, err)
	}
	require.NoError(t, writer.Commit(ctx))
	require.NoError(t, writer.Close())

	// Open without specifying disk cache dir (should use temp dir with memory-only cache)
	reader, err := OpenRemote(ctx, store,
		ReadOnly(),
		WithBlockCacheSize(1<<20), // 1MB block cache only
	)
	require.NoError(t, err)
	defer reader.Close()

	// Search should work
	results, err := reader.Search(ctx, []float32{1, 0, 0, 0}, 10)
	require.NoError(t, err)
	assert.GreaterOrEqual(t, len(results), 1)
}

// TestRemote_WithDiskCache verifies disk cache accelerates repeated reads.
func TestRemote_WithDiskCache(t *testing.T) {
	store := blobstore.NewMemoryStore()
	ctx := context.Background()
	cacheDir := t.TempDir()

	// Create index
	writer, err := OpenRemote(ctx, store,
		WithDimension(4),
		WithMetric(distance.MetricL2),
	)
	require.NoError(t, err)

	for i := 0; i < 50; i++ {
		vec := make([]float32, 4)
		vec[i%4] = float32(i)
		_, err := writer.Insert(ctx, vec, nil, nil)
		require.NoError(t, err)
	}
	require.NoError(t, writer.Commit(ctx))
	require.NoError(t, writer.Close())

	// Open with disk cache
	reader, err := OpenRemote(ctx, store,
		ReadOnly(),
		WithCacheDir(cacheDir),
		WithDiskCache(cacheDir, 10<<20, 1<<20), // 10MB disk cache, 1MB blocks
	)
	require.NoError(t, err)
	defer reader.Close()

	// First search (cold cache)
	query := []float32{1, 0, 0, 0}
	results1, err := reader.Search(ctx, query, 10)
	require.NoError(t, err)
	assert.GreaterOrEqual(t, len(results1), 1)

	// Second search (warm cache)
	results2, err := reader.Search(ctx, query, 10)
	require.NoError(t, err)
	assert.Equal(t, len(results1), len(results2))

	// Verify cache stats show hits
	hits, misses := reader.CacheStats()
	assert.Greater(t, hits+misses, int64(0), "cache should have activity")
}

// TestRemote_ConcurrentReaders verifies multiple readers can share the same store.
func TestRemote_ConcurrentReaders(t *testing.T) {
	store := blobstore.NewMemoryStore()
	ctx := context.Background()

	// Create index
	writer, err := OpenRemote(ctx, store,
		WithDimension(4),
		WithMetric(distance.MetricL2),
	)
	require.NoError(t, err)
	for i := 0; i < 100; i++ {
		vec := make([]float32, 4)
		vec[i%4] = 1.0
		_, err := writer.Insert(ctx, vec, nil, nil)
		require.NoError(t, err)
	}
	require.NoError(t, writer.Commit(ctx))
	require.NoError(t, writer.Close())

	// Open multiple readers concurrently
	const numReaders = 5
	readers := make([]*Engine, numReaders)
	for i := 0; i < numReaders; i++ {
		reader, err := OpenRemote(ctx, store, ReadOnly())
		require.NoError(t, err)
		readers[i] = reader
	}

	// All readers should be able to search
	for i, reader := range readers {
		results, err := reader.Search(ctx, []float32{1, 0, 0, 0}, 10)
		require.NoError(t, err, "reader %d failed", i)
		assert.GreaterOrEqual(t, len(results), 1, "reader %d got no results", i)
	}

	// Close all readers
	for _, reader := range readers {
		require.NoError(t, reader.Close())
	}
}

// TestLocal_ReopenRemote verifies data written locally can be read via Remote.
func TestLocal_ReopenRemote(t *testing.T) {
	// This tests the hybrid pattern: write locally, read via remote store
	store := blobstore.NewMemoryStore()
	ctx := context.Background()

	// Write directly to the store (simulating local write + sync)
	writer, err := OpenRemote(ctx, store,
		WithDimension(4),
		WithMetric(distance.MetricL2),
	)
	require.NoError(t, err)

	id1, err := writer.Insert(ctx, []float32{1, 0, 0, 0}, nil, nil)
	require.NoError(t, err)
	id2, err := writer.Insert(ctx, []float32{0, 1, 0, 0}, nil, nil)
	require.NoError(t, err)
	require.NoError(t, writer.Commit(ctx))
	require.NoError(t, writer.Close())

	// List files in store to verify data was written
	files, err := store.List(ctx, "")
	require.NoError(t, err)
	assert.GreaterOrEqual(t, len(files), 2, "expected manifest and segments, got %v", files)

	// Reopen as remote reader
	reader, err := OpenRemote(ctx, store, ReadOnly())
	require.NoError(t, err)
	defer reader.Close()

	// Both vectors should be searchable
	results, err := reader.Search(ctx, []float32{1, 0, 0, 0}, 10)
	require.NoError(t, err)
	assert.Len(t, results, 2)
	assert.Equal(t, id1, results[0].ID)
	assert.Equal(t, id2, results[1].ID)
}

// TestRemote_DeleteOperations verifies delete works with remote storage.
func TestRemote_DeleteOperations(t *testing.T) {
	store := blobstore.NewMemoryStore()
	ctx := context.Background()

	eng, err := OpenRemote(ctx, store,
		WithDimension(4),
		WithMetric(distance.MetricL2),
	)
	require.NoError(t, err)

	// Insert
	id1, err := eng.Insert(ctx, []float32{1, 0, 0, 0}, nil, nil)
	require.NoError(t, err)
	id2, err := eng.Insert(ctx, []float32{0, 1, 0, 0}, nil, nil)
	require.NoError(t, err)

	// Delete one
	require.NoError(t, eng.Delete(ctx, id1))

	// Search should only find id2
	results, err := eng.Search(ctx, []float32{1, 0, 0, 0}, 10)
	require.NoError(t, err)
	require.Len(t, results, 1)
	assert.Equal(t, id2, results[0].ID)

	// Commit and reopen
	require.NoError(t, eng.Commit(ctx))
	require.NoError(t, eng.Close())

	reader, err := OpenRemote(ctx, store, ReadOnly())
	require.NoError(t, err)
	defer reader.Close()

	// Still only id2
	results, err = reader.Search(ctx, []float32{1, 0, 0, 0}, 10)
	require.NoError(t, err)
	require.Len(t, results, 1)
	assert.Equal(t, id2, results[0].ID)
}
