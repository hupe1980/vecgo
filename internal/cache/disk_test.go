package cache

import (
	"context"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDiskBlockCache(t *testing.T) {
	tmpDir := t.TempDir()
	config := DiskCacheConfig{
		RootDir:      tmpDir,
		MaxSizeBytes: 1024, // 1KB limit
	}

	c, err := NewDiskBlockCache(config)
	require.NoError(t, err)

	ctx := context.Background()
	key1 := CacheKey{Kind: CacheKindBlob, SegmentID: 1, Offset: 0}
	data1 := make([]byte, 400) // 400 bytes

	c.Set(ctx, key1, data1)

	// Wait for async write for testing
	time.Sleep(100 * time.Millisecond)

	// Check file exists
	relPath := c.encodeKeyToRelPath(key1)
	assert.FileExists(t, filepath.Join(tmpDir, relPath))

	// Get
	got, ok := c.Get(ctx, key1)
	assert.True(t, ok)
	assert.Equal(t, len(data1), len(got))

	// Add more to trigger eviction
	key2 := CacheKey{Kind: CacheKindBlob, SegmentID: 1, Offset: 1}
	data2 := make([]byte, 400) // 400 bytes
	c.Set(ctx, key2, data2)

	key3 := CacheKey{Kind: CacheKindBlob, SegmentID: 1, Offset: 2}
	data3 := make([]byte, 400) // 400 bytes
	c.Set(ctx, key3, data3)
	time.Sleep(100 * time.Millisecond)

	// Total 1200 bytes > 1024 limit. Key1 should be evicted (LRU)
	_, ok = c.Get(ctx, key1)
	assert.False(t, ok, "Key1 should be evicted")
	assert.NoFileExists(t, filepath.Join(tmpDir, relPath))

	// Key2 and Key3 should be present
	_, ok = c.Get(ctx, key2)
	assert.True(t, ok)
	_, ok = c.Get(ctx, key3)
	assert.True(t, ok)
}

func TestDiskBlockCache_Reload(t *testing.T) {
	tmpDir := t.TempDir()
	config := DiskCacheConfig{RootDir: tmpDir, MaxSizeBytes: 10000}

	key1 := CacheKey{Kind: CacheKindBlob, SegmentID: 1, Offset: 0}

	// Open and set
	{
		c, _ := NewDiskBlockCache(config)
		c.Set(context.Background(), key1, []byte("hello"))
		time.Sleep(100 * time.Millisecond) // Wait for flush
	}

	// Re-open
	{
		c, _ := NewDiskBlockCache(config)
		got, ok := c.Get(context.Background(), key1)
		assert.True(t, ok)
		assert.Equal(t, "hello", string(got))
		assert.Equal(t, int64(5), c.currentSize)
	}
}

func TestDiskBlockCache_Path(t *testing.T) {
	tmpDir := t.TempDir()
	config := DiskCacheConfig{RootDir: tmpDir, MaxSizeBytes: 10000}
	c, _ := NewDiskBlockCache(config)

	key := CacheKey{Kind: CacheKindBlob, SegmentID: 1, Offset: 0, Path: "foo/bar"}
	c.Set(context.Background(), key, []byte("data"))
	time.Sleep(100 * time.Millisecond)

	// Verify file location
	expectedPath := filepath.Join(tmpDir, "foo/bar", "4-1-0.blk")
	assert.FileExists(t, expectedPath)

	// Verify Get
	got, ok := c.Get(context.Background(), key)
	assert.True(t, ok)
	assert.Equal(t, "data", string(got))
}
