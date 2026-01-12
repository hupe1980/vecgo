package blobstore

import (
	"bytes"
	"context"
	"io"
	"testing"

	"github.com/hupe1980/vecgo/internal/cache"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type mockBlob struct {
	data      []byte
	reads     int
	readBytes int
}

func (m *mockBlob) Close() error { return nil }
func (m *mockBlob) Size() int64  { return int64(len(m.data)) }
func (m *mockBlob) ReadAt(p []byte, off int64) (int, error) {
	m.reads++
	if off >= int64(len(m.data)) {
		return 0, io.EOF
	}
	n := copy(p, m.data[off:])
	m.readBytes += n
	if n < len(p) {
		return n, io.EOF
	}
	return n, nil
}
func (m *mockBlob) ReadRange(off, len int64) (io.ReadCloser, error) {
	return io.NopCloser(bytes.NewReader(m.data[off : off+len])), nil
}

type mockStore struct {
	blobs map[string]*mockBlob
	opens int
}

func (m *mockStore) Open(ctx context.Context, name string) (Blob, error) {
	m.opens++
	if b, ok := m.blobs[name]; ok {
		return b, nil
	}
	return nil, ErrNotFound
}
func (m *mockStore) Create(ctx context.Context, name string) (WritableBlob, error) { return nil, nil }
func (m *mockStore) Put(ctx context.Context, name string, data []byte) error {
	if m.blobs == nil {
		m.blobs = make(map[string]*mockBlob)
	}
	m.blobs[name] = &mockBlob{data: data}
	return nil
}
func (m *mockStore) Delete(ctx context.Context, name string) error             { return nil }
func (m *mockStore) List(ctx context.Context, prefix string) ([]string, error) { return nil, nil }

func TestCachingStore_ReadAt(t *testing.T) {
	// Setup
	data := make([]byte, 1024) // 1KB
	for i := range data {
		data[i] = byte(i % 255)
	}

	inner := &mockStore{
		blobs: map[string]*mockBlob{
			"test": {data: data},
		},
	}

	c := cache.NewLRUBlockCache(1024*1024, nil) // 1MB cache
	store := NewCachingStore(inner, c, 256)     // 256 bytes block size

	// Open
	blob, err := store.Open(context.Background(), "test")
	require.NoError(t, err)

	// 1. Read first block (bytes 0-100)
	buf := make([]byte, 100)
	n, err := blob.ReadAt(buf, 0)
	require.NoError(t, err)
	assert.Equal(t, 100, n)
	assert.Equal(t, data[:100], buf)

	// Inner blob should have been read (Block 0)
	mBlob := inner.blobs["test"]
	assert.Equal(t, 1, mBlob.reads)
	assert.Equal(t, 256, mBlob.readBytes) // Read full block 0 (256 bytes)

	// 2. Read same range again -> Should hit cache
	n, err = blob.ReadAt(buf, 0)
	require.NoError(t, err)
	assert.Equal(t, 100, n)
	assert.Equal(t, 1, mBlob.reads) // Reads count unchanged

	// 3. Read spanning two blocks (bytes 200-300). Block 0 (0-255) and Block 1 (256-511)
	// Block 0 is cached. Block 1 is not.
	buf2 := make([]byte, 100)
	n, err = blob.ReadAt(buf2, 200)
	require.NoError(t, err)
	assert.Equal(t, 100, n)
	assert.Equal(t, data[200:300], buf2)

	// Inner blob should have been read once more (for Block 1)
	assert.Equal(t, 2, mBlob.reads)
	assert.Equal(t, 256+256, mBlob.readBytes) // +256 for Block 1

	// 4. Read Block 1 again -> cache hit
	n, err = blob.ReadAt(buf2, 260)
	require.NoError(t, err)
	assert.Equal(t, 2, mBlob.reads)
}

func TestCachingStore_SmallFile(t *testing.T) {
	data := []byte("hello")
	inner := &mockStore{
		blobs: map[string]*mockBlob{
			"small": {data: data},
		},
	}
	c := cache.NewLRUBlockCache(1024, nil)
	store := NewCachingStore(inner, c, 256)

	blob, err := store.Open(context.Background(), "small")
	require.NoError(t, err)

	buf := make([]byte, 10)
	n, err := blob.ReadAt(buf, 0)
	// Expect partial read? Or just data length
	assert.Equal(t, 5, n)
	// io.ReaderAt allows returning EOF or nil if n < len(p)?
	// "If the data available is less than len(p), ReadAt returns the number of bytes available and an error explaining why."
	// Usually io.EOF.
	// Check mock implementation behavior:
	// It returns EOF if n < len(p)
	// Our CachingBlob should propagate EOF if appropriate?
	// But CachingBlob.ReadAt constructs the result from blocks.
	// If the file ends, the last block will be short.
	assert.Equal(t, data, buf[:n])
}
