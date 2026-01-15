package blobstore

import (
	"bytes"
	"context"
	"io"
	"sync"
)

// MemoryStore is an in-memory BlobStore implementation for testing.
// It stores blobs in memory without any filesystem dependency.
// Thread-safe for concurrent reads and writes.
type MemoryStore struct {
	mu    sync.RWMutex
	blobs map[string][]byte
}

// NewMemoryStore creates a new in-memory blob store.
func NewMemoryStore() *MemoryStore {
	return &MemoryStore{
		blobs: make(map[string][]byte),
	}
}

// Open opens a blob for reading.
func (m *MemoryStore) Open(_ context.Context, name string) (Blob, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	data, ok := m.blobs[name]
	if !ok {
		return nil, ErrNotFound
	}

	// Return a copy to prevent external mutation
	copied := make([]byte, len(data))
	copy(copied, data)

	return &memoryBlob{data: copied}, nil
}

// Create creates a new writable blob.
func (m *MemoryStore) Create(_ context.Context, name string) (WritableBlob, error) {
	return &memoryWritableBlob{
		store: m,
		name:  name,
	}, nil
}

// Put writes a blob atomically.
func (m *MemoryStore) Put(_ context.Context, name string, data []byte) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Copy to prevent external mutation
	copied := make([]byte, len(data))
	copy(copied, data)
	m.blobs[name] = copied
	return nil
}

// Delete removes a blob.
func (m *MemoryStore) Delete(_ context.Context, name string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	delete(m.blobs, name)
	return nil
}

// List returns all blobs matching the prefix.
func (m *MemoryStore) List(_ context.Context, prefix string) ([]string, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	var names []string
	for name := range m.blobs {
		if prefix == "" || len(name) >= len(prefix) && name[:len(prefix)] == prefix {
			names = append(names, name)
		}
	}
	return names, nil
}

// memoryBlob implements Blob for in-memory data.
type memoryBlob struct {
	data []byte
}

func (b *memoryBlob) ReadAt(_ context.Context, p []byte, off int64) (int, error) {
	if off >= int64(len(b.data)) {
		return 0, io.EOF
	}
	n := copy(p, b.data[off:])
	if n < len(p) {
		return n, io.EOF
	}
	return n, nil
}

func (b *memoryBlob) Close() error {
	return nil
}

func (b *memoryBlob) Size() int64 {
	return int64(len(b.data))
}

func (b *memoryBlob) ReadRange(_ context.Context, off, length int64) (io.ReadCloser, error) {
	if off >= int64(len(b.data)) {
		return io.NopCloser(bytes.NewReader(nil)), nil
	}
	end := off + length
	if end > int64(len(b.data)) {
		end = int64(len(b.data))
	}
	return io.NopCloser(bytes.NewReader(b.data[off:end])), nil
}

// memoryWritableBlob implements WritableBlob for in-memory writes.
type memoryWritableBlob struct {
	store *MemoryStore
	name  string
	buf   bytes.Buffer
}

func (w *memoryWritableBlob) Write(p []byte) (int, error) {
	return w.buf.Write(p)
}

func (w *memoryWritableBlob) Close() error {
	w.store.mu.Lock()
	defer w.store.mu.Unlock()

	// Copy buffer to store
	data := make([]byte, w.buf.Len())
	copy(data, w.buf.Bytes())
	w.store.blobs[w.name] = data
	return nil
}

func (w *memoryWritableBlob) Sync() error {
	return nil
}
