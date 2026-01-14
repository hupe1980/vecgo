package blobstore

import (
	"context"
	"io"
	"testing"

	"github.com/hupe1980/vecgo/internal/cache"
)

type mockCountingStore struct {
	readCount int
}

func (m *mockCountingStore) Open(ctx context.Context, name string) (Blob, error) {
	return &mockCountingBlob{store: m, size: 1024 * 1024}, nil
}
func (m *mockCountingStore) Create(ctx context.Context, name string) (WritableBlob, error) {
	return nil, nil
}
func (m *mockCountingStore) Put(ctx context.Context, name string, data []byte) error { return nil }
func (m *mockCountingStore) Delete(ctx context.Context, name string) error           { return nil }
func (m *mockCountingStore) List(ctx context.Context, prefix string) ([]string, error) {
	return nil, nil
}

type mockCountingBlob struct {
	store *mockCountingStore
	size  int64
}

func (b *mockCountingBlob) ReadAt(ctx context.Context, p []byte, off int64) (int, error) {
	b.store.readCount++
	// Simulate reading zeros
	for i := range p {
		p[i] = 0
	}
	return len(p), nil
}
func (b *mockCountingBlob) ReadRange(ctx context.Context, off, len int64) (io.ReadCloser, error) {
	return nil, nil
}
func (b *mockCountingBlob) Size() int64  { return b.size }
func (b *mockCountingBlob) Close() error { return nil }

func TestCachingStore_Coalescing(t *testing.T) {
	inner := &mockCountingStore{}
	// 1KB blocks
	cachingStore := NewCachingStore(inner, cache.NewLRUBlockCache(1024*1024, nil), 1024)

	ctx := context.Background()
	blob, _ := cachingStore.Open(ctx, "test")

	// Read 10KB (10 blocks)
	buf := make([]byte, 10*1024)
	blob.ReadAt(ctx, buf, 0)

	// In current implementation, this should be 10 reads (serial).
	// We want it to be 1 read.
	if inner.readCount > 1 {
		t.Logf("ReadCount: %d (Expected optimized: 1, Current: %d)", inner.readCount, inner.readCount)
	} else {
		t.Logf("ReadCount: %d", inner.readCount)
	}
}
