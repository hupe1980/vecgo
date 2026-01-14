package benchmark_test

import (
	"context"
	"crypto/rand"
	"io"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/hupe1980/vecgo/blobstore"
	"github.com/hupe1980/vecgo/internal/cache"
	"github.com/hupe1980/vecgo/internal/resource"
)

// LatencyStore wraps a BlobStore and adds artificial latency to Open.
// It also returns a LatencyBlob that adds latency to reads.
type LatencyStore struct {
	base    blobstore.BlobStore
	latency time.Duration
}

func (s *LatencyStore) Open(ctx context.Context, name string) (blobstore.Blob, error) {
	time.Sleep(s.latency / 2) // Overhead for metadata check
	b, err := s.base.Open(ctx, name)
	if err != nil {
		return nil, err
	}
	return &LatencyBlob{base: b, latency: s.latency}, nil
}

func (s *LatencyStore) Create(ctx context.Context, name string) (blobstore.WritableBlob, error) {
	return s.base.Create(ctx, name)
}

func (s *LatencyStore) Put(ctx context.Context, name string, data []byte) error {
	w, err := s.Create(ctx, name)
	if err != nil {
		return err
	}
	defer w.Close()
	if _, err := w.Write(data); err != nil {
		return err
	}
	return w.Sync()
}

func (s *LatencyStore) Delete(ctx context.Context, name string) error {
	return s.base.Delete(ctx, name)
}

func (s *LatencyStore) List(ctx context.Context, prefix string) ([]string, error) {
	return s.base.List(ctx, prefix)
}

type LatencyBlob struct {
	base    blobstore.Blob
	latency time.Duration
}

func (b *LatencyBlob) ReadAt(ctx context.Context, p []byte, off int64) (n int, err error) {
	time.Sleep(b.latency)
	return b.base.ReadAt(ctx, p, off)
}

func (b *LatencyBlob) ReadRange(ctx context.Context, off, len int64) (io.ReadCloser, error) {
	time.Sleep(b.latency)
	return b.base.ReadRange(ctx, off, len)
}

func (b *LatencyBlob) Size() int64 {
	return b.base.Size()
}

func (b *LatencyBlob) Close() error {
	return b.base.Close()
}

// -----------------------------------------------------------------------------
// Benchmarks
// -----------------------------------------------------------------------------

func setupStorageData(t *testing.B, size int64) (string, string) {
	dir := t.TempDir()
	name := "benchmark_blob"
	path := filepath.Join(dir, name)

	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	// Write random data
	if _, err := io.CopyN(f, rand.Reader, size); err != nil {
		t.Fatal(err)
	}
	return dir, name
}

// simulate cloud latency (e.g. S3 typical TTFB can be 10-50ms)
const cloudLatency = 30 * time.Millisecond

func BenchmarkStorage_ReadRandom(b *testing.B) {
	size := int64(4 * 1024 * 1024) // 4MB blob
	dir, name := setupStorageData(b, size)

	// Stores
	local := blobstore.NewLocalStore(dir)
	simulatedS3 := &LatencyStore{base: local, latency: cloudLatency}

	rc := resource.NewController(resource.Config{MemoryLimitBytes: 1 << 30})
	blockCache := cache.NewLRUBlockCache(64*1024*1024, rc) // 64MB cache
	cachedS3 := blobstore.NewCachingStore(simulatedS3, blockCache, 4096)

	// Sub-benchmarks
	b.Run("Local_NVMe", func(b *testing.B) {
		blob, err := local.Open(context.Background(), name)
		if err != nil {
			b.Fatal(err)
		}
		defer blob.Close()

		ctx := context.Background()
		buf := make([]byte, 1024)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			// Random read
			off := int64(i*1024) % (size - 1024)
			if _, err := blob.ReadAt(ctx, buf, off); err != nil {
				b.Fatal(err)
			}
		}
	})

	b.Run("Simulated_S3_Direct", func(b *testing.B) {
		blob, err := simulatedS3.Open(context.Background(), name)
		if err != nil {
			b.Fatal(err)
		}
		defer blob.Close()

		ctx := context.Background()
		buf := make([]byte, 1024)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			off := int64(i*1024) % (size - 1024)
			if _, err := blob.ReadAt(ctx, buf, off); err != nil {
				b.Fatal(err)
			}
		}
	})

	b.Run("Cached_S3_Warm", func(b *testing.B) {
		// Warm up by reading everything once
		blob, err := cachedS3.Open(context.Background(), name)
		if err != nil {
			b.Fatal(err)
		}
		defer blob.Close()

		ctx := context.Background()
		buf := make([]byte, 4096) // Match block size
		// Pre-warm the first few blocks
		for i := 0; i < 10; i++ {
			blob.ReadAt(ctx, buf, int64(i*4096))
		}

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			// Read from the warmed up region
			idx := i % 10
			off := int64(idx * 4096)
			if _, err := blob.ReadAt(ctx, buf, off); err != nil {
				b.Fatal(err)
			}
		}
	})
}
