package main

import (
	"context"
	"fmt"
	"io"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"time"

	"github.com/hupe1980/vecgo"
	"github.com/hupe1980/vecgo/blobstore"
)

// SimulatedS3Store wraps a LocalStore but adds latency to ReadAt calls.
type SimulatedS3Store struct {
	inner   blobstore.BlobStore
	latency time.Duration
}

func NewSimulatedS3Store(dir string, latency time.Duration) *SimulatedS3Store {
	return &SimulatedS3Store{
		inner:   blobstore.NewLocalStore(dir),
		latency: latency,
	}
}

func (s *SimulatedS3Store) Open(ctx context.Context, name string) (blobstore.Blob, error) {
	b, err := s.inner.Open(ctx, name)
	if err != nil {
		return nil, err
	}
	return &SimulatedS3Blob{Blob: b, latency: s.latency}, nil
}

func (s *SimulatedS3Store) Create(ctx context.Context, name string) (blobstore.WritableBlob, error) {
	time.Sleep(s.latency)
	return s.inner.Create(ctx, name)
}
func (s *SimulatedS3Store) Put(ctx context.Context, name string, data []byte) error {
	time.Sleep(s.latency)
	return s.inner.Put(ctx, name, data)
}
func (s *SimulatedS3Store) Delete(ctx context.Context, name string) error {
	return s.inner.Delete(ctx, name)
}
func (s *SimulatedS3Store) List(ctx context.Context, prefix string) ([]string, error) {
	time.Sleep(s.latency)
	return s.inner.List(ctx, prefix)
}

type SimulatedS3Blob struct {
	blobstore.Blob
	latency time.Duration
}

func (b *SimulatedS3Blob) ReadAt(p []byte, off int64) (int, error) {
	time.Sleep(b.latency)
	return b.Blob.ReadAt(p, off)
}

func (b *SimulatedS3Blob) ReadRange(off, len int64) (io.ReadCloser, error) {
	time.Sleep(b.latency)
	return b.Blob.ReadRange(off, len)
}

func main() {
	baseDir := "cloud_example_data"
	os.RemoveAll(baseDir)
	os.MkdirAll(baseDir, 0755)

	localBuildDir := filepath.Join(baseDir, "builder")
	s3BucketDir := filepath.Join(baseDir, "s3_bucket")
	cacheDir := filepath.Join(baseDir, "local_cache")
	searcherDir := filepath.Join(baseDir, "searcher_state")

	os.MkdirAll(s3BucketDir, 0755)
	// We don't need cacheDir since the engine manages it inside searcherDir + temp
	// But in my implementation of Open, I used searcherDir as the root.
	// If I want to separate cache, I should use WithDiskCache explicit option or let it be in searcherDir.
	// "Stateless" node usually has one scratch dir.
	_ = cacheDir

	fmt.Println("🏗️  Building Index locally...")
	buildIndex(localBuildDir)

	fmt.Println("☁️  Uploading blocks to 'S3'...")
	copyDir(localBuildDir, s3BucketDir)

	fmt.Println("🚀 Starting Stateless Search Node...")

	s3Store := NewSimulatedS3Store(s3BucketDir, 20*time.Millisecond)

	// LanceDB-style Read-Only Mode: Truly stateless!
	// - No WAL created
	// - No local directory required (uses temp for cache)
	// - Insert/Delete operations return ErrReadOnly
	// - Perfect for serverless deployments
	opts := []vecgo.Option{
		vecgo.ReadOnly(),                           // 🆕 No WAL, purely stateless
		vecgo.WithCacheDir(searcherDir),            // Optional: specify where to cache blocks
		vecgo.WithBlockCacheSize(10 * 1024 * 1024), // Optional: tune memory usage
	}

	startOpen := time.Now()
	// OpenRemote: the source of truth IS the remote store
	eng, err := vecgo.Open(s3Store, opts...)
	if err != nil {
		log.Fatal(err)
	}
	defer eng.Close()
	fmt.Printf("⏱️  Engine Open Time: %v\n", time.Since(startOpen))

	// Try to write (should fail in read-only mode)
	_, writeErr := eng.Insert(randomVector(128), nil, nil)
	if writeErr == vecgo.ErrReadOnly {
		fmt.Println("✅ Write correctly rejected in read-only mode")
	}

	vector := randomVector(128)

	fmt.Println("\n🔎 Executing Query 1 (Cold Cache)...")
	start := time.Now()
	_, err = eng.Search(context.Background(), vector, 10, vecgo.WithNProbes(10))
	if err != nil {
		log.Printf("Query error (might be expected if index empty): %v", err)
	} else {
		fmt.Printf("⏱️  Cold Search Time: %v\n", time.Since(start))
	}

	fmt.Println("\n🔎 Executing Query 2 (Warm Cache)...")
	start = time.Now()
	_, err = eng.Search(context.Background(), vector, 10, vecgo.WithNProbes(10))
	fmt.Printf("⏱️  Warm Search Time: %v\n", time.Since(start))

	hits, misses := eng.CacheStats()
	fmt.Printf("📊 Cache Stats: Hits=%d Misses=%d\n", hits, misses)

	time.Sleep(1 * time.Second)

	fmt.Println("\n🔎 Executing Query 3 (Persistent Disk Cache)...")
	eng.Close()
	eng, _ = vecgo.Open(s3Store, opts...)

	start = time.Now()
	_, err = eng.Search(context.Background(), vector, 10, vecgo.WithNProbes(10))
	fmt.Printf("⏱️  Disk Cache Search Time: %v\n", time.Since(start))

	fmt.Println("\n✅ Demo Complete")
}

func buildIndex(dir string) {
	// For creating a NEW index, use OpenLocal with dimension/metric options
	eng, err := vecgo.Open(dir, vecgo.Create(128, vecgo.MetricL2))
	if err != nil {
		log.Fatalf("Failed to open builder: %v", err)
	}

	for i := 0; i < 2000; i++ {
		eng.Insert(randomVector(128), nil, nil)
	}
	eng.Close()
}

func copyDir(src, dst string) {
	entries, _ := os.ReadDir(src)
	for _, entry := range entries {
		data, _ := os.ReadFile(filepath.Join(src, entry.Name()))
		os.WriteFile(filepath.Join(dst, entry.Name()), data, 0644)
	}
}

func randomVector(dim int) []float32 {
	v := make([]float32, dim)
	for i := range v {
		v[i] = rand.Float32()
	}
	return v
}
