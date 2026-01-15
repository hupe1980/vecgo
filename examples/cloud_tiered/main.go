package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"time"

	"github.com/hupe1980/vecgo"
	"github.com/hupe1980/vecgo/blobstore"
	"github.com/hupe1980/vecgo/testutil"
)

// Use deterministic RNG for reproducible examples
var rng = testutil.NewRNG(42)

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

func (b *SimulatedS3Blob) ReadAt(ctx context.Context, p []byte, off int64) (int, error) {
	time.Sleep(b.latency)
	return b.Blob.ReadAt(ctx, p, off)
}

func (b *SimulatedS3Blob) ReadRange(ctx context.Context, off, len int64) (io.ReadCloser, error) {
	time.Sleep(b.latency)
	return b.Blob.ReadRange(ctx, off, len)
}

func main() {
	baseDir := "cloud_example_data"
	os.RemoveAll(baseDir)
	os.MkdirAll(baseDir, 0755)

	s3BucketDir := filepath.Join(baseDir, "s3_bucket")
	searcherDir := filepath.Join(baseDir, "searcher_state")

	os.MkdirAll(s3BucketDir, 0755)

	fmt.Println("🏗️  Building Index directly to 'S3'...")
	s3Store := NewSimulatedS3Store(s3BucketDir, 20*time.Millisecond)
	buildIndexRemote(s3Store)

	fmt.Println("🚀 Starting Stateless Search Node...")

	// Reader nodes: Use Remote() + ReadOnly() for stateless search
	opts := []vecgo.Option{
		vecgo.ReadOnly(),                           // Stateless read-only node
		vecgo.WithCacheDir(searcherDir),            // Optional: specify where to cache blocks
		vecgo.WithBlockCacheSize(10 * 1024 * 1024), // Optional: tune memory usage
	}

	startOpen := time.Now()
	// Open with Remote() backend - read-only for search nodes
	ctx := context.Background()
	eng, err := vecgo.Open(ctx, vecgo.Remote(s3Store), opts...)
	if err != nil {
		log.Fatal(err)
	}
	defer eng.Close()
	fmt.Printf("⏱️  Engine Open Time: %v\n", time.Since(startOpen))

	// Try to write (should fail in read-only mode)
	_, writeErr := eng.Insert(context.Background(), rng.UnitVector(128), nil, nil)
	if errors.Is(writeErr, vecgo.ErrReadOnly) {
		fmt.Println("✅ Write correctly rejected in read-only mode")
	}

	vector := rng.UnitVector(128)

	fmt.Println("\n🔎 Executing Query 1 (Cold Cache)...")
	start := time.Now()
	results, err := eng.Search(context.Background(), vector, 10, vecgo.WithNProbes(10))
	if err != nil {
		log.Printf("Query error (might be expected if index empty): %v", err)
	} else {
		fmt.Printf("⏱️  Cold Search Time: %v (found %d results)\n", time.Since(start), len(results))
	}

	fmt.Println("\n🔎 Executing Query 2 (Warm Cache)...")
	start = time.Now()
	results, _ = eng.Search(context.Background(), vector, 10, vecgo.WithNProbes(10))
	fmt.Printf("⏱️  Warm Search Time: %v (found %d results)\n", time.Since(start), len(results))

	hits, misses := eng.CacheStats()
	fmt.Printf("📊 Cache Stats: Hits=%d Misses=%d\n", hits, misses)

	time.Sleep(1 * time.Second)

	fmt.Println("\n🔎 Executing Query 3 (Persistent Disk Cache)...")
	eng.Close()
	eng, _ = vecgo.Open(ctx, vecgo.Remote(s3Store), opts...)

	start = time.Now()
	results, _ = eng.Search(context.Background(), vector, 10, vecgo.WithNProbes(10))
	fmt.Printf("⏱️  Disk Cache Search Time: %v (found %d results)\n", time.Since(start), len(results))

	fmt.Println("\n✅ Demo Complete")
}

func buildIndex(dir string) {
	// For creating a NEW index, use Local() backend
	// Use a tiny flush threshold to force memtable to flush to segment files
	ctx := context.Background()
	eng, err := vecgo.Open(ctx, vecgo.Local(dir),
		vecgo.Create(128, vecgo.MetricL2),
		vecgo.WithFlushConfig(vecgo.FlushConfig{MaxMemTableSize: 1}), // Force immediate flush
	)
	if err != nil {
		log.Fatalf("Failed to open builder: %v", err)
	}

	// Use testutil for reproducible vector generation
	vectors := rng.UniformVectors(2000, 128)
	for _, v := range vectors {
		eng.Insert(context.Background(), v, nil, nil)
	}

	// Give background flush loop time to persist segments
	time.Sleep(500 * time.Millisecond)
	eng.Close()
}

// buildIndexRemote builds the index directly to the cloud store.
// This demonstrates direct cloud writes via atomic Put operations.
func buildIndexRemote(store *SimulatedS3Store) {
	ctx := context.Background()
	eng, err := vecgo.Open(ctx, vecgo.Remote(store),
		vecgo.Create(128, vecgo.MetricL2),
		vecgo.WithFlushConfig(vecgo.FlushConfig{MaxMemTableSize: 1}), // Force immediate flush
	)
	if err != nil {
		log.Fatalf("Failed to open remote builder: %v", err)
	}

	// Use testutil for reproducible vector generation
	vectors := rng.UniformVectors(2000, 128)
	for _, v := range vectors {
		eng.Insert(context.Background(), v, nil, nil)
	}

	// Give background flush loop time to persist segments
	time.Sleep(500 * time.Millisecond)
	eng.Close()
	fmt.Println("✅ Built index directly to cloud store!")
}

func copyDir(src, dst string) {
	entries, _ := os.ReadDir(src)
	for _, entry := range entries {
		data, _ := os.ReadFile(filepath.Join(src, entry.Name()))
		os.WriteFile(filepath.Join(dst, entry.Name()), data, 0644)
	}
}
