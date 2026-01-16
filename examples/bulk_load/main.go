// Package main demonstrates Vecgo's bulk loading capabilities.
//
// This example shows how to use BatchInsertDeferred for high-throughput
// data loading — approximately 1000x faster than standard insert.
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/hupe1980/vecgo"
	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/testutil"
)

const (
	dimension = 768 // OpenAI text-embedding-3-small dimension
	totalVecs = 10000
	batchSize = 1000
	searchK   = 10
)

func main() {
	dir := "./data-bulk-load"
	os.RemoveAll(dir)
	defer os.RemoveAll(dir)

	ctx := context.Background()

	// Create engine
	eng, err := vecgo.Open(ctx, vecgo.Local(dir), vecgo.Create(dimension, vecgo.MetricL2))
	if err != nil {
		log.Fatalf("Failed to open: %v", err)
	}
	defer eng.Close()

	// Generate test data
	rng := testutil.NewRNG(42)
	vectors := make([][]float32, batchSize)
	metadatas := make([]metadata.Document, batchSize)

	for i := range vectors {
		vec := make([]float32, dimension)
		rng.FillUniform(vec)
		vectors[i] = vec
		metadatas[i] = metadata.Document{
			"batch": metadata.Int(0),
			"index": metadata.Int(int64(i)),
		}
	}

	// =========================================================================
	// Bulk Loading with BatchInsertDeferred
	// =========================================================================
	fmt.Println("=== Bulk Loading Demo ===")
	fmt.Printf("Loading %d vectors in batches of %d...\n\n", totalVecs, batchSize)

	start := time.Now()
	totalInserted := 0

	for batch := 0; batch < totalVecs/batchSize; batch++ {
		// Update metadata batch number
		for i := range metadatas {
			metadatas[i]["batch"] = metadata.Int(int64(batch))
		}

		// BatchInsertDeferred: NO HNSW indexing, just columnar storage
		// This is ~1000x faster than BatchInsert
		ids, err := eng.BatchInsertDeferred(ctx, vectors, metadatas, nil)
		if err != nil {
			log.Fatalf("BatchInsertDeferred failed: %v", err)
		}

		totalInserted += len(ids)

		if (batch+1)%5 == 0 {
			elapsed := time.Since(start)
			rate := float64(totalInserted) / elapsed.Seconds()
			fmt.Printf("  Batch %d: %d vectors, %.0f vec/s\n", batch+1, totalInserted, rate)
		}
	}

	loadTime := time.Since(start)
	loadRate := float64(totalInserted) / loadTime.Seconds()
	fmt.Printf("\n✅ Loaded %d vectors in %v (%.0f vec/s)\n", totalInserted, loadTime, loadRate)

	// =========================================================================
	// Commit: Flush to DiskANN segment
	// =========================================================================
	fmt.Println("\n=== Committing to DiskANN ===")
	fmt.Println("NOTE: Vectors are NOT searchable until commit!")

	commitStart := time.Now()
	if err := eng.Commit(ctx); err != nil {
		log.Fatalf("Commit failed: %v", err)
	}
	commitTime := time.Since(commitStart)
	fmt.Printf("✅ Committed in %v\n", commitTime)

	// =========================================================================
	// Search: Now vectors are searchable via DiskANN
	// =========================================================================
	fmt.Println("\n=== Searching (via DiskANN) ===")

	query := make([]float32, dimension)
	rng.FillUniform(query)

	searchStart := time.Now()
	results, err := eng.Search(ctx, query, searchK)
	if err != nil {
		log.Fatalf("Search failed: %v", err)
	}
	searchTime := time.Since(searchStart)

	fmt.Printf("Found %d results in %v:\n", len(results), searchTime)
	for i, r := range results[:min(5, len(results))] {
		batch, _ := r.Metadata["batch"].AsInt64()
		fmt.Printf("  %d. ID=%d, Score=%.4f, Batch=%d\n",
			i+1, r.ID, r.Score, batch)
	}

	// =========================================================================
	// Comparison: Standard BatchInsert (for reference)
	// =========================================================================
	fmt.Println("\n=== Comparison: Standard BatchInsert ===")

	// Create new engine for comparison
	dir2 := "./data-bulk-load-standard"
	os.RemoveAll(dir2)
	defer os.RemoveAll(dir2)

	eng2, _ := vecgo.Open(ctx, vecgo.Local(dir2), vecgo.Create(dimension, vecgo.MetricL2))
	defer eng2.Close()

	// Insert just 1 batch with standard method
	standardStart := time.Now()
	_, err = eng2.BatchInsert(ctx, vectors, metadatas, nil)
	if err != nil {
		log.Fatalf("BatchInsert failed: %v", err)
	}
	standardTime := time.Since(standardStart)
	standardRate := float64(batchSize) / standardTime.Seconds()

	fmt.Printf("Standard BatchInsert: %d vectors in %v (%.0f vec/s)\n", batchSize, standardTime, standardRate)
	fmt.Printf("Deferred speedup: %.1fx faster\n", loadRate/standardRate)

	// =========================================================================
	// Summary
	// =========================================================================
	fmt.Println("\n=== Summary ===")
	fmt.Printf("BatchInsertDeferred: %.0f vec/s (vectors searchable after Commit)\n", loadRate)
	fmt.Printf("BatchInsert:         %.0f vec/s (vectors searchable immediately)\n", standardRate)

	fmt.Println("\nUse BatchInsertDeferred for:")
	fmt.Println("  - Initial data loading")
	fmt.Println("  - Database migrations")
	fmt.Println("  - Nightly reindex jobs")

	fmt.Println("\nUse BatchInsert for:")
	fmt.Println("  - Real-time RAG (immediate searchability)")
	fmt.Println("  - Interactive applications")
}
