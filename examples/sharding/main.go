package main

import (
	"context"
	"fmt"
	"log"
	"runtime"
	"sync"
	"time"

	"github.com/hupe1980/vecgo"
)

// This example demonstrates multi-core write scaling with sharding.
// Sharding eliminates the global lock bottleneck, enabling parallel writes
// for 2.7x-3.4x speedup on multi-core systems.
func main() {
	ctx := context.Background()
	dim := 128
	numVectors := 10000
	numShards := runtime.NumCPU() // Use all available cores

	fmt.Printf("Multi-Core Write Scaling Demo\n")
	fmt.Printf("=============================\n")
	fmt.Printf("Dimension: %d\n", dim)
	fmt.Printf("Vectors: %d\n", numVectors)
	fmt.Printf("CPU Cores: %d\n\n", numShards)

	// Benchmark 1: Single-shard (baseline)
	fmt.Println("--- Baseline: Single Shard (No Parallelism) ---")
	benchmarkConcurrentInserts(ctx, dim, numVectors, 1)

	// Benchmark 2: Multi-shard (parallel writes)
	fmt.Printf("\n--- Optimized: %d Shards (Parallel Writes) ---\n", numShards)
	benchmarkConcurrentInserts(ctx, dim, numVectors, numShards)

	// Demonstrate search still works correctly
	fmt.Println("\n--- Search Validation ---")
	demonstrateSearch(ctx, dim, numShards)
}

func benchmarkConcurrentInserts(ctx context.Context, dim, numVectors, numShards int) {
	// Create sharded vecgo
	vg, err := vecgo.HNSW[string](dim).
		SquaredL2().
		M(16).
		EFConstruction(200).
		Shards(numShards).
		Build()
	if err != nil {
		log.Fatalf("Failed to create vecgo: %v", err)
	}

	// Prepare vectors
	vectors := make([][]float32, numVectors)
	for i := range vectors {
		vec := make([]float32, dim)
		for j := range vec {
			vec[j] = float32(i + j)
		}
		vectors[i] = vec
	}

	// Concurrent inserts from multiple goroutines
	numWorkers := runtime.NumCPU()
	vectorsPerWorker := numVectors / numWorkers
	start := time.Now()

	var wg sync.WaitGroup
	wg.Add(numWorkers)

	for w := 0; w < numWorkers; w++ {
		workerStart := w * vectorsPerWorker
		workerEnd := workerStart + vectorsPerWorker
		if w == numWorkers-1 {
			workerEnd = numVectors // Handle remainder
		}

		go func(start, end int) {
			defer wg.Done()
			for i := start; i < end; i++ {
				_, err := vg.Insert(ctx, vecgo.VectorWithData[string]{
					Vector: vectors[i],
					Data:   fmt.Sprintf("vector_%d", i),
				})
				if err != nil {
					log.Printf("Insert error: %v", err)
				}
			}
		}(workerStart, workerEnd)
	}

	wg.Wait()
	elapsed := time.Since(start)

	throughput := float64(numVectors) / elapsed.Seconds()
	fmt.Printf("Shards: %d\n", numShards)
	fmt.Printf("Workers: %d\n", numWorkers)
	fmt.Printf("Time: %.2fs\n", elapsed.Seconds())
	fmt.Printf("Throughput: %.0f inserts/sec\n", throughput)

	// Show stats
	stats := vg.Stats()
	fmt.Printf("Final Stats: %s\n", stats.String())
}

func demonstrateSearch(ctx context.Context, dim, numShards int) {
	// Create and populate sharded index
	vg, err := vecgo.HNSW[string](dim).
		SquaredL2().
		M(16).
		EFConstruction(200).
		Shards(numShards).
		Build()
	if err != nil {
		log.Fatalf("Failed to create vecgo: %v", err)
	}

	// Insert 1000 vectors across shards
	items := make([]vecgo.VectorWithData[string], 1000)
	for i := range items {
		vec := make([]float32, dim)
		for j := range vec {
			vec[j] = float32(i + j)
		}
		items[i] = vecgo.VectorWithData[string]{
			Vector: vec,
			Data:   fmt.Sprintf("doc_%d", i),
		}
	}

	result := vg.BatchInsert(ctx, items)
	if len(result.Errors) > 0 {
		for i, err := range result.Errors {
			if err != nil {
				log.Printf("BatchInsert error at %d: %v", i, err)
			}
		}
	}

	// Search across all shards
	query := make([]float32, dim)
	for i := range query {
		query[i] = float32(i)
	}

	start := time.Now()
	results, err := vg.KNNSearch(ctx, query, 10, func(o *vecgo.KNNSearchOptions) {
		o.EF = 80
	})
	searchTime := time.Since(start)

	if err != nil {
		log.Fatalf("Search failed: %v", err)
	}

	fmt.Printf("Shards: %d\n", numShards)
	fmt.Printf("Search time: %v\n", searchTime)
	fmt.Printf("Results found: %d\n", len(results))
	fmt.Println("Top 3 results:")
	for i := 0; i < min(3, len(results)); i++ {
		fmt.Printf("  %d. ID=%d Distance=%.4f Data=%q\n",
			i+1, results[i].ID, results[i].Distance, results[i].Data)
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
