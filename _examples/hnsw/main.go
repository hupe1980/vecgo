package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/hupe1980/vecgo"
	"github.com/hupe1980/vecgo/testutil"
)

func main() {
	seed := int64(4711)
	dim := 32
	size := 50000
	k := 10

	// Create HNSW index using Fluent API
	//
	// Storage: HNSW uses columnar vector storage (SOA layout) by default
	// for optimal cache locality and SIMD performance.
	//
	// Performance Tips:
	// 1. For multi-core write scaling (2.7x-3.4x speedup), add: .Shards(4)
	//    - Eliminates global lock bottleneck
	//    - Enables parallel writes across shards
	//    - Search automatically fans out to all shards in parallel
	//
	// 2. Tune M and EF for your accuracy/speed trade-off:
	//    - M: Number of connections per layer (default 16)
	//      Higher M = better recall but more memory
	//    - EF: Search depth (default 200)
	//      Higher EF = better recall but slower search
	//
	// 3. SIMD distance kernels are automatically enabled on ARM64/x86_64
	//    - 10-15x faster than naive Go loops
	//    - Zero allocations (pooled buffers)
	vg, err := vecgo.HNSW[int](dim).
		SquaredL2().
		M(32). // Increase connections for better recall
		Build()
	if err != nil {
		log.Fatalf("failed to create vecgo: %v", err)
	}

	rng := testutil.NewRNG(seed)

	items := make([]vecgo.VectorWithData[int], 0, size)
	for i, v := range rng.GenerateRandomVectors(size, dim) {
		items = append(items, vecgo.VectorWithData[int]{
			Vector: v,
			Data:   i,
		})
	}

	query := rng.GenerateRandomVectors(1, dim)[0]

	fmt.Println("--- Insert ---")
	fmt.Println("Dimension:", dim)
	fmt.Println("Size:", size)

	start := time.Now()

	// Use batch insert for better performance
	batchResult := vg.BatchInsert(context.Background(), items)

	// Check for errors
	errorCount := 0
	for _, err := range batchResult.Errors {
		if err != nil {
			errorCount++
		}
	}
	if errorCount > 0 {
		log.Printf("Failed to insert %d vectors", errorCount)
	}

	end := time.Since(start)

	fmt.Printf("Seconds: %.2f\n\n", end.Seconds())

	fmt.Print(vg.Stats().String())
	fmt.Println()

	var result []vecgo.SearchResult[int]

	fmt.Println("--- KNN (Fluent API) ---")

	start = time.Now()

	// Fluent search API: clean, chainable, discoverable
	result, err = vg.Search(query).
		KNN(k).
		EF(80).
		Execute(context.Background())
	if err != nil {
		log.Fatal(err)
	}

	end = time.Since(start)

	printResult(result)

	fmt.Printf("Seconds: %.8f\n\n", end.Seconds())

	fmt.Println("--- Brute ---")

	start = time.Now()

	result, err = vg.BruteSearch(context.Background(), query, k)
	if err != nil {
		log.Fatal(err)
	}

	end = time.Since(start)

	printResult(result)

	fmt.Printf("Seconds: %.8f\n\n", end.Seconds())
}

func printResult[T any](result []vecgo.SearchResult[T]) {
	for _, r := range result {
		fmt.Printf("ID: %d, Distance: %.2f, Data: %v\n", r.ID, r.Distance, r.Data)
	}
}
