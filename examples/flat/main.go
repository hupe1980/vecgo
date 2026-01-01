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

	// Create Flat index using Fluent API for exact nearest neighbor search
	//
	// Performance Tips:
	// 1. Flat provides 100% recall (exact search) vs HNSW's approximate search
	// 2. Best for small datasets (<100k vectors) or when exact results are required
	// 3. SIMD distance kernels provide 10-15x speedup on ARM64/x86_64
	// 4. For multi-core scaling, add: .Shards(4)
	// 5. Consider quantization for memory reduction:
	//    - 8-bit scalar: 4x memory reduction, minimal accuracy loss
	//    - Product Quantization: 8-32x reduction, moderate accuracy loss
	//    - Binary Quantization: 32x reduction, suitable for coarse filtering
	vg, err := vecgo.Flat[int](dim).
		SquaredL2().
		Build()
	if err != nil {
		log.Fatalf("failed to create vecgo: %v", err)
	}

	rng := testutil.NewRNG(seed)

	items := make([]vecgo.VectorWithData[int], 0, size)
	for i, v := range rng.UniformVectors(size, dim) {
		items = append(items, vecgo.VectorWithData[int]{
			Vector: v,
			Data:   i,
		})
	}

	query := rng.UniformVectors(1, dim)[0]

	fmt.Println("--- Insert ---")
	fmt.Println("Dimension:", dim)
	fmt.Println("Size:", size)

	start := time.Now()

	// Use batch insert for better performance
	// Split into chunks to respect default validation limit (10,000 vectors per batch)
	batchSize := 10000
	for i := 0; i < len(items); i += batchSize {
		end := i + batchSize
		if end > len(items) {
			end = len(items)
		}
		batch := items[i:end]

		batchResult := vg.BatchInsert(context.Background(), batch)

		// Check for errors
		errorCount := 0
		for _, err := range batchResult.Errors {
			if err != nil {
				errorCount++
			}
		}
		if errorCount > 0 {
			log.Fatalf("Failed to insert vectors in batch %d: %d errors", i/batchSize, errorCount)
		}
	}

	end := time.Since(start)

	fmt.Printf("Seconds: %.2f\n\n", end.Seconds())

	fmt.Print(vg.Stats().String())
	fmt.Println()

	var result []vecgo.SearchResult[int]

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
