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
	// Configuration
	dim := 128
	size := 50000
	k := 10

	fmt.Println("=== Index Performance Comparison ===")
	fmt.Printf("Dataset: %d vectors, %d dimensions\n", size, dim)
	fmt.Println("Generating vectors...")

	rng := testutil.NewRNG(0)
	vectors := rng.UniformVectors(size, dim)
	query := rng.UniformVectors(1, dim)[0]

	// ---------------------------------------------------------
	// 1. Flat Index (Brute Force)
	// ---------------------------------------------------------
	fmt.Println("\n--- 1. Flat Index (Baseline) ---")
	flatIndex, err := vecgo.Flat[int](dim).SquaredL2().Build()
	if err != nil {
		log.Fatal(err)
	}
	defer flatIndex.Close()

	// Batch insert
	start := time.Now()
	batchSize := 10000
	for i := 0; i < size; i += batchSize {
		end := i + batchSize
		if end > size {
			end = size
		}
		batch := make([]vecgo.VectorWithData[int], end-i)
		for j, vec := range vectors[i:end] {
			batch[j] = vecgo.VectorWithData[int]{Vector: vec, Data: i + j}
		}
		res := flatIndex.BatchInsert(context.Background(), batch)
		for _, err := range res.Errors {
			if err != nil {
				log.Fatal(err)
			}
		}
	}
	fmt.Printf("Build Time: %.2fs\n", time.Since(start).Seconds())

	// Search
	start = time.Now()
	_, _ = flatIndex.Search(query).KNN(k).Execute(context.Background())
	flatDuration := time.Since(start)
	fmt.Printf("Search Latency: %v\n", flatDuration)

	// ---------------------------------------------------------
	// 2. HNSW Index (Approximate)
	// ---------------------------------------------------------
	fmt.Println("\n--- 2. HNSW Index (Optimized) ---")
	hnswIndex, err := vecgo.HNSW[int](dim).
		SquaredL2().
		M(16).
		EFConstruction(200).
		Build()
	if err != nil {
		log.Fatal(err)
	}
	defer hnswIndex.Close()

	// Batch insert
	start = time.Now()
	for i := 0; i < size; i += batchSize {
		end := i + batchSize
		if end > size {
			end = size
		}
		batch := make([]vecgo.VectorWithData[int], end-i)
		for j, vec := range vectors[i:end] {
			batch[j] = vecgo.VectorWithData[int]{Vector: vec, Data: i + j}
		}
		res := hnswIndex.BatchInsert(context.Background(), batch)
		for _, err := range res.Errors {
			if err != nil {
				log.Fatal(err)
			}
		}
	}
	fmt.Printf("Build Time: %.2fs\n", time.Since(start).Seconds())

	// Search
	start = time.Now()
	_, _ = hnswIndex.Search(query).KNN(k).Execute(context.Background())
	hnswDuration := time.Since(start)
	fmt.Printf("Search Latency: %v\n", hnswDuration)

	// ---------------------------------------------------------
	// Summary
	// ---------------------------------------------------------
	fmt.Printf("\nSummary:\n")
	fmt.Printf("Flat Latency: %v\n", flatDuration)
	fmt.Printf("HNSW Latency: %v\n", hnswDuration)
	if hnswDuration < flatDuration {
		fmt.Printf("Speedup: %.2fx\n", float64(flatDuration)/float64(hnswDuration))
	} else {
		fmt.Printf("Slowdown: %.2fx (HNSW overhead dominates at this scale/config)\n", float64(hnswDuration)/float64(flatDuration))
	}
}
