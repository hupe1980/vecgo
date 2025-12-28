package main

import (
	"context"
	"fmt"
	"log"
	"math/rand"

	"github.com/hupe1980/vecgo"
)

func main() {
	ctx := context.Background()

	fmt.Println("=== Vecgo Streaming Search API Demo ===")
	fmt.Println()

	// Create an HNSW index using Fluent API
	db, err := vecgo.HNSW[string](128).
		SquaredL2().
		M(16).
		EF(200).
		Build()
	if err != nil {
		log.Fatal(err)
	}
	defer db.Close()

	// Insert 1000 vectors
	fmt.Println("Inserting 1000 vectors...")
	for i := 0; i < 1000; i++ {
		vec := randomVector(128)
		_, err := db.Insert(ctx, vecgo.VectorWithData[string]{
			Vector: vec,
			Data:   fmt.Sprintf("item-%d", i),
		})
		if err != nil {
			log.Fatal(err)
		}
	}

	query := randomVector(128)

	// Example 1: Basic streaming search using Fluent API
	fmt.Println()
	fmt.Println("1. Basic Streaming Search (50 results)")
	fmt.Println("   Processing results as they arrive:")

	count := 0
	for result, err := range db.Search(query).KNN(50).Stream(ctx) {
		if err != nil {
			log.Fatal(err)
		}
		if count < 5 {
			fmt.Printf("   Result %d: ID=%d, Distance=%.4f, Data=%q\n",
				count+1, result.ID, result.Distance, result.Data)
		}
		count++
	}
	fmt.Printf("   ... and %d more results\n", count-5)

	// Example 2: Early termination with distance threshold
	fmt.Println()
	fmt.Println("2. Early Termination (stop when distance > threshold)")

	threshold := float32(500.0)
	count = 0
	for result, err := range db.Search(query).KNN(500).EF(500).Stream(ctx) {
		if err != nil {
			log.Fatal(err)
		}
		if result.Distance > threshold {
			fmt.Printf("   Stopped at distance %.2f (threshold: %.2f)\n",
				result.Distance, threshold)
			break // Early termination!
		}
		count++
	}
	fmt.Printf("   Processed %d results before threshold\n", count)

	// Example 3: Pipeline processing (find first match)
	fmt.Println()
	fmt.Println("3. Pipeline Processing (find first item with ID > 500)")

	for result, err := range db.Search(query).KNN(500).EF(500).Stream(ctx) {
		if err != nil {
			log.Fatal(err)
		}
		if result.ID > 500 {
			fmt.Printf("   Found: ID=%d, Distance=%.4f\n", result.ID, result.Distance)
			break // Found what we need, stop searching
		}
	}

	// Example 4: Collect top N with custom condition
	fmt.Println()
	fmt.Println("4. Collect Top 5 even-numbered IDs")

	var evenResults []vecgo.SearchResult[string]
	for result, err := range db.Search(query).KNN(200).EF(200).Stream(ctx) {
		if err != nil {
			log.Fatal(err)
		}
		if result.ID%2 == 0 {
			evenResults = append(evenResults, result)
			if len(evenResults) >= 5 {
				break
			}
		}
	}
	for i, r := range evenResults {
		fmt.Printf("   %d. ID=%d (even), Distance=%.4f\n", i+1, r.ID, r.Distance)
	}

	// Example 5: Streaming with options
	fmt.Println()
	fmt.Println("5. Streaming with Custom Options (EF=500)")

	count = 0
	for result, err := range db.Search(query).KNN(50).EF(500).Stream(ctx) {
		if err != nil {
			log.Fatal(err)
		}
		count++
		_ = result
	}
	fmt.Printf("   Processed %d results with EF=500\n", count)

	fmt.Println()
	fmt.Println("✅ Streaming search demo complete!")
}

func randomVector(dim int) []float32 {
	vec := make([]float32, dim)
	for i := range vec {
		vec[i] = rand.Float32()
	}
	return vec
}
