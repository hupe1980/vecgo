package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/hupe1980/vecgo"
)

func main() {
	dir := "./data-basic"
	os.RemoveAll(dir) // Start fresh
	defer os.RemoveAll(dir)

	// 1. Open the engine
	// Unified API: Open(backend, opts...)
	// Use Local() for filesystem, Remote() for cloud storage
	eng, err := vecgo.Open(vecgo.Local(dir), vecgo.Create(4, vecgo.MetricL2))
	if err != nil {
		log.Fatalf("Failed to open engine: %v", err)
	}
	defer eng.Close()

	// 2. Insert vectors
	vectors := [][]float32{
		{1.0, 0.0, 0.0, 0.0},
		{0.0, 1.0, 0.0, 0.0},
		{0.0, 0.0, 1.0, 0.0},
		{0.0, 0.0, 0.0, 1.0},
		{0.5, 0.5, 0.0, 0.0},
	}

	fmt.Println("Inserting vectors...")
	ctx := context.Background()
	for _, v := range vectors {
		if _, err := eng.Insert(ctx, v, nil, nil); err != nil {
			log.Fatalf("Insert failed: %v", err)
		}
	}

	// 3. Search
	query := []float32{1.0, 0.1, 0.0, 0.0}
	k := 3
	fmt.Printf("Searching for %v (k=%d)...\n", query, k)

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()

	results, err := eng.Search(ctx, query, k)
	if err != nil {
		log.Fatalf("Search failed: %v", err)
	}

	for i, res := range results {
		fmt.Printf("%d. ID: %v, Score: %.4f\n", i+1, res.ID, res.Score)
	}

	// 4. Commit (force persistence)
	if err := eng.Commit(context.Background()); err != nil {
		log.Fatalf("Commit failed: %v", err)
	}
	fmt.Println("Committed to disk.")
}
