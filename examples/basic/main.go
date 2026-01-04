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
	// Dimension 4, L2 distance
	eng, err := vecgo.Open(dir, 4, vecgo.MetricL2)
	if err != nil {
		log.Fatalf("Failed to open engine: %v", err)
	}
	defer eng.Close()

	// 2. Insert vectors
	vectors := []struct {
		id  uint64
		vec []float32
	}{
		{1, []float32{1.0, 0.0, 0.0, 0.0}},
		{2, []float32{0.0, 1.0, 0.0, 0.0}},
		{3, []float32{0.0, 0.0, 1.0, 0.0}},
		{4, []float32{0.0, 0.0, 0.0, 1.0}},
		{5, []float32{0.5, 0.5, 0.0, 0.0}},
	}

	fmt.Println("Inserting vectors...")
	for _, v := range vectors {
		if err := eng.Insert(vecgo.PrimaryKey(v.id), v.vec, nil, nil); err != nil {
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
		fmt.Printf("%d. PK: %d, Score: %.4f\n", i+1, res.PK, res.Score)
	}

	// 4. Flush (force persistence)
	if err := eng.Flush(); err != nil {
		log.Fatalf("Flush failed: %v", err)
	}
	fmt.Println("Flushed to disk.")
}
