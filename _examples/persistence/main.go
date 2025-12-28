package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/hupe1980/vecgo/index"
	"github.com/hupe1980/vecgo/index/hnsw"
)

func main() {
	ctx := context.Background()

	h, err := hnsw.New(func(o *hnsw.Options) {
		o.Dimension = 4
		o.M = 16
		o.EF = 64
		o.DistanceType = index.DistanceTypeSquaredL2
	})
	if err != nil {
		log.Fatalf("failed to create hnsw: %v", err)
	}

	fmt.Println("Inserting vectors...")
	vectors := [][]float32{
		{1.0, 2.0, 3.0, 4.0},
		{5.0, 6.0, 7.0, 8.0},
		{9.0, 10.0, 11.0, 12.0},
	}

	for i, vec := range vectors {
		id, err := h.Insert(ctx, vec)
		if err != nil {
			log.Fatalf("Insert failed: %v", err)
		}
		fmt.Printf("Inserted vector %d with ID %d\n", i, id)
	}

	filename := "vectors.bin"
	fmt.Printf("\nSaving to %s...\n", filename)
	if err := h.SaveToFile(filename); err != nil {
		log.Fatalf("Save failed: %v", err)
	}
	defer os.Remove(filename)

	fmt.Println("Loading from file...")
	loaded, err := hnsw.LoadFromFile(filename, hnsw.Options{
		M:            16,
		EF:           64,
		DistanceType: index.DistanceTypeSquaredL2,
	})
	if err != nil {
		log.Fatalf("Load failed: %v", err)
	}

	results, err := loaded.KNNSearch(ctx, vectors[0], 2, nil)
	if err != nil {
		log.Fatalf("Search failed: %v", err)
	}

	fmt.Printf("\nSearch results:\n")
	for i, result := range results {
		fmt.Printf("  %d. ID=%d, Distance=%.4f\n", i+1, result.ID, result.Distance)
	}

	fmt.Println("\n✅ Binary persistence works!")
}
