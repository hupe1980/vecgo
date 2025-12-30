// Example: DiskANN index for billion-scale vector search
// Now using integrated Vecgo API with full metadata and WAL support
package main

import (
	"context"
	"fmt"
	"os"
	"time"

	"github.com/hupe1980/vecgo"
	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/testutil"
)

func main() {
	indexPath := "./diskann_example_index"
	defer os.RemoveAll(indexPath)

	dimension := 64
	numVectors := 1000
	ctx := context.Background()

	fmt.Println("=== DiskANN Example (Integrated Vecgo API) ===")
	fmt.Printf("Dimension: %d, Vectors: %d\n\n", dimension, numVectors)

	rng := testutil.NewRNG(42)

	fmt.Println("Creating DiskANN index with fluent builder...")
	startBuild := time.Now()

	// Use integrated Vecgo API with DiskANN
	db, err := vecgo.DiskANN[string](indexPath, dimension).
		SquaredL2().
		R(32).                      // Graph degree
		L(50).                      // Build list size
		BeamWidth(4).               // Search beam width
		EnableAutoCompaction(true). // Background cleanup
		CompactionThreshold(0.2).   // Compact at 20% deleted
		Build()
	if err != nil {
		fmt.Printf("Error creating index: %v\n", err)
		return
	}
	defer db.Close()

	fmt.Println("Inserting vectors with metadata...")
	categories := []string{"tech", "science", "business"}

	vectors := rng.UniformVectors(numVectors, dimension)

	for i := 0; i < numVectors; i++ {
		vec := vectors[i]
		category := categories[i%len(categories)]

		_, err := db.Insert(ctx, vecgo.VectorWithData[string]{
			Vector: vec,
			Data:   fmt.Sprintf("Document %d", i),
			Metadata: metadata.Metadata{
				"category": metadata.String(category),
				"index":    metadata.Int(int64(i)),
			},
		})
		if err != nil {
			fmt.Printf("Error inserting vector: %v\n", err)
			return
		}
	}
	fmt.Printf("Build completed in %v\n\n", time.Since(startBuild))

	// Create query vector
	query := rng.UniformVectors(1, dimension)[0]
	k := 10

	// Standard KNN search
	fmt.Printf("KNN Search (k=%d):\n", k)
	startSearch := time.Now()
	results, err := db.Search(query).
		KNN(k).
		Execute(ctx)
	if err != nil {
		fmt.Printf("Error searching: %v\n", err)
		return
	}
	searchTime := time.Since(startSearch)

	for i, r := range results {
		fmt.Printf("  %d. Data=%s, Distance=%.4f\n", i+1, r.Data, r.Distance)
	}
	fmt.Printf("Search completed in %v\n\n", searchTime)

	// Search with metadata filtering
	fmt.Printf("Filtered Search (category='tech', k=%d):\n", k)
	filteredResults, err := db.HybridSearch(ctx, query, k, func(o *vecgo.HybridSearchOptions) {
		o.MetadataFilters = metadata.NewFilterSet(
			metadata.Filter{
				Key:      "category",
				Operator: metadata.OpEqual,
				Value:    metadata.String("tech"),
			},
		)
	})
	if err != nil {
		fmt.Printf("Error in filtered search: %v\n", err)
		return
	}

	for i, r := range filteredResults {
		fmt.Printf("  %d. Data=%s, Category=%v, Distance=%.4f\n",
			i+1, r.Data, r.Metadata["category"], r.Distance)
	}
	fmt.Println()

	// Update example
	if len(results) > 0 {
		firstID := results[0].ID
		fmt.Printf("Updating vector ID=%d...\n", firstID)
		err = db.Update(ctx, firstID, vecgo.VectorWithData[string]{
			Vector: randomVector(rng, dimension),
			Data:   "Updated Document",
			Metadata: metadata.Metadata{
				"category": metadata.String("updated"),
				"index":    metadata.Int(9999),
			},
		})
		if err != nil {
			fmt.Printf("Error updating: %v\n", err)
		} else {
			fmt.Println("Update successful!")
		}
		fmt.Println()
	}

	// Delete example
	if len(results) > 1 {
		secondID := results[1].ID
		fmt.Printf("Deleting vector ID=%d...\n", secondID)
		err = db.Delete(ctx, secondID)
		if err != nil {
			fmt.Printf("Error deleting: %v\n", err)
		} else {
			fmt.Println("Delete successful!")
		}
		fmt.Println()
	}

	fmt.Println("\n=== Summary ===")
	fmt.Println("DiskANN is now fully integrated into Vecgo!")
	fmt.Println("✅ Works with metadata filtering")
	fmt.Println("✅ Works with typed data (generics)")
	fmt.Println("✅ Full CRUD operations (Insert/Update/Delete)")
	fmt.Println("✅ Background compaction for deleted vectors")
	fmt.Println("✅ Compatible with WAL for durability")
	fmt.Println("\nPerfect for billion-scale datasets that don't fit in RAM!")
}
