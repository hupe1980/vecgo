package vecgo_test

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/hupe1980/vecgo"
	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/wal"
)

// Example_hnswBuilder demonstrates creating an HNSW index with the fluent builder.
func Example_hnswBuilder() {
	// Create HNSW index with fluent builder
	db, err := vecgo.HNSW[string](128). // 128-dimensional vectors
						SquaredL2().         // Distance function
						M(32).               // Graph connectivity
						EFConstruction(200). // Build-time search quality
						Shards(4).           // Multi-core scaling
						Build()
	if err != nil {
		log.Fatal(err)
	}
	defer db.Close()

	fmt.Println("HNSW index created successfully")
	// Output: HNSW index created successfully
}

// Example_flatBuilder demonstrates creating a Flat index for exact search.
func Example_flatBuilder() {
	// Create Flat index with fluent builder
	db, err := vecgo.Flat[string](128). // 128-dimensional vectors
						Cosine(). // Cosine similarity
						Build()
	if err != nil {
		log.Fatal(err)
	}
	defer db.Close()

	fmt.Println("Flat index created successfully")
	// Output: Flat index created successfully
}

// Example_diskannBuilder demonstrates creating a DiskANN index for billion-scale datasets.
func Example_diskannBuilder() {
	dataPath := "./example_data"
	defer os.RemoveAll(dataPath) // Cleanup after example

	// Create DiskANN index with fluent builder
	db, err := vecgo.DiskANN[string](dataPath, 128). // Path and dimension
								SquaredL2().                // Distance function
								R(64).                      // Graph degree
								L(100).                     // Search list size
								EnableAutoCompaction(true). // Background cleanup
								CompactionThreshold(0.2).   // Compact at 20% deleted
								Build()
	if err != nil {
		log.Fatal(err)
	}
	defer db.Close()

	fmt.Println("DiskANN index created successfully")
	// Output: DiskANN index created successfully
}

// Example_insert demonstrates inserting vectors with metadata.
func Example_insert() {
	ctx := context.Background()
	db, _ := vecgo.Flat[string](3).SquaredL2().Build()
	defer db.Close()

	// Insert vector with metadata
	id, err := db.Insert(ctx, vecgo.VectorWithData[string]{
		Vector: []float32{1.0, 2.0, 3.0},
		Data:   "document-1",
		Metadata: metadata.Metadata{
			"category": metadata.String("tech"),
			"year":     metadata.Int(2024),
		},
	})
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Inserted vector with ID: %d\n", id)
	// Output: Inserted vector with ID: 0
}

// Example_batchInsert demonstrates batch insertion for better performance.
func Example_batchInsert() {
	ctx := context.Background()
	db, _ := vecgo.Flat[string](3).SquaredL2().Build()
	defer db.Close()

	// Prepare batch of vectors
	items := []vecgo.VectorWithData[string]{
		{Vector: []float32{1.0, 2.0, 3.0}, Data: "doc-1"},
		{Vector: []float32{4.0, 5.0, 6.0}, Data: "doc-2"},
		{Vector: []float32{7.0, 8.0, 9.0}, Data: "doc-3"},
	}

	// Batch insert (single lock + single WAL entry)
	result := db.BatchInsert(ctx, items)

	successful := len(result.IDs)
	fmt.Printf("Successfully inserted %d vectors\n", successful)
	// Output: Successfully inserted 3 vectors
}

// Example_search demonstrates basic KNN search.
func Example_search() {
	ctx := context.Background()
	db, _ := vecgo.Flat[string](3).SquaredL2().Build()
	defer db.Close()

	// Insert some vectors
	db.Insert(ctx, vecgo.VectorWithData[string]{
		Vector: []float32{1.0, 2.0, 3.0},
		Data:   "doc-1",
	})
	db.Insert(ctx, vecgo.VectorWithData[string]{
		Vector: []float32{1.1, 2.1, 3.1},
		Data:   "doc-2",
	})

	// Search
	query := []float32{1.0, 2.0, 3.0}
	results, err := db.KNNSearch(ctx, query, 5)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Found %d results\n", len(results))
	// Output: Found 2 results
}

// Example_streamingSearch demonstrates KNNSearchStream with early termination.
func Example_streamingSearch() {
	ctx := context.Background()
	db, _ := vecgo.Flat[string](3).SquaredL2().Build()
	defer db.Close()

	// Insert some vectors
	for i := 0; i < 100; i++ {
		db.Insert(ctx, vecgo.VectorWithData[string]{
			Vector: []float32{float32(i), float32(i + 1), float32(i + 2)},
			Data:   fmt.Sprintf("doc-%d", i),
		})
	}

	// Stream search with early termination
	query := []float32{1.0, 2.0, 3.0}
	count := 0
	threshold := float32(10.0)

	for result := range db.KNNSearchStream(ctx, query, 100) {
		if result.Distance > threshold {
			break // Stop early
		}
		count++
	}

	fmt.Printf("Found %d results within distance threshold\n", count)
	// Output: Found 3 results within distance threshold
}

// Example_wal demonstrates enabling Write-Ahead Log for durability.
func Example_wal() {
	walPath := "./example_wal"
	defer os.RemoveAll(walPath) // Cleanup after example

	db, err := vecgo.HNSW[string](128).
		SquaredL2().
		WAL(walPath, func(o *wal.Options) {
			o.DurabilityMode = wal.DurabilityGroupCommit
			o.GroupCommitInterval = 10 * time.Millisecond
			o.GroupCommitMaxOps = 100
		}).
		Build()
	if err != nil {
		log.Fatal(err)
	}
	defer db.Close()

	fmt.Println("WAL enabled successfully")
	// Output: WAL enabled successfully
}

// Example_sharding demonstrates multi-core write scaling with sharding.
func Example_sharding() {
	// Create HNSW index with 4 shards
	db, err := vecgo.HNSW[string](128).
		SquaredL2().
		Shards(4). // Enable sharding for parallel writes
		Build()
	if err != nil {
		log.Fatal(err)
	}
	defer db.Close()

	fmt.Println("Sharded HNSW index created")
	// Output: Sharded HNSW index created
}
