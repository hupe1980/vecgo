package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/hupe1980/vecgo"
	"github.com/hupe1980/vecgo/engine"
	"github.com/hupe1980/vecgo/model"
	"github.com/hupe1980/vecgo/resource"
	"github.com/hupe1980/vecgo/testutil"
)

func main() {
	dir := "./data-advanced"
	os.RemoveAll(dir)
	defer os.RemoveAll(dir)

	// Configure resource limits
	rc := resource.NewController(resource.Config{
		MemoryLimitBytes: 100 * 1024 * 1024, // 100MB limit
	})

	// Open engine with advanced options
	eng, err := vecgo.Open(dir, 128, vecgo.MetricL2,
		// Control flushing behavior
		engine.WithFlushConfig(engine.FlushConfig{
			MaxMemTableSize: 10 * 1024 * 1024, // Flush at 10MB
			MaxWALSize:      20 * 1024 * 1024, // Or 20MB WAL
		}),
		// Control compaction
		engine.WithCompactionThreshold(4), // Compact when 4 segments exist
		engine.WithCompactionConfig(engine.CompactionConfig{
			DiskANNThreshold: 10000, // Use DiskANN for segments > 10k vectors
		}),
		// Durability settings
		engine.WithWALOptions(engine.WALOptions{Durability: engine.DurabilityAsync}),
		// Resource governance
		engine.WithResourceController(rc),
	)
	if err != nil {
		log.Fatalf("Failed to open engine: %v", err)
	}
	defer eng.Close()
	rng := testutil.NewRNG(1)

	fmt.Println("Inserting 5000 vectors...")
	vec := make([]float32, 128)
	for i := 0; i < 5000; i++ {
		rng.FillUniform(vec)
		if err := eng.Insert(model.PKUint64(uint64(i)), vec, nil, nil); err != nil {
			log.Fatalf("Insert failed: %v", err)
		}
		if i%1000 == 0 {
			fmt.Printf("Inserted %d\n", i)
		}
	}

	// Wait a bit for background flush/compaction
	time.Sleep(1 * time.Second)

	fmt.Println("Searching...")
	q := make([]float32, 128)
	rng.FillUniform(q)

	start := time.Now()
	res, err := eng.Search(context.Background(), q, 10)
	if err != nil {
		log.Fatalf("Search failed: %v", err)
	}
	fmt.Printf("Found %d results in %v\n", len(res), time.Since(start))
	for i, c := range res {
		fmt.Printf("%d. PK: %v, Score: %.4f\n", i+1, c.PK, c.Score)
	}
}
