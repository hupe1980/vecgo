package main

import (
	"context"
	"fmt"
	"log/slog"
	"time"

	"github.com/hupe1980/vecgo"
)

func main() {
	ctx := context.Background()

	// Create logger and metrics collector
	logger := vecgo.NewTextLogger(slog.LevelInfo)
	metrics := &vecgo.BasicMetricsCollector{}

	// Create HNSW index with logging and metrics
	vg, err := vecgo.HNSW[string](128).
		SquaredL2().
		Logger(logger).
		Metrics(metrics).
		Build()
	if err != nil {
		panic(err)
	}
	defer vg.Close()

	// Insert some vectors
	fmt.Println("Inserting vectors...")
	for i := 0; i < 100; i++ {
		vec := make([]float32, 128)
		for j := range vec {
			vec[j] = float32(i + j)
		}
		_, err := vg.Insert(ctx, vecgo.VectorWithData[string]{
			Vector: vec,
			Data:   fmt.Sprintf("item-%d", i),
		})
		if err != nil {
			fmt.Printf("Insert failed: %v\n", err)
		}
	}

	// Perform searches
	fmt.Println("\nPerforming searches...")
	query := make([]float32, 128)
	for j := range query {
		query[j] = float32(50 + j)
	}
	for i := 0; i < 10; i++ {
		results, err := vg.KNNSearch(ctx, query, 5)
		if err != nil {
			fmt.Printf("Search failed: %v\n", err)
		} else {
			fmt.Printf("Search %d: Found %d results\n", i+1, len(results))
		}
	}

	// Get metrics summary
	fmt.Println("\n=== Metrics Summary ===")
	stats := metrics.GetStats()
	fmt.Printf("Total inserts:       %d\n", stats.InsertCount)
	fmt.Printf("Insert errors:       %d\n", stats.InsertErrors)
	fmt.Printf("Avg insert latency:  %s\n", time.Duration(stats.InsertAvgNanos))
	fmt.Printf("Total searches:      %d\n", stats.SearchCount)
	fmt.Printf("Search errors:       %d\n", stats.SearchErrors)
	fmt.Printf("Avg search latency:  %s\n", time.Duration(stats.SearchAvgNanos))
}
