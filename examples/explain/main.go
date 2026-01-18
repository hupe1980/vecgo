// Package main demonstrates Vecgo's query explanation and statistics capabilities.
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/hupe1980/vecgo"
	"github.com/hupe1980/vecgo/metadata"
)

func main() {
	dir := "./data-explain"
	os.RemoveAll(dir)
	defer os.RemoveAll(dir)

	ctx := context.Background()

	// Setup
	fmt.Println("=== Setup: Creating Index ===")

	schema := metadata.Schema{
		"category": metadata.FieldTypeString,
		"price":    metadata.FieldTypeFloat,
		"status":   metadata.FieldTypeString,
	}

	eng, err := vecgo.Open(ctx, vecgo.Local(dir),
		vecgo.Create(128, vecgo.MetricL2),
		vecgo.WithSchema(schema),
	)
	if err != nil {
		log.Fatalf("Failed to open engine: %v", err)
	}
	defer eng.Close()

	categories := []string{"electronics", "books", "clothing", "home", "sports"}
	statuses := []string{"active", "inactive", "pending"}

	fmt.Println("Inserting 1000 vectors...")
	for i := 0; i < 1000; i++ {
		vec := make([]float32, 128)
		catIdx := i % len(categories)
		vec[catIdx*10] = 1.0
		vec[catIdx*10+1] = float32(i%100) / 100.0

		doc := metadata.Document{
			"category": metadata.String(categories[catIdx]),
			"price":    metadata.Float(float64(10 + i%500)),
			"status":   metadata.String(statuses[i%len(statuses)]),
		}

		if _, err := eng.Insert(ctx, vec, doc, nil); err != nil {
			log.Fatalf("Insert failed: %v", err)
		}
	}

	if err := eng.Commit(ctx); err != nil {
		log.Fatalf("Commit failed: %v", err)
	}

	fmt.Printf("Committed. Stats: %+v\n\n", eng.Stats())

	query := make([]float32, 128)
	query[0] = 1.0

	// Example 1: Basic search with statistics
	fmt.Println("=== Example 1: Basic Search with Stats ===")

	var stats vecgo.QueryStats
	results, err := eng.Search(ctx, query, 10, vecgo.WithStats(&stats))
	if err != nil {
		log.Fatalf("Search failed: %v", err)
	}

	fmt.Printf("Results: %d\n", len(results))
	fmt.Printf("\nExplain:\n%s\n", stats.Explain())

	fmt.Printf("\nDetailed Stats:\n")
	fmt.Printf("  Strategy: %s\n", stats.Strategy)
	fmt.Printf("  Total Duration: %dμs\n", stats.TotalDurationMicros)
	fmt.Printf("  - Filter Time: %dμs\n", stats.FilterTimeMicros)
	fmt.Printf("  - Search Time: %dμs\n", stats.SearchTimeMicros)
	fmt.Printf("  - Merge Time: %dμs\n", stats.MergeTimeMicros)
	fmt.Printf("  Segments Searched: %d\n", stats.SegmentsSearched)
	fmt.Printf("  Segments Pruned: %d\n", stats.SegmentsPruned)
	fmt.Printf("  Distance Computations: %d\n", stats.DistanceComputations)
	fmt.Printf("  Short-Circuits: %d\n", stats.DistanceShortCircuits)
	fmt.Printf("  Estimated Cost: %.2f\n", stats.EstimatedCost())

	// Example 2: Filtered search statistics
	fmt.Println("\n=== Example 2: Filtered Search Stats ===")

	filter := metadata.NewFilterSet(
		metadata.Filter{Key: "category", Operator: metadata.OpEqual, Value: metadata.String("electronics")},
		metadata.Filter{Key: "status", Operator: metadata.OpEqual, Value: metadata.String("active")},
	)

	var filteredStats vecgo.QueryStats
	filteredResults, err := eng.Search(ctx, query, 10,
		vecgo.WithFilter(filter),
		vecgo.WithStats(&filteredStats),
	)
	if err != nil {
		log.Fatalf("Filtered search failed: %v", err)
	}

	fmt.Printf("Filtered Results: %d\n", len(filteredResults))
	fmt.Printf("\nExplain:\n%s\n", filteredStats.Explain())
	fmt.Printf("\nFilter Impact:\n")
	fmt.Printf("  Filter Time: %dμs (%.1f%% of total)\n",
		filteredStats.FilterTimeMicros,
		float64(filteredStats.FilterTimeMicros)/float64(filteredStats.TotalDurationMicros)*100)

	// Example 3: Range filter statistics
	fmt.Println("\n=== Example 3: Range Filter Stats ===")

	rangeFilter := metadata.NewFilterSet(
		metadata.Filter{Key: "price", Operator: metadata.OpGreaterThan, Value: metadata.Float(100)},
		metadata.Filter{Key: "price", Operator: metadata.OpLessThan, Value: metadata.Float(200)},
	)

	var rangeStats vecgo.QueryStats
	rangeResults, err := eng.Search(ctx, query, 10,
		vecgo.WithFilter(rangeFilter),
		vecgo.WithStats(&rangeStats),
	)
	if err != nil {
		log.Fatalf("Range search failed: %v", err)
	}

	fmt.Printf("Range Filter Results: %d\n", len(rangeResults))
	fmt.Printf("\nExplain:\n%s\n", rangeStats.Explain())

	// Example 4: Cost comparison
	fmt.Println("\n=== Example 4: Cost Comparison ===")

	queries := []struct {
		name   string
		filter *metadata.FilterSet
	}{
		{"No filter", nil},
		{"Equality filter", metadata.NewFilterSet(
			metadata.Filter{Key: "category", Operator: metadata.OpEqual, Value: metadata.String("electronics")},
		)},
		{"Range filter", metadata.NewFilterSet(
			metadata.Filter{Key: "price", Operator: metadata.OpGreaterThan, Value: metadata.Float(250)},
		)},
		{"Compound filter", metadata.NewFilterSet(
			metadata.Filter{Key: "category", Operator: metadata.OpEqual, Value: metadata.String("books")},
			metadata.Filter{Key: "price", Operator: metadata.OpLessThan, Value: metadata.Float(50)},
		)},
	}

	fmt.Printf("%-20s %10s %12s %10s\n", "Query Type", "Latency", "Dist Calcs", "Est Cost")
	fmt.Println("--------------------------------------------------------")

	for _, q := range queries {
		var s vecgo.QueryStats
		opts := []vecgo.SearchOption{vecgo.WithStats(&s)}
		if q.filter != nil {
			opts = append(opts, vecgo.WithFilter(q.filter))
		}

		_, err := eng.Search(ctx, query, 10, opts...)
		if err != nil {
			log.Printf("Search failed for %s: %v", q.name, err)
			continue
		}

		fmt.Printf("%-20s %8dμs %12d %10.1f\n",
			q.name, s.TotalDurationMicros, s.DistanceComputations, s.EstimatedCost())
	}

	// Example 5: Per-segment breakdown
	fmt.Println("\n=== Example 5: Per-Segment Breakdown ===")

	var segStats vecgo.QueryStats
	_, _ = eng.Search(ctx, query, 10, vecgo.WithStats(&segStats))

	if len(segStats.SegmentStats) > 0 {
		fmt.Printf("Searched %d segments:\n", len(segStats.SegmentStats))
		for i, seg := range segStats.SegmentStats {
			fmt.Printf("  Segment %d: %s index, %dμs, %d dist calcs, %d candidates\n",
				i+1, seg.IndexType, seg.DurationMicros, seg.DistanceComputations, seg.CandidatesFound)
		}
	} else {
		fmt.Println("No per-segment stats available (single segment or L0 only)")
	}

	fmt.Println("\nExplain example completed successfully!")
}
