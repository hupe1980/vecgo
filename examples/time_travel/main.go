// Package main demonstrates Vecgo's time-travel query capabilities.
//
// Time-travel allows querying the database at historical points in time,
// useful for debugging, auditing, and A/B testing across versions.
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
	dir := "./data-time-travel"
	os.RemoveAll(dir)
	defer os.RemoveAll(dir)

	ctx := context.Background()

	// Phase 1: Create initial dataset and commit
	fmt.Println("=== Phase 1: Initial Data ===")

	eng, err := vecgo.Open(ctx, vecgo.Local(dir), vecgo.Create(4, vecgo.MetricL2))
	if err != nil {
		log.Fatalf("Failed to open engine: %v", err)
	}

	vectors := [][]float32{
		{1.0, 0.0, 0.0, 0.0},
		{0.0, 1.0, 0.0, 0.0},
		{0.0, 0.0, 1.0, 0.0},
	}

	for _, v := range vectors {
		if _, err := eng.Insert(ctx, v, nil, nil); err != nil {
			log.Fatalf("Insert failed: %v", err)
		}
	}

	if err := eng.Commit(ctx); err != nil {
		log.Fatalf("Commit failed: %v", err)
	}

	version1 := eng.Stats().ManifestID
	time1 := time.Now()
	fmt.Printf("Version %d committed at %v (3 vectors)\n", version1, time1.Format(time.RFC3339))

	time.Sleep(100 * time.Millisecond)

	// Phase 2: Add more data and commit again
	fmt.Println("\n=== Phase 2: Add More Data ===")

	moreVectors := [][]float32{
		{0.0, 0.0, 0.0, 1.0},
		{0.5, 0.5, 0.0, 0.0},
	}

	for _, v := range moreVectors {
		if _, err := eng.Insert(ctx, v, nil, nil); err != nil {
			log.Fatalf("Insert failed: %v", err)
		}
	}

	if err := eng.Commit(ctx); err != nil {
		log.Fatalf("Commit failed: %v", err)
	}

	version2 := eng.Stats().ManifestID
	time2 := time.Now()
	fmt.Printf("Version %d committed at %v (5 vectors total)\n", version2, time2.Format(time.RFC3339))

	eng.Close()

	// Phase 3: Time-travel queries
	fmt.Println("\n=== Phase 3: Time-Travel Queries ===")

	query := []float32{0.9, 0.1, 0.0, 0.0}

	// Query at Version 1
	fmt.Printf("\n--- Query at Version %d (3 vectors) ---\n", version1)
	engV1, err := vecgo.Open(ctx, vecgo.Local(dir), vecgo.WithVersion(version1))
	if err != nil {
		log.Fatalf("Failed to open at version %d: %v", version1, err)
	}

	resultsV1, err := engV1.Search(ctx, query, 5)
	if err != nil {
		log.Fatalf("Search failed: %v", err)
	}

	fmt.Printf("Found %d results:\n", len(resultsV1))
	for i, r := range resultsV1 {
		fmt.Printf("  %d. ID=%d, Score=%.4f\n", i+1, r.ID, r.Score)
	}
	engV1.Close()

	// Query at Version 2
	fmt.Printf("\n--- Query at Version %d (5 vectors) ---\n", version2)
	engV2, err := vecgo.Open(ctx, vecgo.Local(dir), vecgo.WithVersion(version2))
	if err != nil {
		log.Fatalf("Failed to open at version %d: %v", version2, err)
	}

	resultsV2, err := engV2.Search(ctx, query, 5)
	if err != nil {
		log.Fatalf("Search failed: %v", err)
	}

	fmt.Printf("Found %d results:\n", len(resultsV2))
	for i, r := range resultsV2 {
		fmt.Printf("  %d. ID=%d, Score=%.4f\n", i+1, r.ID, r.Score)
	}
	engV2.Close()

	// Query by timestamp
	fmt.Printf("\n--- Query at Timestamp %v ---\n", time1.Format(time.RFC3339))
	engT1, err := vecgo.Open(ctx, vecgo.Local(dir), vecgo.WithTimestamp(time1))
	if err != nil {
		log.Fatalf("Failed to open at timestamp: %v", err)
	}

	resultsT1, err := engT1.Search(ctx, query, 5)
	if err != nil {
		log.Fatalf("Search failed: %v", err)
	}

	fmt.Printf("Found %d results (should match version %d):\n", len(resultsT1), version1)
	for i, r := range resultsT1 {
		fmt.Printf("  %d. ID=%d, Score=%.4f\n", i+1, r.ID, r.Score)
	}
	engT1.Close()

	// Phase 4: A/B testing recall
	fmt.Println("\n=== Use Case: A/B Testing Recall ===")

	engOld, _ := vecgo.Open(ctx, vecgo.Local(dir), vecgo.WithVersion(version1))
	engNew, _ := vecgo.Open(ctx, vecgo.Local(dir), vecgo.WithVersion(version2))

	oldResults, _ := engOld.Search(ctx, query, 10)
	newResults, _ := engNew.Search(ctx, query, 10)

	oldIDs := make(map[uint64]bool)
	for _, r := range oldResults {
		oldIDs[uint64(r.ID)] = true
	}

	overlap := 0
	for _, r := range newResults {
		if oldIDs[uint64(r.ID)] {
			overlap++
		}
	}

	fmt.Printf("Recall comparison:\n")
	fmt.Printf("  Old version: %d results\n", len(oldResults))
	fmt.Printf("  New version: %d results\n", len(newResults))
	fmt.Printf("  Overlap: %d (%.0f%% of old results preserved)\n",
		overlap, float64(overlap)/float64(len(oldResults))*100)

	engOld.Close()
	engNew.Close()

	fmt.Println("\nTime-travel example completed successfully!")
}
