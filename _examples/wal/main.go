package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/hupe1980/vecgo"
	"github.com/hupe1980/vecgo/wal"
)

func main() {
	ctx := context.Background()
	walPath := "./wal_example_data"

	// Clean up from previous runs
	os.RemoveAll(walPath)
	defer os.RemoveAll(walPath)

	fmt.Println("=== Vecgo WAL Durability Modes Demo ===")
	fmt.Println()

	// Example 1: Group Commit (default, recommended)
	fmt.Println("1. Group Commit Mode (default, balanced)")
	fmt.Println("   - Batches fsync operations for ~83x speedup vs sync")
	fmt.Println("   - Configurable interval and batch size")
	fmt.Println()

	db1, err := vecgo.HNSW[string](4).
		SquaredL2().
		M(16).
		EF(64).
		WAL(walPath+"/groupcommit", func(o *wal.Options) {
			o.DurabilityMode = wal.DurabilityGroupCommit
			o.GroupCommitInterval = 10 * time.Millisecond
			o.GroupCommitMaxOps = 100
		}).
		Build()
	if err != nil {
		log.Fatal(err)
	}

	start := time.Now()
	for i := 0; i < 100; i++ {
		vec := []float32{float32(i), float32(i + 1), float32(i + 2), float32(i + 3)}
		_, _ = db1.Insert(ctx, vecgo.VectorWithData[string]{
			Vector: vec,
			Data:   fmt.Sprintf("item-%d", i),
		})
	}
	fmt.Printf("   Group Commit: 100 inserts in %v\n", time.Since(start))
	db1.Close()

	// Example 2: Async Mode (fastest, but data loss risk)
	fmt.Println()
	fmt.Println("2. Async Mode (fastest, no fsync)")
	fmt.Println("   - Maximum performance")
	fmt.Println("   - Data may be lost on crash")
	fmt.Println()

	db2, err := vecgo.HNSW[string](4).
		SquaredL2().
		M(16).
		EF(64).
		WAL(walPath+"/async", func(o *wal.Options) {
			o.DurabilityMode = wal.DurabilityAsync
		}).
		Build()
	if err != nil {
		log.Fatal(err)
	}

	start = time.Now()
	for i := 0; i < 100; i++ {
		vec := []float32{float32(i), float32(i + 1), float32(i + 2), float32(i + 3)}
		_, _ = db2.Insert(ctx, vecgo.VectorWithData[string]{
			Vector: vec,
			Data:   fmt.Sprintf("item-%d", i),
		})
	}
	fmt.Printf("   Async: 100 inserts in %v\n", time.Since(start))
	db2.Close()

	// Example 3: Sync Mode (strongest durability)
	fmt.Println()
	fmt.Println("3. Sync Mode (strongest durability)")
	fmt.Println("   - fsync after every operation")
	fmt.Println("   - Slowest but safest")
	fmt.Println()

	db3, err := vecgo.HNSW[string](4).
		SquaredL2().
		M(16).
		EF(64).
		WAL(walPath+"/sync", func(o *wal.Options) {
			o.DurabilityMode = wal.DurabilitySync
		}).
		Build()
	if err != nil {
		log.Fatal(err)
	}

	start = time.Now()
	for i := 0; i < 10; i++ { // Only 10 inserts (sync is slow)
		vec := []float32{float32(i), float32(i + 1), float32(i + 2), float32(i + 3)}
		_, _ = db3.Insert(ctx, vecgo.VectorWithData[string]{
			Vector: vec,
			Data:   fmt.Sprintf("item-%d", i),
		})
	}
	fmt.Printf("   Sync: 10 inserts in %v\n", time.Since(start))
	db3.Close()

	// Example 4: Recovery demonstration
	fmt.Println()
	fmt.Println("4. Recovery Demonstration")
	fmt.Println()

	// Create new DB and recover from group commit WAL
	db4, err := vecgo.HNSW[string](4).
		SquaredL2().
		M(16).
		EF(64).
		WAL(walPath+"/groupcommit", func(o *wal.Options) {
			o.DurabilityMode = wal.DurabilityGroupCommit
		}).
		Build()
	if err != nil {
		log.Fatal(err)
	}

	if err := db4.RecoverFromWAL(ctx); err != nil {
		log.Fatal(err)
	}

	// Search to verify recovery
	query := []float32{50, 51, 52, 53}
	results, err := db4.KNNSearch(ctx, query, 3)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("   Recovered database and searched for nearest to [50,51,52,53]:")
	for i, r := range results {
		fmt.Printf("   %d. ID=%d Distance=%.2f Data=%q\n", i+1, r.ID, r.Distance, r.Data)
	}
	db4.Close()

	fmt.Println()
	fmt.Println("✅ WAL durability modes demo complete!")
}
