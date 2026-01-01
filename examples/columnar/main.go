// Package main demonstrates columnar storage usage.
//
// The columnar package provides high-performance vector storage with:
//   - SOA (Structure-of-Arrays) layout for optimal cache locality
//   - Zero-copy mmap loading for read-only access
//   - Soft deletes with compaction for space reclamation
package main

import (
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/hupe1980/vecgo/core"
	"github.com/hupe1980/vecgo/vectorstore/columnar"
)

func main() {
	fmt.Println("=== Vecgo Columnar Storage Demo ===")
	fmt.Println()

	// 1. Create an in-memory columnar store
	fmt.Println("1. Creating columnar store (128 dimensions)")
	store := columnar.New(128)

	// 2. Append vectors
	fmt.Println("2. Appending 1000 vectors...")
	vectors := make([][]float32, 1000)
	for i := range vectors {
		vectors[i] = make([]float32, 128)
		for j := range vectors[i] {
			vectors[i][j] = float32(i*128 + j)
		}
		if _, err := store.Append(vectors[i]); err != nil {
			log.Fatalf("Append failed: %v", err)
		}
	}
	fmt.Printf("   Count: %d, LiveCount: %d\n", store.Count(), store.LiveCount())

	// 3. Access vectors by ID
	fmt.Println("\n3. Accessing vectors by ID")
	vec, ok := store.GetVector(42)
	if !ok {
		log.Fatal("GetVector failed")
	}
	fmt.Printf("   Vector 42: [%.1f, %.1f, %.1f, ...] (len=%d)\n", vec[0], vec[1], vec[2], len(vec))

	// 4. Soft delete vectors
	fmt.Println("\n4. Soft deleting vectors 100-199")
	for i := uint64(100); i < 200; i++ {
		if err := store.DeleteVector(core.LocalID(i)); err != nil {
			log.Fatalf("DeleteVector failed: %v", err)
		}
	}
	fmt.Printf("   Count: %d, LiveCount: %d\n", store.Count(), store.LiveCount())

	// Deleted vectors return false
	_, ok = store.GetVector(150)
	fmt.Printf("   GetVector(150) exists: %v (deleted)\n", ok)

	// 5. Iterate over live vectors
	fmt.Println("\n5. Iterating over live vectors (first 5)")
	count := 0
	store.Iterate(func(id core.LocalID, vec []float32) bool {
		if count < 5 {
			fmt.Printf("   ID=%d, vec[0]=%.1f\n", id, vec[0])
		}
		count++
		return count < 5
	})

	// 6. Compact to reclaim deleted space
	fmt.Println("\n6. Compacting store")
	fmt.Printf("   Before: Count=%d, LiveCount=%d\n", store.Count(), store.LiveCount())
	idMap := store.Compact()
	fmt.Printf("   After:  Count=%d, LiveCount=%d\n", store.Count(), store.LiveCount())
	if idMap != nil {
		fmt.Printf("   ID remapping: old ID 0 -> new ID %d\n", idMap[0])
		fmt.Printf("   ID remapping: old ID 500 -> new ID %d\n", idMap[500])
	}

	// 7. Save to file
	fmt.Println("\n7. Saving to file")
	tmpDir, err := os.MkdirTemp("", "columnar-demo")
	if err != nil {
		log.Fatalf("MkdirTemp failed: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	filename := filepath.Join(tmpDir, "vectors.col")
	f, err := os.Create(filename)
	if err != nil {
		log.Fatalf("Create failed: %v", err)
	}
	n, err := store.WriteTo(f)
	f.Close()
	if err != nil {
		log.Fatalf("WriteTo failed: %v", err)
	}
	fmt.Printf("   Wrote %d bytes to %s\n", n, filename)

	// 8. Load from file
	fmt.Println("\n8. Loading from file")
	store2 := columnar.New(0)
	f2, err := os.Open(filename)
	if err != nil {
		log.Fatalf("Open failed: %v", err)
	}
	_, err = store2.ReadFrom(f2)
	f2.Close()
	if err != nil {
		log.Fatalf("ReadFrom failed: %v", err)
	}
	fmt.Printf("   Loaded store: Dimension=%d, Count=%d, LiveCount=%d\n",
		store2.Dimension(), store2.Count(), store2.LiveCount())

	// 9. Zero-copy mmap loading
	fmt.Println("\n9. Zero-copy mmap loading")
	mmapStore, closer, err := columnar.OpenMmap(filename)
	if err != nil {
		log.Fatalf("OpenMmap failed: %v", err)
	}
	defer closer.Close()

	fmt.Printf("   Mmap store: Dimension=%d, Count=%d, LiveCount=%d\n",
		mmapStore.Dimension(), mmapStore.Count(), mmapStore.LiveCount())

	// Access vector from mmap (direct pointer to file, no copy)
	mmapVec, ok := mmapStore.GetVector(0)
	if !ok {
		log.Fatal("Mmap GetVector failed")
	}
	fmt.Printf("   Mmap vector 0: [%.1f, %.1f, %.1f, ...]\n", mmapVec[0], mmapVec[1], mmapVec[2])

	// Mmap store is read-only
	err = mmapStore.SetVector(0, mmapVec)
	fmt.Printf("   SetVector on mmap: error=%v (expected: read-only)\n", err != nil)

	// 10. Raw data access for batch operations
	fmt.Println("\n10. Raw data access (contiguous memory)")
	rawData := store.RawData()
	fmt.Printf("   Raw data length: %d floats (%d vectors × %d dimensions)\n",
		len(rawData), store.Count(), store.Dimension())

	fmt.Println("\n✅ Columnar storage demo complete!")
}
