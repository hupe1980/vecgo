package hnsw

import (
	"context"
	"runtime"
	"sync"
	"testing"

	"github.com/hupe1980/vecgo/model"
)

// TestConcurrentInserts verifies that multiple inserts can run concurrently without data races
func TestConcurrentInserts(t *testing.T) {
	const dim = 32

	// Reduce scale on Windows to avoid paging file exhaustion on CI runners
	numGoroutines := 10
	vectorsPerGoroutine := 100
	if runtime.GOOS == "windows" {
		numGoroutines = 4
		vectorsPerGoroutine = 25
	}

	idx, err := New(func(o *Options) {
		o.Dimension = dim
		o.M = 8
		o.EF = 200
		// Use small arena chunks to test multi-chunk allocation paths.
		// This exercises chunk boundary transitions during concurrent operations.
		// Default (1GB) would fit in single chunk, masking potential races.
		o.InitialArenaSize = 16 * 1024 * 1024 // 16MB
	})
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()

	var wg sync.WaitGroup
	wg.Add(numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		go func(offset int) {
			defer wg.Done()
			for j := 0; j < vectorsPerGoroutine; j++ {
				vector := make([]float32, dim)
				for k := range vector {
					vector[k] = float32(offset*vectorsPerGoroutine + j + k)
				}
				if _, err := idx.Insert(context.Background(), vector); err != nil {
					t.Errorf("Insert failed: %v", err)
				}
			}
		}(i)
	}

	wg.Wait()

	// Verify we have the expected number of nodes
	expectedNodes := numGoroutines * vectorsPerGoroutine
	actualNodes := idx.VectorCount()
	if actualNodes != expectedNodes {
		t.Errorf("Expected %d nodes, got %d", expectedNodes, actualNodes)
	}
}

// TestConcurrentSearches verifies that multiple searches can run concurrently
func TestConcurrentSearches(t *testing.T) {
	const dim = 32
	idx, err := New(func(o *Options) {
		o.Dimension = dim
		o.M = 8
		o.EF = 200
		o.InitialArenaSize = 64 * 1024 * 1024 // 64MB - smaller for CI
	})
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()

	const numVectors = 1000

	// Insert some vectors
	for i := 0; i < numVectors; i++ {
		vector := make([]float32, dim)
		for j := range vector {
			vector[j] = float32(i + j)
		}
		if _, err := idx.Insert(context.Background(), vector); err != nil {
			t.Fatal(err)
		}
	}

	// Now do concurrent searches
	const numGoroutines = 20
	const searchesPerGoroutine = 50

	var wg sync.WaitGroup
	wg.Add(numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		go func() {
			defer wg.Done()
			for j := 0; j < searchesPerGoroutine; j++ {
				query := make([]float32, dim)
				for k := range query {
					query[k] = float32(j + k)
				}
				if _, err := idx.KNNSearch(context.Background(), query, 10, &SearchOptions{EFSearch: 100}); err != nil {
					t.Errorf("KNNSearch failed: %v", err)
				}
			}
		}()
	}

	wg.Wait()
}

// TestConcurrentInsertsAndSearches verifies that inserts and searches can run concurrently
func TestConcurrentInsertsAndSearches(t *testing.T) {
	const dim = 32
	idx, err := New(func(o *Options) {
		o.Dimension = dim
		o.M = 8
		o.EF = 200
		o.InitialArenaSize = 64 * 1024 * 1024 // 64MB - smaller for CI
	})
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()

	const numInitialVectors = 500
	const numInsertGoroutines = 5
	const vectorsPerGoroutine = 100
	const numSearchGoroutines = 10
	const searchesPerGoroutine = 50

	// Initial population
	for i := 0; i < numInitialVectors; i++ {
		vector := make([]float32, dim)
		for j := range vector {
			vector[j] = float32(i + j)
		}
		if _, err := idx.Insert(context.Background(), vector); err != nil {
			t.Fatal(err)
		}
	}

	var wg sync.WaitGroup
	wg.Add(numInsertGoroutines + numSearchGoroutines)

	// Start inserts
	for i := 0; i < numInsertGoroutines; i++ {
		go func(offset int) {
			defer wg.Done()
			for j := 0; j < vectorsPerGoroutine; j++ {
				vector := make([]float32, dim)
				for k := range vector {
					vector[k] = float32(numInitialVectors + offset*vectorsPerGoroutine + j + k)
				}
				if _, err := idx.Insert(context.Background(), vector); err != nil {
					t.Errorf("Insert failed: %v", err)
				}
			}
		}(i)
	}

	// Start searches
	for i := 0; i < numSearchGoroutines; i++ {
		go func() {
			defer wg.Done()
			for j := 0; j < searchesPerGoroutine; j++ {
				query := make([]float32, dim)
				for k := range query {
					query[k] = float32(j + k)
				}
				if _, err := idx.KNNSearch(context.Background(), query, 10, &SearchOptions{EFSearch: 100}); err != nil {
					t.Errorf("KNNSearch failed: %v", err)
				}
			}
		}()
	}

	wg.Wait()

	// Verify final count
	expectedNodes := numInitialVectors + numInsertGoroutines*vectorsPerGoroutine
	actualNodes := idx.VectorCount()
	if actualNodes != expectedNodes {
		t.Errorf("Expected %d nodes, got %d", expectedNodes, actualNodes)
	}
}

// TestConcurrentDeletes verifies that deletes can run concurrently with searches
func TestConcurrentDeletes(t *testing.T) {
	const dim = 32
	idx, err := New(func(o *Options) {
		o.Dimension = dim
		o.M = 8
		o.EF = 200
		o.InitialArenaSize = 64 * 1024 * 1024 // 64MB - smaller for CI
	})
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()

	const numVectors = 1000

	// Insert vectors
	ids := make([]model.RowID, numVectors)
	for i := 0; i < numVectors; i++ {
		vector := make([]float32, dim)
		for j := range vector {
			vector[j] = float32(i + j)
		}
		id, err := idx.Insert(context.Background(), vector)
		if err != nil {
			t.Fatal(err)
		}
		ids[i] = id
	}

	var wg sync.WaitGroup
	wg.Add(2)

	// Deleter
	go func() {
		defer wg.Done()
		for i := 0; i < numVectors/2; i++ {
			if err := idx.Delete(context.Background(), ids[i]); err != nil {
				t.Errorf("Delete failed: %v", err)
			}
		}
	}()

	// Searcher
	go func() {
		defer wg.Done()
		for i := 0; i < 100; i++ {
			query := make([]float32, dim)
			if _, err := idx.KNNSearch(context.Background(), query, 10, nil); err != nil {
				t.Errorf("KNNSearch failed: %v", err)
			}
		}
	}()

	wg.Wait()

	// Verify count
	expectedNodes := numVectors - numVectors/2
	actualNodes := idx.VectorCount()
	if actualNodes != expectedNodes {
		t.Errorf("Expected %d nodes, got %d", expectedNodes, actualNodes)
	}
}
