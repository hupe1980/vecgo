package engine

import (
	"sync"
	"testing"
)

func TestMapStore(t *testing.T) {
	st := NewMapStore[string]()

	// Test Set and Get
	if err := st.Set(1, "one"); err != nil {
		t.Fatalf("Set failed: %v", err)
	}

	val, ok := st.Get(1)
	if !ok || val != "one" {
		t.Fatalf("Get failed: expected 'one', got '%s', ok=%v", val, ok)
	}

	// Test Get non-existent
	_, ok = st.Get(999)
	if ok {
		t.Fatal("Get should return false for non-existent ID")
	}

	// Test BatchSet
	batch := map[uint64]string{
		2: "two",
		3: "three",
		4: "four",
	}
	if err := st.BatchSet(batch); err != nil {
		t.Fatalf("BatchSet failed: %v", err)
	}

	if st.Len() != 4 {
		t.Fatalf("Len should be 4, got %d", st.Len())
	}

	// Test BatchGet
	results, err := st.BatchGet([]uint64{1, 2, 3, 999})
	if err != nil {
		t.Fatalf("BatchGet failed: %v", err)
	}

	if len(results) != 3 {
		t.Fatalf("BatchGet should return 3 items, got %d", len(results))
	}

	if results[1] != "one" || results[2] != "two" || results[3] != "three" {
		t.Fatal("BatchGet returned incorrect values")
	}

	// Test Delete
	if err := st.Delete(1); err != nil {
		t.Fatalf("Delete failed: %v", err)
	}

	_, ok = st.Get(1)
	if ok {
		t.Fatal("Get should return false after Delete")
	}

	// Test Delete non-existent
	if err := st.Delete(999); err == nil {
		t.Fatal("Delete should return error for non-existent ID")
	}

	// Test BatchDelete
	if err := st.BatchDelete([]uint64{2, 3}); err != nil {
		t.Fatalf("BatchDelete failed: %v", err)
	}

	if st.Len() != 1 {
		t.Fatalf("Len should be 1 after BatchDelete, got %d", st.Len())
	}

	// Test ToMap
	storeMap := st.ToMap()
	if len(storeMap) != 1 {
		t.Fatalf("ToMap should return 1 item, got %d", len(storeMap))
	}

	if storeMap[4] != "four" {
		t.Fatalf("ToMap returned incorrect value: %v", storeMap)
	}

	// Test Clear
	if err := st.Clear(); err != nil {
		t.Fatalf("Clear failed: %v", err)
	}

	if st.Len() != 0 {
		t.Fatalf("Len should be 0 after Clear, got %d", st.Len())
	}
}

func TestMapStoreConcurrent(t *testing.T) {
	st := NewMapStore[int]()
	var wg sync.WaitGroup

	// Concurrent writes
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			_ = st.Set(uint64(id), id*2)
		}(i)
	}

	// Concurrent reads
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			_, _ = st.Get(uint64(id))
		}(i)
	}

	wg.Wait()

	if st.Len() != 100 {
		t.Fatalf("Expected 100 items, got %d", st.Len())
	}
}
