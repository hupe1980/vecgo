package hnsw

import (
	"context"
	"os"
	"testing"

	"github.com/hupe1980/vecgo/index"
)

func TestBinaryPersistence_SaveLoad(t *testing.T) {
	tmpfile := "test_hnsw_binary.bin"
	defer os.Remove(tmpfile)

	// Create and populate index
	h, err := New(func(o *Options) {
		o.Dimension = 4
		o.M = 8
		o.EF = 32
		o.DistanceType = index.DistanceTypeSquaredL2
	})
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}

	// Insert test vectors
	vectors := [][]float32{
		{1.0, 2.0, 3.0, 4.0},
		{5.0, 6.0, 7.0, 8.0},
		{9.0, 10.0, 11.0, 12.0},
		{13.0, 14.0, 15.0, 16.0},
	}

	ctx := context.Background()
	for _, vec := range vectors {
		if _, err := h.Insert(ctx, vec); err != nil {
			t.Fatalf("Insert failed: %v", err)
		}
	}

	// Save to file
	if err := h.SaveToFile(tmpfile); err != nil {
		t.Fatalf("SaveToFile failed: %v", err)
	}

	// Load from file
	loaded, err := LoadFromFile(tmpfile, Options{
		Dimension:    4,
		M:            8,
		EF:           32,
		DistanceType: index.DistanceTypeSquaredL2,
	})
	if err != nil {
		t.Fatalf("LoadFromFile failed: %v", err)
	}

	// Verify loaded index
	if loaded.VectorCount() != h.VectorCount() {
		t.Errorf("Vector count mismatch: got %d, want %d", loaded.VectorCount(), h.VectorCount())
	}

	// Verify each vector
	nextID := h.nextIDAtomic.Load()
	for i := uint64(0); i < nextID; i++ {
		originalOffset := h.getNodeOffset(i)
		if originalOffset == 0 {
			if loaded.getNodeOffset(i) != 0 {
				t.Errorf("Node %d should be nil", i)
			}
			continue
		}

		loadedOffset := loaded.getNodeOffset(i)
		if loadedOffset == 0 {
			t.Errorf("Node %d should not be nil", i)
			continue
		}

		origVec, ok := h.vectors.GetVector(i)
		if !ok {
			t.Fatalf("missing original vector for id=%d", i)
		}
		loadedVec, ok := loaded.vectors.GetVector(i)
		if !ok {
			t.Fatalf("missing loaded vector for id=%d", i)
		}
		if len(loadedVec) != len(origVec) {
			t.Errorf("Node %d vector length mismatch: got %d, want %d", i, len(loadedVec), len(origVec))
			continue
		}
		for j, v := range origVec {
			if loadedVec[j] != v {
				t.Errorf("Node %d vector[%d] mismatch: got %f, want %f", i, j, loadedVec[j], v)
			}
		}
	}

	// Verify search works on loaded index
	q := []float32{1.0, 2.0, 3.0, 4.0}
	res, err := loaded.KNNSearch(ctx, q, 1, nil)
	if err != nil {
		t.Fatalf("KNNSearch failed: %v", err)
	}
	if len(res) != 1 {
		t.Errorf("Expected 1 result, got %d", len(res))
	}
	if res[0].ID != 0 {
		t.Errorf("Expected ID 0, got %d", res[0].ID)
	}
}

func TestBinaryPersistence_WithDeletions(t *testing.T) {
	tmpfile := "test_hnsw_binary_del.bin"
	defer os.Remove(tmpfile)

	h, _ := New(func(o *Options) {
		o.Dimension = 2
		o.M = 8
		o.EF = 32
	})

	ctx := context.Background()
	h.Insert(ctx, []float32{1.0, 1.0})
	id2, _ := h.Insert(ctx, []float32{2.0, 2.0})
	h.Insert(ctx, []float32{3.0, 3.0})

	// Delete middle node
	h.Delete(ctx, id2)

	if err := h.SaveToFile(tmpfile); err != nil {
		t.Fatalf("SaveToFile failed: %v", err)
	}

	loaded, err := LoadFromFile(tmpfile, Options{
		Dimension: 2,
		M:         8,
		EF:        32,
	})
	if err != nil {
		t.Fatalf("LoadFromFile failed: %v", err)
	}

	if loaded.VectorCount() != 2 {
		t.Errorf("Expected 2 vectors, got %d", loaded.VectorCount())
	}

	if loaded.ContainsID(id2) {
		t.Errorf("Deleted node %d should be nil in loaded index", id2)
	}
}

func TestBinaryPersistence_MmapLoad(t *testing.T) {
	tmpfile := "test_hnsw_mmap.bin"
	defer os.Remove(tmpfile)

	// Create and populate index
	h, err := New(func(o *Options) {
		o.Dimension = 4
		o.M = 8
		o.EF = 32
		o.DistanceType = index.DistanceTypeSquaredL2
	})
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}

	// Insert test vectors
	vectors := [][]float32{
		{1.0, 2.0, 3.0, 4.0},
		{5.0, 6.0, 7.0, 8.0},
		{9.0, 10.0, 11.0, 12.0},
		{13.0, 14.0, 15.0, 16.0},
	}

	ctx := context.Background()
	for _, vec := range vectors {
		if _, err := h.Insert(ctx, vec); err != nil {
			t.Fatalf("Insert failed: %v", err)
		}
	}

	// Save to file
	if err := h.SaveToFile(tmpfile); err != nil {
		t.Fatalf("SaveToFile failed: %v", err)
	}

	// Read file into memory (simulate mmap)
	data, err := os.ReadFile(tmpfile)
	if err != nil {
		t.Fatalf("ReadFile failed: %v", err)
	}

	// Load using Mmap loader
	loadedIdx, _, err := index.LoadBinaryIndexMmap(data)
	if err != nil {
		t.Fatalf("LoadBinaryIndexMmap failed: %v", err)
	}

	loaded, ok := loadedIdx.(*HNSW)
	if !ok {
		t.Fatalf("Loaded index is not HNSW")
	}

	// Verify loaded index
	if loaded.VectorCount() != h.VectorCount() {
		t.Errorf("Vector count mismatch: got %d, want %d", loaded.VectorCount(), h.VectorCount())
	}

	// Verify mmapOffsets are used
	if loaded.mmapOffsets == nil {
		t.Error("mmapOffsets should not be nil for mmap loaded index")
	}

	// Verify each vector
	nextID := h.nextIDAtomic.Load()
	for i := uint64(0); i < nextID; i++ {
		originalOffset := h.getNodeOffset(i)
		if originalOffset == 0 {
			if loaded.getNodeOffset(i) != 0 {
				t.Errorf("Node %d should be nil", i)
			}
			continue
		}

		loadedOffset := loaded.getNodeOffset(i)
		if loadedOffset == 0 {
			t.Errorf("Node %d should not be nil", i)
			continue
		}

		if loadedOffset != originalOffset {
			t.Errorf("Node %d offset mismatch: got %d, want %d", i, loadedOffset, originalOffset)
		}

		origVec, ok := h.vectors.GetVector(i)
		if !ok {
			t.Fatalf("missing original vector for id=%d", i)
		}
		loadedVec, ok := loaded.vectors.GetVector(i)
		if !ok {
			t.Fatalf("missing loaded vector for id=%d", i)
		}
		if len(loadedVec) != len(origVec) {
			t.Errorf("Node %d vector length mismatch: got %d, want %d", i, len(loadedVec), len(origVec))
			continue
		}
		for j, v := range origVec {
			if loadedVec[j] != v {
				t.Errorf("Node %d vector[%d] mismatch: got %f, want %f", i, j, loadedVec[j], v)
			}
		}
	}
}
