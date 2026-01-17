package vectorstore

import (
	"context"
	"testing"

	"github.com/hupe1980/vecgo/model"
)

func TestPrefetch_Basic(t *testing.T) {
	store, err := New(128, nil)
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}
	defer store.Close()

	ctx := context.Background()

	// Add some vectors
	for i := 0; i < 100; i++ {
		vec := make([]float32, 128)
		for j := range vec {
			vec[j] = float32(i + j)
		}
		if _, err := store.Append(ctx, vec); err != nil {
			t.Fatalf("failed to append vector %d: %v", i, err)
		}
	}

	// Prefetch should not panic
	ids := []model.RowID{0, 5, 10, 50, 99}
	store.Prefetch(ids)

	// Prefetch with out-of-bounds ID should not panic
	store.Prefetch([]model.RowID{0, 1000, 2000})

	// Prefetch with empty slice should not panic
	store.Prefetch(nil)
	store.Prefetch([]model.RowID{})
}

func TestPrefetchBatch_Pool(t *testing.T) {
	// Get a batch from pool
	batch := GetPrefetchBatch()
	if batch == nil {
		t.Fatal("expected non-nil batch")
	}

	// Should be empty
	if batch.Len() != 0 {
		t.Errorf("expected empty batch, got %d", batch.Len())
	}

	// Add some IDs
	batch.Add(1)
	batch.Add(5)
	batch.Add(10)

	if batch.Len() != 3 {
		t.Errorf("expected 3 IDs, got %d", batch.Len())
	}

	// Reset should clear
	batch.Reset()
	if batch.Len() != 0 {
		t.Errorf("expected empty after reset, got %d", batch.Len())
	}

	// Return to pool
	PutPrefetchBatch(batch)

	// Get again - should be reused and empty
	batch2 := GetPrefetchBatch()
	if batch2.Len() != 0 {
		t.Errorf("expected empty from pool, got %d", batch2.Len())
	}
	PutPrefetchBatch(batch2)
}

func TestPrefetchBatchFromStore(t *testing.T) {
	store, err := New(64, nil)
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}
	defer store.Close()

	ctx := context.Background()

	// Add vectors
	for i := 0; i < 50; i++ {
		vec := make([]float32, 64)
		store.Append(ctx, vec)
	}

	// Should not panic
	PrefetchBatchFromStore(store, []model.RowID{0, 10, 20})
	PrefetchBatchFromStore(nil, []model.RowID{0, 10})
	PrefetchBatchFromStore(store, nil)
}

func BenchmarkPrefetch(b *testing.B) {
	store, _ := New(768, nil)
	defer store.Close()

	ctx := context.Background()

	// Add 10K vectors
	for i := 0; i < 10000; i++ {
		vec := make([]float32, 768)
		for j := range vec {
			vec[j] = float32(i*768 + j)
		}
		store.Append(ctx, vec)
	}

	// Create typical HNSW neighbor batch (32 neighbors)
	ids := make([]model.RowID, 32)
	for i := range ids {
		ids[i] = model.RowID(i * 100)
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		store.Prefetch(ids)
	}
}

func BenchmarkPrefetchBatch_GetPut(b *testing.B) {
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		batch := GetPrefetchBatch()
		batch.Add(1)
		batch.Add(2)
		batch.Add(3)
		PutPrefetchBatch(batch)
	}
}
