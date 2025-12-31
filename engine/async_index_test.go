package engine

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/hupe1980/vecgo/codec"
	"github.com/hupe1980/vecgo/index"
	"github.com/hupe1980/vecgo/index/hnsw"
	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func setupAsyncTest(t *testing.T) (*Tx[string], *hnsw.HNSW) {
	idx, err := hnsw.New(func(o *hnsw.Options) {
		o.Dimension = 4
		o.DistanceType = index.DistanceTypeSquaredL2
		o.M = 16
		o.EF = 100
	})
	require.NoError(t, err)

	dataStore := NewMapStore[string]()
	metaStore := metadata.NewUnifiedIndex()

	// Use NoopDurability for speed, we are testing memory/index logic
	durability := NoopDurability{}

	coord, err := New[string](idx, dataStore, metaStore, durability, codec.Default, WithDimension(4))
	require.NoError(t, err)

	tx, ok := coord.(*Tx[string])
	require.True(t, ok)

	return tx, idx
}

func TestAsyncIndex_InsertAndSearch(t *testing.T) {
	tx, _ := setupAsyncTest(t)
	defer tx.Close()
	ctx := context.Background()

	// 1. Insert into MemTable
	vec := []float32{0.1, 0.1, 0.1, 0.1}
	id, err := tx.Insert(ctx, vec, "test", nil)
	require.NoError(t, err)

	// 2. Search immediately (should be in MemTable)
	results, err := tx.HybridSearch(ctx, vec, 1, nil)
	require.NoError(t, err)
	require.Len(t, results, 1)
	assert.Equal(t, id, results[0].ID)
	assert.Equal(t, float32(0.0), results[0].Distance)

	// 3. Flush MemTable
	tx.flushMemTable()

	// 4. Search again (should be in HNSW)
	results, err = tx.HybridSearch(ctx, vec, 1, nil)
	require.NoError(t, err)
	require.Len(t, results, 1)
	assert.Equal(t, id, results[0].ID)
}

func TestAsyncIndex_DeleteFromMemTable(t *testing.T) {
	tx, _ := setupAsyncTest(t)
	defer tx.Close()
	ctx := context.Background()

	vec := []float32{0.1, 0.1, 0.1, 0.1}
	id, err := tx.Insert(ctx, vec, "test", nil)
	require.NoError(t, err)

	// Verify it's there
	results, err := tx.HybridSearch(ctx, vec, 1, nil)
	require.Len(t, results, 1)

	// Delete (should remove from MemTable)
	err = tx.Delete(ctx, id)
	require.NoError(t, err)

	// Verify it's gone
	results, err = tx.HybridSearch(ctx, vec, 1, nil)
	require.NoError(t, err)
	require.Len(t, results, 0)
}

func TestAsyncIndex_DeleteAfterFlush(t *testing.T) {
	tx, _ := setupAsyncTest(t)
	defer tx.Close()
	ctx := context.Background()

	vec := []float32{0.1, 0.1, 0.1, 0.1}
	id, err := tx.Insert(ctx, vec, "test", nil)
	require.NoError(t, err)

	tx.flushMemTable()

	// Verify it's in index
	results, err := tx.HybridSearch(ctx, vec, 1, nil)
	require.Len(t, results, 1)

	// Delete
	err = tx.Delete(ctx, id)
	require.NoError(t, err)

	// Verify it's gone
	results, err = tx.HybridSearch(ctx, vec, 1, nil)
	require.NoError(t, err)
	require.Len(t, results, 0)
}

func TestAsyncIndex_Race_FlushAndDelete(t *testing.T) {
	tx, _ := setupAsyncTest(t)
	defer tx.Close()
	ctx := context.Background()

	var wg sync.WaitGroup
	n := 1000

	rng := testutil.NewRNG(1)

	// Insert items
	ids := make([]uint64, n)
	for i := 0; i < n; i++ {
		vec := make([]float32, 4)
		for j := 0; j < 4; j++ {
			vec[j] = rng.Float32()
		}
		id, err := tx.Insert(ctx, vec, fmt.Sprintf("item-%d", i), nil)
		require.NoError(t, err)
		ids[i] = id
	}

	// Concurrent Flush and Delete
	wg.Add(2)

	go func() {
		defer wg.Done()
		// Trigger flushes repeatedly
		for i := 0; i < 10; i++ {
			tx.flushMemTable()
			time.Sleep(1 * time.Millisecond)
		}
	}()

	go func() {
		defer wg.Done()
		// Delete half the items
		for i := 0; i < n/2; i++ {
			err := tx.Delete(ctx, ids[i])
			// We expect no error, but if race condition exists, it might fail
			if err != nil {
				t.Errorf("Delete failed for id %d: %v", ids[i], err)
			}
		}
	}()

	wg.Wait()

	// Verify remaining items are searchable
	for i := n / 2; i < n; i++ {
		// Use public API to verify, or lock manually if using internal
		// Since vectorByIDLocked is internal and requires lock, we can't easily use it here without exposing lock.
		// But we can use HybridSearch directly if we know the vector.
		// Wait, we need the vector to search.
		// We can get it from the original slice if we stored it, but we didn't store vectors in a map, just ids.
		// Let's just use tx.txIndex.VectorByID which is public on the index, but might miss MemTable items?
		// No, we want to check if it's in the system (MemTable OR Index).

		// Let's use a helper that locks
		vec, err := tx.vectorByID(ctx, ids[i])

		require.NoError(t, err)
		results, err := tx.HybridSearch(ctx, vec, 1, nil)
		require.NoError(t, err)
		if len(results) == 0 {
			t.Errorf("Item %d not found after race test", ids[i])
		}
	}
}

func TestAsyncIndex_EntryPointDeletion(t *testing.T) {
	// This test specifically targets the "entry point deleted" error
	tx, _ := setupAsyncTest(t)
	defer tx.Close()
	ctx := context.Background()

	// 1. Insert Entry Point
	epVec := []float32{0, 0, 0, 0}
	epID, err := tx.Insert(ctx, epVec, "ep", nil)
	require.NoError(t, err)

	// Force flush to make it the HNSW entry point
	tx.flushMemTable()

	// 2. Insert many items into MemTable (referencing EP)
	n := 100
	for i := 0; i < n; i++ {
		vec := []float32{0.1, 0.1, 0.1, 0.1}
		_, err := tx.Insert(ctx, vec, "item", nil)
		require.NoError(t, err)
	}

	// 3. Concurrent Delete(EP) and Flush
	var wg sync.WaitGroup
	wg.Add(2)

	go func() {
		defer wg.Done()
		// Flush the new items. They will try to traverse from EP.
		tx.flushMemTable()
	}()

	go func() {
		defer wg.Done()
		// Delete the entry point
		time.Sleep(100 * time.Microsecond) // Try to hit the race window
		err := tx.Delete(ctx, epID)
		if err != nil {
			t.Logf("Delete EP failed: %v", err)
		}
	}()

	wg.Wait()

	// If the flush failed silently or crashed, we might have missing items.
	// But mainly we want to ensure no panic and no "entry point deleted" error logged (if we can catch it).
	// The previous implementation printed to stdout, so we might see it there.
}
