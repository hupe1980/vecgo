package integration_test

import (
	"context"
	"log/slog"
	"os"
	"testing"
	"time"

	"github.com/hupe1980/vecgo"
	"github.com/hupe1980/vecgo/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestMixedSegments tests searching across multiple segment types:
// - MemTable (in-memory)
// - Flat segments (flushed disk segments)
// - DiskANN segments (compacted segments)
func TestMixedSegments(t *testing.T) {
	dir := t.TempDir()
	dim := 4

	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelInfo}))
	compactionCfg := vecgo.CompactionConfig{
		DiskANNThreshold: 5, // Low threshold for testing
	}

	db, err := vecgo.Open(vecgo.Local(dir),
		vecgo.Create(dim, vecgo.MetricL2),
		vecgo.WithLogger(logger),
		vecgo.WithCompactionConfig(compactionCfg),
		vecgo.WithCompactionThreshold(2),
	)
	require.NoError(t, err)
	defer db.Close()

	// Phase 1: Insert and flush first batch
	for i := 1; i <= 5; i++ {
		vec := make([]float32, dim)
		for j := range vec {
			vec[j] = float32(i - 1)
		}
		_, err := db.Insert(context.Background(), vec, nil, nil)
		require.NoError(t, err)
	}
	require.NoError(t, db.Commit(context.Background()))
	t.Log("Flushed segment 1")

	// Phase 2: Insert and flush second batch
	for i := 6; i <= 10; i++ {
		vec := make([]float32, dim)
		for j := range vec {
			vec[j] = float32(i - 1)
		}
		_, err := db.Insert(context.Background(), vec, nil, nil)
		require.NoError(t, err)
	}
	require.NoError(t, db.Commit(context.Background()))
	t.Log("Flushed segment 2")

	// Wait for compaction
	time.Sleep(500 * time.Millisecond)

	// Verify IDs 1-10 are accessible after compaction
	t.Log("Verifying IDs 1-10 via Get:")
	for id := model.ID(1); id <= 10; id++ {
		rec, err := db.Get(context.Background(), id)
		require.NoError(t, err, "Get ID=%d after compaction", id)
		expectedVal := float32(id - 1)
		expected := []float32{expectedVal, expectedVal, expectedVal, expectedVal}
		if !assert.Equal(t, expected, rec.Vector, "ID=%d vector mismatch after compaction", id) {
			t.Logf("  ID=%d expected=%v got=%v", id, expected, rec.Vector)
		} else {
			t.Logf("  ID=%d OK vec=%v", id, rec.Vector)
		}
	}

	// DON'T add more data - just search the DiskANN segment directly
	// This isolates the DiskANN search issue

	// Search should find results from all segments
	query := make([]float32, dim)
	for j := range query {
		query[j] = 0 // Close to ID=1's vector
	}

	results, err := db.Search(context.Background(), query, 5)
	require.NoError(t, err)

	t.Logf("Search found %d results", len(results))
	for _, r := range results {
		t.Logf("  ID=%d score=%.4f", r.ID, r.Score)
	}

	assert.Len(t, results, 5)
	assert.Equal(t, model.ID(1), results[0].ID) // Should be closest
}

// TestSimpleFlushCompaction tests basic flush and compaction without sharding complexity
func TestSimpleFlushCompaction(t *testing.T) {
	dir := t.TempDir()
	dim := 4

	// DiskANN threshold at 10 to force DiskANN creation
	compactionCfg := vecgo.CompactionConfig{
		DiskANNThreshold: 10,
	}

	// Enable logging to see compaction details
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelDebug}))

	e, err := vecgo.Open(vecgo.Local(dir), vecgo.Create(dim, vecgo.MetricL2),
		vecgo.WithCompactionConfig(compactionCfg),
		vecgo.WithCompactionThreshold(2),
		vecgo.WithLogger(logger),
	)
	require.NoError(t, err)
	defer e.Close()

	// Insert 5 vectors
	for i := 0; i < 5; i++ {
		vec := []float32{float32(i), float32(i), float32(i), float32(i)}
		id, err := e.Insert(context.Background(), vec, nil, nil)
		require.NoError(t, err)
		t.Logf("Inserted ID=%d vec=%v", id, vec)
	}

	// Flush
	err = e.Commit(context.Background())
	require.NoError(t, err)
	t.Log("Flushed segment 1")

	// Insert 5 more
	for i := 5; i < 10; i++ {
		vec := []float32{float32(i), float32(i), float32(i), float32(i)}
		id, err := e.Insert(context.Background(), vec, nil, nil)
		require.NoError(t, err)
		t.Logf("Inserted ID=%d vec=%v", id, vec)
	}

	// Flush
	err = e.Commit(context.Background())
	require.NoError(t, err)
	t.Log("Flushed segment 2")

	// Insert 5 more
	for i := 10; i < 15; i++ {
		vec := []float32{float32(i), float32(i), float32(i), float32(i)}
		id, err := e.Insert(context.Background(), vec, nil, nil)
		require.NoError(t, err)
		t.Logf("Inserted ID=%d vec=%v", id, vec)
	}

	// Flush - this should trigger compaction (3 segments >= threshold of 2)
	err = e.Commit(context.Background())
	require.NoError(t, err)
	t.Log("Flushed segment 3")

	// Wait for compaction
	time.Sleep(1 * time.Second)

	stats := e.Stats()
	t.Logf("After compaction: segments=%d, rows=%d", stats.SegmentCount, stats.RowCount)

	// Verify each ID maps to correct vector
	for id := model.ID(1); id <= 15; id++ {
		rec, err := e.Get(context.Background(), id)
		require.NoError(t, err, "Get ID=%d", id)
		expectedVal := float32(id - 1) // ID=1 has vec=[0,0,0,0]
		expected := []float32{expectedVal, expectedVal, expectedVal, expectedVal}
		assert.Equal(t, expected, rec.Vector, "ID=%d vector mismatch", id)
	}

	// Search
	q := []float32{0, 0, 0, 0}
	res, err := e.Search(context.Background(), q, 5)
	require.NoError(t, err)
	require.NotEmpty(t, res, "Search should return results")
	t.Logf("Search found %d results:", len(res))
	for _, r := range res {
		t.Logf("  ID=%d score=%.4f", r.ID, r.Score)
	}
	assert.Equal(t, model.ID(1), res[0].ID, "ID=1 should be closest to query [0,0,0,0]")
}
