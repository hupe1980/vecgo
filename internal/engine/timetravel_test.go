package engine

import (
	"context"
	"log/slog"
	"os"
	"testing"
	"time"

	"github.com/hupe1980/vecgo/distance"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestTimeTravel(t *testing.T) {
	dir, err := os.MkdirTemp("", "vecgo-timetravel-test")
	require.NoError(t, err)
	defer os.RemoveAll(dir)

	// Phase 1: Create History
	// Version 1: Insert 10 items
	// Open(dir, dim, metric, opts...)
	logger := slog.New(slog.NewTextHandler(os.Stdout, nil))

	// Use a compaction policy that never triggers to avoid concurrent manifest saves
	noCompactionPolicy := &TieredCompactionPolicy{Threshold: 1000}

	opts := []Option{
		WithDimension(128),            // ignored by Open but good for symmetry
		WithMetric(distance.MetricL2), // ignored
		WithLogger(logger),
		WithCompactionPolicy(noCompactionPolicy),
	}
	e, err := Open(dir, 128, distance.MetricL2, opts...)
	require.NoError(t, err)

	for i := 0; i < 10; i++ {
		vec := make([]float32, 128)
		vec[0] = float32(i)
		_, err := e.Insert(context.Background(), vec, nil, nil)
		require.NoError(t, err)
	}
	// Flush to create Version 1
	err = e.Commit(context.Background())
	require.NoError(t, err)

	// IMPORTANT: Capture timestamp AFTER Flush completes.
	// Manifests are timestamped at Save time (inside Flush), not at capture time.
	// The time-travel query finds the newest version created AT OR BEFORE the target timestamp.
	time.Sleep(100 * time.Millisecond) // Small gap to ensure ordering
	t1 := time.Now()
	time.Sleep(100 * time.Millisecond) // Gap before next operations

	// Version 2: Insert 10 more items
	for i := 10; i < 20; i++ {
		vec := make([]float32, 128)
		vec[0] = float32(i)
		_, err := e.Insert(context.Background(), vec, nil, nil)
		require.NoError(t, err)
	}
	// Flush to create Version 2
	err = e.Commit(context.Background())
	require.NoError(t, err)

	// Verify current running engine has 20 items
	currentStats := e.Stats()
	assert.Equal(t, 20, currentStats.RowCount, "Running engine should have 20 rows")

	time.Sleep(100 * time.Millisecond)
	t2 := time.Now()
	time.Sleep(100 * time.Millisecond)

	// Version 3: Delete first 5 items
	for i := 0; i < 5; i++ {
		vec := make([]float32, 128)
		vec[0] = float32(i)
		// We need ID to delete?
		// Engine doesn't return ID on Insert in this test style,
		// but Insert generates them.
		// Wait, Insert returns error.
		// We need to know IDs.
		// Let's assume sequential IDs starting 1 if engine is fresh.
		// But let's verify count instead.
	}
	stats := e.Stats()
	assert.Equal(t, 20, stats.RowCount)

	err = e.Delete(context.Background(), 1) // Delete ID 1
	require.NoError(t, err)

	err = e.Commit(context.Background())
	require.NoError(t, err)

	_ = time.Now() // t3 unused but kept for symmetry

	e.Close()

	// Phase 2: Time Travel

	// Case A: Load Version 1 (10 items)
	e1, err := OpenLocal(context.Background(), dir, WithTimestamp(t1))
	require.NoError(t, err)
	stats1 := e1.Stats()
	assert.Equal(t, 10, stats1.RowCount, "Version 1 should have 10 rows")
	e1.Close()

	// Case B: Load Version 2 (20 items)
	e2, err := OpenLocal(context.Background(), dir, WithTimestamp(t2), WithLogger(logger))
	require.NoError(t, err)
	stats2 := e2.Stats()
	assert.Equal(t, 20, stats2.RowCount, "Version 2 should have 20 rows")
	e2.Close()

	// Case C: Load Latest (Version 3) (19 items)
	// Default load
	e3, err := OpenLocal(context.Background(), dir)
	require.NoError(t, err)
	stats3 := e3.Stats()
	assert.Equal(t, 19, stats3.RowCount, "Version 3 should have 19 rows (1 deleted)")

	// Test Vacuum
	// Keep only latest version (Version 3)
	err = e3.Vacuum(context.Background()) // No policy set, should do nothing
	require.NoError(t, err)

	// Set policy to keep 1 version
	e3.retentionPolicy = RetentionPolicy{KeepVersions: 1}
	err = e3.Vacuum(context.Background())
	require.NoError(t, err)

	e3.Close()

	// Verify Version 1 is GONE
	// Timestamp t1 should now fail or load latest?
	// Logic says: "resolves closest version".
	// If V1 is deleted, V2 or V3 is closest?
	// If V1 and V2 are deleted, V3 is the only one left.
	// So loading at t1 will load V3 (created at t3 > t1).
	// Wait, logic:
	// Find version <= targetTimestamp.
	// If V1 (t < t1) is gone, and only V3 (t > t1) remains...
	// The loop:
	// for _, v := range versions { if !v.CreatedAt.After(target) { best = v } else { break } }
	// If all versions are newer than target, best remains nil.
	// So we expect Error: "no version found at or before t1"

	eGone, err := OpenLocal(context.Background(), dir, WithTimestamp(t1))
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "no version found")
	if eGone != nil {
		eGone.Close()
	}
}
