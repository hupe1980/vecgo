package engine

import (
	"context"
	"os"
	"testing"

	"github.com/hupe1980/vecgo/distance"
	"github.com/hupe1980/vecgo/metadata"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestIntrospection(t *testing.T) {
	dir := t.TempDir()
	e, err := OpenLocal(context.Background(), dir, WithDimension(2), WithMetric(distance.MetricL2))
	require.NoError(t, err)
	defer e.Close()

	// Initial stats
	stats := e.Stats()
	assert.Equal(t, 0, stats.SegmentCount)

	// Insert
	_, err = e.Insert(context.Background(), []float32{1.0, 0.0}, nil, nil)
	require.NoError(t, err)

	stats = e.Stats()
	assert.Greater(t, stats.MemoryUsageBytes, int64(0))

	// Flush
	err = e.Commit(context.Background())
	require.NoError(t, err)

	stats = e.Stats()
	assert.Equal(t, 1, stats.SegmentCount)
	assert.Greater(t, stats.DiskUsageBytes, int64(0))

	infos := e.SegmentInfo()
	assert.Len(t, infos, 1)
	assert.Equal(t, uint32(1), infos[0].RowCount)
}

func TestSegmentStatsPersistence(t *testing.T) {
	dir, err := os.MkdirTemp("", "vecgo-stats-persist-test")
	require.NoError(t, err)
	defer os.RemoveAll(dir)

	ctx := context.Background()

	// Phase 1: Create engine, insert data with metadata, flush, close
	{
		e, err := OpenLocal(ctx, dir, WithDimension(4), WithMetric(distance.MetricL2))
		require.NoError(t, err)

		// Insert 100 items with numeric and categorical metadata
		for i := 0; i < 100; i++ {
			vec := []float32{float32(i), float32(i), float32(i), float32(i)}
			md := metadata.Document{
				"price":    metadata.Int(int64(i * 10)),
				"category": metadata.String("cat" + string(rune('A'+i%5))),
			}
			_, err := e.Insert(ctx, vec, md, nil)
			require.NoError(t, err)
		}

		// Flush to create segment with stats
		err = e.Commit(ctx)
		require.NoError(t, err)

		// Verify stats exist before close
		infos := e.SegmentInfo()
		require.Len(t, infos, 1, "should have 1 segment")
		require.NotNil(t, infos[0].Stats, "segment should have stats before close")
		require.Contains(t, infos[0].Stats.Numeric, "price", "should have price stats")

		// Verify price min/max
		priceStats := infos[0].Stats.Numeric["price"]
		assert.Equal(t, float64(0), priceStats.Min, "price min should be 0")
		assert.Equal(t, float64(990), priceStats.Max, "price max should be 990")

		// Verify categorical stats
		require.Contains(t, infos[0].Stats.Categorical, "category", "should have category stats")
		catStats := infos[0].Stats.Categorical["category"]
		assert.Equal(t, uint32(5), catStats.DistinctCount, "should have 5 distinct categories")

		// Verify vector stats
		require.NotNil(t, infos[0].Stats.Vector, "should have vector stats")
		assert.Greater(t, infos[0].Stats.Vector.MaxNorm, infos[0].Stats.Vector.MinNorm)

		e.Close()
	}

	// Phase 2: Reopen and verify stats survived
	{
		e, err := OpenLocal(ctx, dir, WithDimension(4), WithMetric(distance.MetricL2))
		require.NoError(t, err)
		defer e.Close()

		infos := e.SegmentInfo()
		require.Len(t, infos, 1, "should have 1 segment after reopen")
		require.NotNil(t, infos[0].Stats, "segment should have stats after reopen")

		// Verify numeric stats survived
		require.Contains(t, infos[0].Stats.Numeric, "price", "should have price stats after reopen")
		priceStats := infos[0].Stats.Numeric["price"]
		assert.Equal(t, float64(0), priceStats.Min, "price min should be 0 after reopen")
		assert.Equal(t, float64(990), priceStats.Max, "price max should be 990 after reopen")

		// Verify categorical stats survived
		require.Contains(t, infos[0].Stats.Categorical, "category", "should have category stats after reopen")
		catStats := infos[0].Stats.Categorical["category"]
		assert.Equal(t, uint32(5), catStats.DistinctCount, "should have 5 distinct categories after reopen")

		// Verify vector stats survived
		require.NotNil(t, infos[0].Stats.Vector, "should have vector stats after reopen")
		assert.Greater(t, infos[0].Stats.Vector.MaxNorm, infos[0].Stats.Vector.MinNorm)

		// Verify HasFields
		require.True(t, infos[0].Stats.HasFields["price"], "should have price field")
		require.True(t, infos[0].Stats.HasFields["category"], "should have category field")

		t.Logf("Stats persisted correctly: price=[%.0f, %.0f], categories=%d, normRange=[%.2f, %.2f]",
			priceStats.Min, priceStats.Max,
			catStats.DistinctCount,
			infos[0].Stats.Vector.MinNorm, infos[0].Stats.Vector.MaxNorm)
	}
}
