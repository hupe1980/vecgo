package integration_test

import (
	"context"
	"testing"
	"time"

	"github.com/hupe1980/vecgo"
	"github.com/hupe1980/vecgo/engine"
	"github.com/hupe1980/vecgo/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMixedSegments(t *testing.T) {
	dir := t.TempDir()
	dim := 4

	// Configure to force compaction quickly
	// DiskANN threshold low to force DiskANN creation
	compactionCfg := engine.CompactionConfig{
		DiskANNThreshold: 100, // Very low threshold
	}

	e, err := vecgo.Open(dir, dim, vecgo.MetricL2,
		engine.WithCompactionConfig(compactionCfg),
		engine.WithCompactionThreshold(2),                                 // Compact every 2 segments
		engine.WithFlushConfig(engine.FlushConfig{MaxMemTableSize: 1024}), // Flush often
	)
	require.NoError(t, err)
	defer e.Close()

	// Insert enough data to trigger flush and compaction
	// 200 vectors -> multiple flushes -> compaction -> DiskANN
	n := 200
	for i := 0; i < n; i++ {
		pk := model.PKUint64(uint64(i))
		vec := []float32{float32(i), float32(i), float32(i), float32(i)}
		err := e.Insert(pk, vec, nil, nil)
		assert.NoError(t, err)

		// Small sleep to allow background tasks to run
		if i%50 == 0 {
			time.Sleep(100 * time.Millisecond)
		}
	}

	// Wait for compaction
	time.Sleep(1 * time.Second)

	// Search
	q := []float32{0, 0, 0, 0}
	res, err := e.Search(context.Background(), q, 10)
	require.NoError(t, err)
	assert.NotEmpty(t, res)
	// assert.Equal(t, model.PrimaryKey(0), res[0].PK)

	// Verify persistence
	e.Close()

	e2, err := vecgo.Open(dir, dim, vecgo.MetricL2)
	require.NoError(t, err)
	defer e2.Close()

	res2, err := e2.Search(context.Background(), q, 10)
	require.NoError(t, err)
	assert.NotEmpty(t, res2)
	// assert.Equal(t, model.PrimaryKey(0), res2[0].PK)
}
