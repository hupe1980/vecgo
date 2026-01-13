package engine

import (
	"context"
	"testing"
	"time"

	"github.com/hupe1980/vecgo/distance"
	"github.com/hupe1980/vecgo/internal/segment/flat"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCompactionConfig_Quantization(t *testing.T) {
	dir := t.TempDir()

	// Configure engine to use SQ8 for Flat segments
	cfg := CompactionConfig{
		FlatQuantizationType: flat.QuantizationSQ8,
	}

	e, err := Open(dir, 4, distance.MetricL2,
		WithCompactionConfig(cfg),
		WithCompactionThreshold(2), // Trigger compaction after 2 segments
	)
	require.NoError(t, err)
	defer e.Close()

	// Insert enough data to create 2 segments
	// Segment 1
	for i := 0; i < 100; i++ {
		_, err := e.Insert(context.Background(), []float32{0.1, 0.2, 0.3, 0.4}, nil, nil)
		require.NoError(t, err)
	}
	require.NoError(t, e.Commit(context.Background()))

	// Segment 2
	for i := 100; i < 200; i++ {
		_, err := e.Insert(context.Background(), []float32{0.5, 0.6, 0.7, 0.8}, nil, nil)
		require.NoError(t, err)
	}
	require.NoError(t, e.Commit(context.Background()))

	// Wait for compaction
	// Compaction runs in background. We can poll for segment count to drop to 1.
	assert.Eventually(t, func() bool {
		snap := e.current.Load()
		return len(snap.segments) == 1
	}, 5*time.Second, 100*time.Millisecond)

	// Verify the segment is quantized
	e.mu.RLock()
	snap := e.current.Load()
	require.Len(t, snap.segments, 1)
	var seg *RefCountedSegment
	for _, s := range snap.segments {
		seg = s
		break
	}
	e.mu.RUnlock()

	// Cast underlying segment to *flat.Segment
	flatSeg, ok := seg.Segment.(*flat.Segment)
	require.True(t, ok, "expected flat segment")

	assert.Equal(t, flat.QuantizationSQ8, flatSeg.QuantizationType())
}
