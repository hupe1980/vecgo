package engine

import (
	"testing"

	"github.com/hupe1980/vecgo/distance"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestIntrospection(t *testing.T) {
	dir := t.TempDir()
	e, err := Open(dir, 2, distance.MetricL2)
	require.NoError(t, err)
	defer e.Close()

	// Initial stats
	stats := e.Stats()
	assert.Equal(t, 0, stats.SegmentCount)

	// Insert
	_, err = e.Insert([]float32{1.0, 0.0}, nil, nil)
	require.NoError(t, err)

	stats = e.Stats()
	assert.Greater(t, stats.MemoryUsageBytes, int64(0))

	// Flush
	err = e.Flush()
	require.NoError(t, err)

	stats = e.Stats()
	assert.Equal(t, 1, stats.SegmentCount)
	assert.Greater(t, stats.DiskUsageBytes, int64(0))

	infos := e.SegmentInfo()
	assert.Len(t, infos, 1)
	assert.Equal(t, uint32(1), infos[0].RowCount)
}
