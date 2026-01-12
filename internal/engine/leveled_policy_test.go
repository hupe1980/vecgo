package engine

import (
	"testing"

	"github.com/hupe1980/vecgo/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestLeveledCompactionPolicy(t *testing.T) {
	p := NewLeveledCompactionPolicy()
	p.L0Threshold = 2 // Low threshold for testing

	// 1. No compaction needed
	task := p.Pick([]SegmentStats{
		{ID: 1, Level: 0, Size: 100},
	})
	assert.Nil(t, task)

	// 2. L0 Compaction Triggered
	segments := []SegmentStats{
		{ID: 1, Level: 0, Size: 100},
		{ID: 2, Level: 0, Size: 100},
	}
	task = p.Pick(segments)
	require.NotNil(t, task)
	assert.Equal(t, 1, task.TargetLevel)
	assert.Equal(t, []model.SegmentID{1, 2}, task.Segments)

	// 3. L1 Compaction Triggered (Size limit)
	// BaseSize is 100MB.
	segments = []SegmentStats{
		{ID: 3, Level: 1, Size: 150 * 1024 * 1024}, // 150MB > 100MB
	}
	task = p.Pick(segments)
	require.NotNil(t, task)
	assert.Equal(t, 2, task.TargetLevel)
	assert.Equal(t, []model.SegmentID{3}, task.Segments)
	// Logic in Leveled: Pick oldest victim in L1 to move to L2

	// 4. L2 Compaction Triggered
	// Target L2 = 100MB * 10 = 1000MB
	segments = []SegmentStats{
		{ID: 4, Level: 2, Size: 1200 * 1024 * 1024}, // 1.2GB
	}
	task = p.Pick(segments)
	require.NotNil(t, task)
	assert.Equal(t, 3, task.TargetLevel)
}
