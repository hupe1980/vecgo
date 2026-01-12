package engine

import (
	"testing"

	"github.com/hupe1980/vecgo/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestBoundedSizeTieredPolicy(t *testing.T) {
	policy := &BoundedSizeTieredPolicy{Threshold: 4}

	tests := []struct {
		name     string
		segments []SegmentStats
		want     []model.SegmentID
	}{
		{
			name:     "Empty",
			segments: []SegmentStats{},
			want:     nil,
		},
		{
			name: "Below Threshold",
			segments: []SegmentStats{
				{ID: 1, Size: 100},
				{ID: 2, Size: 100},
				{ID: 3, Size: 100},
			},
			want: nil,
		},
		{
			name: "Above Threshold - Small Bucket",
			segments: []SegmentStats{
				{ID: 1, Size: 100},
				{ID: 2, Size: 100},
				{ID: 3, Size: 100},
				{ID: 4, Size: 100},
			},
			want: []model.SegmentID{1, 2, 3, 4},
		},
		{
			name: "Mixed Buckets - Only Small Eligible",
			segments: []SegmentStats{
				{ID: 1, Size: 100},
				{ID: 2, Size: 100},
				{ID: 3, Size: 100},
				{ID: 4, Size: 100},
				{ID: 5, Size: 20 * 1024 * 1024}, // 20MB (Bucket 1)
			},
			want: []model.SegmentID{1, 2, 3, 4},
		},
		{
			name: "Mixed Buckets - Only Medium Eligible",
			segments: []SegmentStats{
				{ID: 1, Size: 100},
				{ID: 2, Size: 20 * 1024 * 1024}, // 20MB
				{ID: 3, Size: 20 * 1024 * 1024},
				{ID: 4, Size: 20 * 1024 * 1024},
				{ID: 5, Size: 20 * 1024 * 1024},
			},
			want: []model.SegmentID{2, 3, 4, 5},
		},
		{
			name: "Respects 2GB Limit",
			segments: []SegmentStats{
				{ID: 1, Size: 1 * 1024 * 1024 * 1024}, // 1GB
				{ID: 2, Size: 1 * 1024 * 1024 * 1024}, // 1GB
				{ID: 3, Size: 1 * 1024 * 1024 * 1024}, // 1GB
				{ID: 4, Size: 1 * 1024 * 1024 * 1024}, // 1GB
			},
			// Should pick first 2 (2GB total)
			want: []model.SegmentID{1, 2},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotTask := policy.Pick(tt.segments)
			if tt.want == nil {
				assert.Nil(t, gotTask)
			} else {
				require.NotNil(t, gotTask)
				assert.Equal(t, tt.want, gotTask.Segments)
				// We won't verify TargetLevel in this generic loop unless we update struct
			}
		})
	}
}
