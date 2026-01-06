package imetadata

import (
	"testing"

	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestUnifiedIndex_CreateStreamingFilter(t *testing.T) {
	ui := NewUnifiedIndex()

	// Add test documents
	ui.Set(1, metadata.Document{"category": metadata.String("tech"), "year": metadata.Int(2023)})
	ui.Set(2, metadata.Document{"category": metadata.String("science"), "year": metadata.Int(2024)})
	ui.Set(3, metadata.Document{"category": metadata.String("tech"), "year": metadata.Int(2024)})
	ui.Set(4, metadata.Document{"category": metadata.String("art"), "year": metadata.Int(2023)})

	tests := []struct {
		name     string
		filters  *metadata.FilterSet
		expected []uint64
	}{
		{
			name: "Equal",
			filters: &metadata.FilterSet{
				Filters: []metadata.Filter{
					{Key: "category", Operator: metadata.OpEqual, Value: metadata.String("tech")},
				},
			},
			expected: []uint64{1, 3},
		},
		{
			name: "In",
			filters: &metadata.FilterSet{
				Filters: []metadata.Filter{
					{Key: "category", Operator: metadata.OpIn, Value: metadata.Array([]metadata.Value{metadata.String("tech"), metadata.String("art")})},
				},
			},
			expected: []uint64{1, 3, 4},
		},
		{
			name: "AND (Equal + Equal)",
			filters: &metadata.FilterSet{
				Filters: []metadata.Filter{
					{Key: "category", Operator: metadata.OpEqual, Value: metadata.String("tech")},
					{Key: "year", Operator: metadata.OpEqual, Value: metadata.Int(2024)},
				},
			},
			expected: []uint64{3},
		},
		{
			name: "Fallback (GreaterThan)",
			filters: &metadata.FilterSet{
				Filters: []metadata.Filter{
					{Key: "year", Operator: metadata.OpGreaterThan, Value: metadata.Int(2023)},
				},
			},
			expected: []uint64{2, 3},
		},
		{
			name: "Mixed (Equal + GreaterThan)",
			filters: &metadata.FilterSet{
				Filters: []metadata.Filter{
					{Key: "category", Operator: metadata.OpEqual, Value: metadata.String("tech")},
					{Key: "year", Operator: metadata.OpGreaterThan, Value: metadata.Int(2023)},
				},
			},
			expected: []uint64{3},
		},
		{
			name: "No Match",
			filters: &metadata.FilterSet{
				Filters: []metadata.Filter{
					{Key: "category", Operator: metadata.OpEqual, Value: metadata.String("space")},
				},
			},
			expected: []uint64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ui.RLock()
			defer ui.RUnlock()

			filter := ui.CreateStreamingFilter(tt.filters)
			require.NotNil(t, filter)

			var matches []uint64
			for id := model.RowID(1); id <= 4; id++ {
				if filter(id) {
					matches = append(matches, uint64(id))
				}
			}

			assert.ElementsMatch(t, tt.expected, matches)
		})
	}
}
