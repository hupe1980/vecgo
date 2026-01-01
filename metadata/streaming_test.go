package metadata

import (
	"testing"

	"github.com/hupe1980/vecgo/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestUnifiedIndex_CreateStreamingFilter(t *testing.T) {
	ui := NewUnifiedIndex()

	// Add test documents
	ui.Set(1, Document{"category": String("tech"), "year": Int(2023)})
	ui.Set(2, Document{"category": String("science"), "year": Int(2024)})
	ui.Set(3, Document{"category": String("tech"), "year": Int(2024)})
	ui.Set(4, Document{"category": String("art"), "year": Int(2023)})

	tests := []struct {
		name     string
		filters  *FilterSet
		expected []uint64
	}{
		{
			name: "Equal",
			filters: &FilterSet{
				Filters: []Filter{
					{Key: "category", Operator: OpEqual, Value: String("tech")},
				},
			},
			expected: []uint64{1, 3},
		},
		{
			name: "In",
			filters: &FilterSet{
				Filters: []Filter{
					{Key: "category", Operator: OpIn, Value: Array([]Value{String("tech"), String("art")})},
				},
			},
			expected: []uint64{1, 3, 4},
		},
		{
			name: "AND (Equal + Equal)",
			filters: &FilterSet{
				Filters: []Filter{
					{Key: "category", Operator: OpEqual, Value: String("tech")},
					{Key: "year", Operator: OpEqual, Value: Int(2024)},
				},
			},
			expected: []uint64{3},
		},
		{
			name: "Fallback (GreaterThan)",
			filters: &FilterSet{
				Filters: []Filter{
					{Key: "year", Operator: OpGreaterThan, Value: Int(2023)},
				},
			},
			expected: []uint64{2, 3},
		},
		{
			name: "Mixed (Equal + GreaterThan)",
			filters: &FilterSet{
				Filters: []Filter{
					{Key: "category", Operator: OpEqual, Value: String("tech")},
					{Key: "year", Operator: OpGreaterThan, Value: Int(2023)},
				},
			},
			expected: []uint64{3},
		},
		{
			name: "No Match",
			filters: &FilterSet{
				Filters: []Filter{
					{Key: "category", Operator: OpEqual, Value: String("space")},
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
			for id := core.LocalID(1); id <= 4; id++ {
				if filter(id) {
					matches = append(matches, uint64(id))
				}
			}

			assert.ElementsMatch(t, tt.expected, matches)
		})
	}
}
