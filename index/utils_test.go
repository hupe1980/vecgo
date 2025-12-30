package index

import (
	"reflect"
	"testing"
)

func TestMergeNSearchResults(t *testing.T) {
	tests := []struct {
		name  string
		k     int
		lists [][]SearchResult
		want  []SearchResult
	}{
		{
			name: "empty",
			k:    5,
			lists: [][]SearchResult{
				{},
				{},
			},
			want: nil,
		},
		{
			name: "single list",
			k:    5,
			lists: [][]SearchResult{
				{{ID: 1, Distance: 0.1}, {ID: 2, Distance: 0.2}},
			},
			want: []SearchResult{{ID: 1, Distance: 0.1}, {ID: 2, Distance: 0.2}},
		},
		{
			name: "two lists",
			k:    3,
			lists: [][]SearchResult{
				{{ID: 1, Distance: 0.1}, {ID: 3, Distance: 0.3}},
				{{ID: 2, Distance: 0.2}, {ID: 4, Distance: 0.4}},
			},
			want: []SearchResult{
				{ID: 1, Distance: 0.1},
				{ID: 2, Distance: 0.2},
				{ID: 3, Distance: 0.3},
			},
		},
		{
			name: "three lists",
			k:    4,
			lists: [][]SearchResult{
				{{ID: 1, Distance: 0.1}, {ID: 4, Distance: 0.4}},
				{{ID: 2, Distance: 0.2}, {ID: 5, Distance: 0.5}},
				{{ID: 3, Distance: 0.3}, {ID: 6, Distance: 0.6}},
			},
			want: []SearchResult{
				{ID: 1, Distance: 0.1},
				{ID: 2, Distance: 0.2},
				{ID: 3, Distance: 0.3},
				{ID: 4, Distance: 0.4},
			},
		},
		{
			name: "lists with different lengths",
			k:    5,
			lists: [][]SearchResult{
				{{ID: 1, Distance: 0.1}},
				{{ID: 2, Distance: 0.2}, {ID: 3, Distance: 0.3}},
				{},
				{{ID: 4, Distance: 0.4}},
			},
			want: []SearchResult{
				{ID: 1, Distance: 0.1},
				{ID: 2, Distance: 0.2},
				{ID: 3, Distance: 0.3},
				{ID: 4, Distance: 0.4},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := MergeNSearchResults(tt.k, tt.lists...)
			if !reflect.DeepEqual(got, tt.want) {
				// Handle nil vs empty slice mismatch
				if len(got) == 0 && len(tt.want) == 0 {
					return
				}
				t.Errorf("MergeNSearchResults() = %v, want %v", got, tt.want)
			}
		})
	}
}
