package metadata

import "testing"

func TestFilterMatches(t *testing.T) {
	tests := []struct {
		name     string
		filter   Filter
		metadata Document
		want     bool
	}{
		{
			name:     "OpEqual string match",
			filter:   Filter{Key: "category", Operator: OpEqual, Value: String("tech")},
			metadata: Document{"category": String("tech")},
			want:     true,
		},
		{
			name:     "OpEqual string no match",
			filter:   Filter{Key: "category", Operator: OpEqual, Value: String("tech")},
			metadata: Document{"category": String("sports")},
			want:     false,
		},
		{
			name:     "OpEqual int match",
			filter:   Filter{Key: "count", Operator: OpEqual, Value: Int(10)},
			metadata: Document{"count": Int(10)},
			want:     true,
		},
		{
			name:     "OpNotEqual",
			filter:   Filter{Key: "status", Operator: OpNotEqual, Value: String("active")},
			metadata: Document{"status": String("inactive")},
			want:     true,
		},
		{
			name:     "OpGreaterThan",
			filter:   Filter{Key: "score", Operator: OpGreaterThan, Value: Int(50)},
			metadata: Document{"score": Int(75)},
			want:     true,
		},
		{
			name:     "OpGreaterThan false",
			filter:   Filter{Key: "score", Operator: OpGreaterThan, Value: Int(50)},
			metadata: Document{"score": Int(25)},
			want:     false,
		},
		{
			name:     "OpGreaterEqual equal",
			filter:   Filter{Key: "age", Operator: OpGreaterEqual, Value: Int(18)},
			metadata: Document{"age": Int(18)},
			want:     true,
		},
		{
			name:     "OpGreaterEqual greater",
			filter:   Filter{Key: "age", Operator: OpGreaterEqual, Value: Int(18)},
			metadata: Document{"age": Int(25)},
			want:     true,
		},
		{
			name:     "OpLessThan",
			filter:   Filter{Key: "temperature", Operator: OpLessThan, Value: Int(100)},
			metadata: Document{"temperature": Int(75)},
			want:     true,
		},
		{
			name:     "OpLessEqual equal",
			filter:   Filter{Key: "limit", Operator: OpLessEqual, Value: Int(10)},
			metadata: Document{"limit": Int(10)},
			want:     true,
		},
		{
			name:     "OpIn string list",
			filter:   Filter{Key: "color", Operator: OpIn, Value: Array([]Value{String("red"), String("blue"), String("green")})},
			metadata: Document{"color": String("blue")},
			want:     true,
		},
		{
			name:     "OpIn mixed list",
			filter:   Filter{Key: "status", Operator: OpIn, Value: Array([]Value{String("active"), String("pending")})},
			metadata: Document{"status": String("active")},
			want:     true,
		},
		{
			name:     "OpIn not found",
			filter:   Filter{Key: "color", Operator: OpIn, Value: Array([]Value{String("red"), String("blue")})},
			metadata: Document{"color": String("yellow")},
			want:     false,
		},
		{
			name:     "OpContains substring",
			filter:   Filter{Key: "description", Operator: OpContains, Value: String("vector")},
			metadata: Document{"description": String("This is a vector database")},
			want:     true,
		},
		{
			name:     "OpContains not found",
			filter:   Filter{Key: "description", Operator: OpContains, Value: String("database")},
			metadata: Document{"description": String("This is a search engine")},
			want:     false,
		},
		{
			name:     "Key not exists",
			filter:   Filter{Key: "missing", Operator: OpEqual, Value: String("test")},
			metadata: Document{"other": String("value")},
			want:     false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.filter.Matches(tt.metadata)
			if got != tt.want {
				t.Errorf("Filter.Matches() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestFilterSetMatches(t *testing.T) {
	tests := []struct {
		name      string
		filterSet *FilterSet
		metadata  Document
		want      bool
	}{
		{
			name: "All filters match",
			filterSet: NewFilterSet(
				Filter{Key: "category", Operator: OpEqual, Value: String("tech")},
				Filter{Key: "score", Operator: OpGreaterThan, Value: Int(50)},
			),
			metadata: Document{"category": String("tech"), "score": Int(75)},
			want:     true,
		},
		{
			name: "One filter doesn't match",
			filterSet: NewFilterSet(
				Filter{Key: "category", Operator: OpEqual, Value: String("tech")},
				Filter{Key: "score", Operator: OpGreaterThan, Value: Int(50)},
			),
			metadata: Document{"category": String("tech"), "score": Int(25)},
			want:     false,
		},
		{
			name:      "Empty filter set",
			filterSet: NewFilterSet(),
			metadata:  Document{"anything": String("goes")},
			want:      true,
		},
		{
			name: "Complex filters all match",
			filterSet: NewFilterSet(
				Filter{Key: "status", Operator: OpIn, Value: Array([]Value{String("active"), String("pending")})},
				Filter{Key: "age", Operator: OpGreaterEqual, Value: Int(18)},
				Filter{Key: "country", Operator: OpEqual, Value: String("US")},
			),
			metadata: Document{"status": String("active"), "age": Int(25), "country": String("US")},
			want:     true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.filterSet.Matches(tt.metadata)
			if got != tt.want {
				t.Errorf("FilterSet.Matches() = %v, want %v", got, tt.want)
			}
		})
	}
}
