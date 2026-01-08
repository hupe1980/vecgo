package metadata

import (
	"testing"
	"unique"

	"github.com/stretchr/testify/assert"
)

func TestFilterMatchesInterned(t *testing.T) {
	// Recreate some of the scenarios from TestFilterMatches but using InternedDocument

	doc := Intern(Document{
		"s": String("hello"),
		"i": Int(10),
	})

	tests := []struct {
		name     string
		filter   Filter
		expected bool
	}{
		{
			"Equal_String_Match",
			Filter{Key: "s", Operator: OpEqual, Value: String("hello")},
			true,
		},
		{
			"Equal_String_NoMatch",
			Filter{Key: "s", Operator: OpEqual, Value: String("world")},
			false,
		},
		{
			"Equal_Int_Match",
			Filter{Key: "i", Operator: OpEqual, Value: Int(10)},
			true,
		},
		{
			"GreaterThan_Int",
			Filter{Key: "i", Operator: OpGreaterThan, Value: Int(5)},
			true,
		},
		{
			"Key_Not_Exists",
			Filter{Key: "missing", Operator: OpEqual, Value: Int(10)},
			false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.expected, tt.filter.MatchesInterned(doc), "Filter.MatchesInterned")

			// Test FilterSet as well
			fs := NewFilterSet(tt.filter)
			assert.Equal(t, tt.expected, fs.MatchesInterned(doc), "FilterSet.MatchesInterned")
		})
	}
}

func TestFilterInternedKeyPerformance(t *testing.T) {
	// Verify that we are using unique handles correctly (coverage only, performance is tricky to assert)
	k := "test_key"
	h := unique.Make(k)

	doc := make(InternedDocument)
	doc[h] = Bool(true)

	f := Filter{Key: k, Operator: OpEqual, Value: Bool(true)}

	assert.True(t, f.MatchesInterned(doc))
}
