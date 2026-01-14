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
			// Test FilterSet
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

	fs := NewFilterSet(Filter{Key: k, Operator: OpEqual, Value: Bool(true)})

	assert.True(t, fs.MatchesInterned(doc))
}

func TestFilterSetInternedKeyCaching(t *testing.T) {
	// Verify that FilterSet caches interned keys after first call
	fs := NewFilterSet(
		Filter{Key: "category", Operator: OpEqual, Value: String("tech")},
		Filter{Key: "year", Operator: OpGreaterEqual, Value: Int(2023)},
	)

	// Before first MatchesInterned, cache should be nil
	assert.Nil(t, fs.internedKeys)

	doc := Intern(Document{
		"category": String("tech"),
		"year":     Int(2024),
	})

	// First call should initialize cache
	result := fs.MatchesInterned(doc)
	assert.True(t, result)
	assert.NotNil(t, fs.internedKeys)
	assert.Len(t, fs.internedKeys, 2)

	// Cache should contain correct handles
	assert.Equal(t, unique.Make("category"), fs.internedKeys[0])
	assert.Equal(t, unique.Make("year"), fs.internedKeys[1])

	// Second call should reuse cache (no way to directly verify, but coverage)
	doc2 := Intern(Document{
		"category": String("science"),
		"year":     Int(2024),
	})
	result2 := fs.MatchesInterned(doc2)
	assert.False(t, result2) // category mismatch

	// Cache should still be the same
	assert.Len(t, fs.internedKeys, 2)
}

func TestEmptyFilterSetMatchesInterned(t *testing.T) {
	// Empty FilterSet should match everything without initializing cache
	fs := NewFilterSet()

	doc := Intern(Document{"any": String("value")})

	result := fs.MatchesInterned(doc)
	assert.True(t, result)
	assert.Nil(t, fs.internedKeys) // Cache should stay nil for empty FilterSet
}
