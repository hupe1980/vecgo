package metadata

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestUnifiedIndex_SetGet(t *testing.T) {
	ui := NewUnifiedIndex()

	doc := Document{
		"category": String("technology"),
		"year":     Int(2024),
		"active":   Bool(true),
	}

	// Set metadata
	ui.Set(1, doc)

	// Get metadata
	retrieved, ok := ui.Get(1)
	require.True(t, ok)
	assert.Equal(t, doc, retrieved)

	// Non-existent ID
	_, ok = ui.Get(999)
	assert.False(t, ok)
}

func TestUnifiedIndex_Update(t *testing.T) {
	ui := NewUnifiedIndex()

	// Initial document
	doc1 := Document{
		"category": String("technology"),
		"year":     Int(2023),
	}
	ui.Set(1, doc1)

	// Update document
	doc2 := Document{
		"category": String("science"),
		"year":     Int(2024),
	}
	ui.Set(1, doc2)

	// Verify update
	retrieved, ok := ui.Get(1)
	require.True(t, ok)
	assert.Equal(t, String("science"), retrieved["category"])
	assert.Equal(t, Int(2024), retrieved["year"])
}

func TestUnifiedIndex_Delete(t *testing.T) {
	ui := NewUnifiedIndex()

	doc := Document{
		"category": String("technology"),
	}
	ui.Set(1, doc)

	// Verify exists
	_, ok := ui.Get(1)
	require.True(t, ok)

	// Delete
	ui.Delete(1)

	// Verify deleted
	_, ok = ui.Get(1)
	assert.False(t, ok)
}

func TestUnifiedIndex_CompileFilter_Equal(t *testing.T) {
	ui := NewUnifiedIndex()

	// Add test documents
	ui.Set(1, Document{"category": String("tech")})
	ui.Set(2, Document{"category": String("science")})
	ui.Set(3, Document{"category": String("tech")})

	// Compile filter: category == "tech"
	fs := &FilterSet{
		Filters: []Filter{
			{Key: "category", Operator: OpEqual, Value: String("tech")},
		},
	}

	bitmap := ui.CompileFilter(fs)
	require.NotNil(t, bitmap)
	assert.Equal(t, uint64(2), bitmap.GetCardinality())
	assert.True(t, bitmap.Contains(1))
	assert.False(t, bitmap.Contains(2))
	assert.True(t, bitmap.Contains(3))
}

func TestUnifiedIndex_CompileFilter_In(t *testing.T) {
	ui := NewUnifiedIndex()

	ui.Set(1, Document{"status": String("active")})
	ui.Set(2, Document{"status": String("pending")})
	ui.Set(3, Document{"status": String("inactive")})
	ui.Set(4, Document{"status": String("active")})

	// Compile filter: status IN ["active", "pending"]
	fs := &FilterSet{
		Filters: []Filter{
			{Key: "status", Operator: OpIn, Value: Array([]Value{String("active"), String("pending")})},
		},
	}

	bitmap := ui.CompileFilter(fs)
	require.NotNil(t, bitmap)
	assert.Equal(t, uint64(3), bitmap.GetCardinality())
	assert.True(t, bitmap.Contains(1))
	assert.True(t, bitmap.Contains(2))
	assert.False(t, bitmap.Contains(3))
	assert.True(t, bitmap.Contains(4))
}

func TestUnifiedIndex_CompileFilter_MultipleConditions(t *testing.T) {
	ui := NewUnifiedIndex()

	ui.Set(1, Document{"category": String("tech"), "year": Int(2024)})
	ui.Set(2, Document{"category": String("tech"), "year": Int(2023)})
	ui.Set(3, Document{"category": String("science"), "year": Int(2024)})

	// Compile filter: category == "tech" AND year == 2024
	fs := &FilterSet{
		Filters: []Filter{
			{Key: "category", Operator: OpEqual, Value: String("tech")},
			{Key: "year", Operator: OpEqual, Value: Int(2024)},
		},
	}

	bitmap := ui.CompileFilter(fs)
	require.NotNil(t, bitmap)
	assert.Equal(t, uint64(1), bitmap.GetCardinality())
	assert.True(t, bitmap.Contains(1))
	assert.False(t, bitmap.Contains(2))
	assert.False(t, bitmap.Contains(3))
}

func TestUnifiedIndex_CompileFilter_NoMatches(t *testing.T) {
	ui := NewUnifiedIndex()

	ui.Set(1, Document{"category": String("tech")})

	// Filter with no matches
	fs := &FilterSet{
		Filters: []Filter{
			{Key: "category", Operator: OpEqual, Value: String("nonexistent")},
		},
	}

	bitmap := ui.CompileFilter(fs)
	require.NotNil(t, bitmap)
	assert.True(t, bitmap.IsEmpty())
}

func TestUnifiedIndex_ScanFilter(t *testing.T) {
	ui := NewUnifiedIndex()

	ui.Set(1, Document{"score": Int(100)})
	ui.Set(2, Document{"score": Int(50)})
	ui.Set(3, Document{"score": Int(150)})

	// Scan filter: score > 75 (not supported by CompileFilter)
	fs := &FilterSet{
		Filters: []Filter{
			{Key: "score", Operator: OpGreaterThan, Value: Int(75)},
		},
	}

	// CompileFilter should return nil (operator not supported)
	bitmap := ui.CompileFilter(fs)
	assert.Nil(t, bitmap)

	// ScanFilter should work
	ids := ui.ScanFilter(fs)
	assert.Len(t, ids, 2)
	assert.Contains(t, ids, uint32(1))
	assert.Contains(t, ids, uint32(3))
}

func TestUnifiedIndex_CreateFilterFunc(t *testing.T) {
	ui := NewUnifiedIndex()

	ui.Set(1, Document{"category": String("tech")})
	ui.Set(2, Document{"category": String("science")})
	ui.Set(3, Document{"category": String("tech")})

	// Create filter function
	fs := &FilterSet{
		Filters: []Filter{
			{Key: "category", Operator: OpEqual, Value: String("tech")},
		},
	}

	filterFn := ui.CreateFilterFunc(fs)
	require.NotNil(t, filterFn)

	assert.True(t, filterFn(1))
	assert.False(t, filterFn(2))
	assert.True(t, filterFn(3))
	assert.False(t, filterFn(999)) // Non-existent
}

func TestUnifiedIndex_InvertedIndexUpdate(t *testing.T) {
	ui := NewUnifiedIndex()

	// Initial document
	ui.Set(1, Document{"category": String("tech")})

	// Verify inverted index has entry
	fs1 := &FilterSet{
		Filters: []Filter{
			{Key: "category", Operator: OpEqual, Value: String("tech")},
		},
	}
	bitmap1 := ui.CompileFilter(fs1)
	require.NotNil(t, bitmap1)
	assert.True(t, bitmap1.Contains(1))

	// Update to different category
	ui.Set(1, Document{"category": String("science")})

	// Old inverted index entry should be cleaned up
	bitmap2 := ui.CompileFilter(fs1)
	require.NotNil(t, bitmap2)
	assert.False(t, bitmap2.Contains(1)) // No longer in "tech"

	// New category should be indexed
	fs2 := &FilterSet{
		Filters: []Filter{
			{Key: "category", Operator: OpEqual, Value: String("science")},
		},
	}
	bitmap3 := ui.CompileFilter(fs2)
	require.NotNil(t, bitmap3)
	assert.True(t, bitmap3.Contains(1))
}

func TestUnifiedIndex_InvertedIndexDelete(t *testing.T) {
	ui := NewUnifiedIndex()

	ui.Set(1, Document{"category": String("tech")})
	ui.Set(2, Document{"category": String("tech")})

	// Both should match
	fs := &FilterSet{
		Filters: []Filter{
			{Key: "category", Operator: OpEqual, Value: String("tech")},
		},
	}
	bitmap := ui.CompileFilter(fs)
	require.NotNil(t, bitmap)
	assert.Equal(t, uint64(2), bitmap.GetCardinality())

	// Delete one
	ui.Delete(1)

	// Only one should remain
	bitmap = ui.CompileFilter(fs)
	require.NotNil(t, bitmap)
	assert.Equal(t, uint64(1), bitmap.GetCardinality())
	assert.False(t, bitmap.Contains(1))
	assert.True(t, bitmap.Contains(2))

	// Delete last one
	ui.Delete(2)

	// Should be empty (and cleaned up)
	bitmap = ui.CompileFilter(fs)
	require.NotNil(t, bitmap)
	assert.True(t, bitmap.IsEmpty())
}

func TestUnifiedIndex_Stats(t *testing.T) {
	ui := NewUnifiedIndex()

	// Initially empty
	stats := ui.GetStats()
	assert.Equal(t, 0, stats.DocumentCount)
	assert.Equal(t, 0, stats.FieldCount)
	assert.Equal(t, 0, stats.BitmapCount)

	// Add documents
	ui.Set(1, Document{"category": String("tech"), "year": Int(2024)})
	ui.Set(2, Document{"category": String("science"), "year": Int(2024)})
	ui.Set(3, Document{"category": String("tech"), "year": Int(2023)})

	stats = ui.GetStats()
	assert.Equal(t, 3, stats.DocumentCount)
	assert.Equal(t, 2, stats.FieldCount)               // category, year
	assert.Equal(t, 4, stats.BitmapCount)              // tech, science, 2024, 2023
	assert.Equal(t, uint64(6), stats.TotalCardinality) // Sum of all bitmap sizes
	assert.Greater(t, stats.MemoryBytes, uint64(0))
}

func TestUnifiedIndex_Len(t *testing.T) {
	ui := NewUnifiedIndex()

	assert.Equal(t, 0, ui.Len())

	ui.Set(1, Document{"key": String("value")})
	assert.Equal(t, 1, ui.Len())

	ui.Set(2, Document{"key": String("value")})
	assert.Equal(t, 2, ui.Len())

	ui.Delete(1)
	assert.Equal(t, 1, ui.Len())
}

// Benchmark CompileFilter vs ScanFilter
func BenchmarkUnifiedIndex_CompileFilter(b *testing.B) {
	ui := NewUnifiedIndex()

	// Add 10k documents
	for i := uint32(0); i < 10000; i++ {
		category := "tech"
		if i%3 == 0 {
			category = "science"
		}
		ui.Set(i, Document{"category": String(category)})
	}

	fs := &FilterSet{
		Filters: []Filter{
			{Key: "category", Operator: OpEqual, Value: String("tech")},
		},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bitmap := ui.CompileFilter(fs)
		_ = bitmap.GetCardinality()
	}
}

func BenchmarkUnifiedIndex_ScanFilter(b *testing.B) {
	ui := NewUnifiedIndex()

	// Add 10k documents
	for i := uint32(0); i < 10000; i++ {
		score := int64(i % 100)
		ui.Set(i, Document{"score": Int(score)})
	}

	fs := &FilterSet{
		Filters: []Filter{
			{Key: "score", Operator: OpGreaterThan, Value: Int(50)},
		},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ids := ui.ScanFilter(fs)
		_ = len(ids)
	}
}
