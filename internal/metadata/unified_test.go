package imetadata

import (
	"context"
	"testing"

	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestUnifiedIndex_SetGet(t *testing.T) {
	ui := NewUnifiedIndex()

	doc := metadata.Document{
		"category": metadata.String("technology"),
		"year":     metadata.Int(2024),
		"active":   metadata.Bool(true),
	}

	// Set metadata
	ui.Set(1, doc)

	// Get metadata
	retrieved, ok := ui.Get(context.Background(), 1)
	require.True(t, ok)
	assert.Equal(t, doc, retrieved)

	// Non-existent ID
	_, ok = ui.Get(context.Background(), 999)
	assert.False(t, ok)
}

func TestUnifiedIndex_Update(t *testing.T) {
	ui := NewUnifiedIndex()

	// Initial document
	doc1 := metadata.Document{
		"category": metadata.String("technology"),
		"year":     metadata.Int(2023),
	}
	ui.Set(1, doc1)

	// Update document
	doc2 := metadata.Document{
		"category": metadata.String("science"),
		"year":     metadata.Int(2024),
	}
	ui.Set(1, doc2)

	// Verify update
	retrieved, ok := ui.Get(context.Background(), 1)
	require.True(t, ok)
	assert.Equal(t, metadata.String("science"), retrieved["category"])
	assert.Equal(t, metadata.Int(2024), retrieved["year"])
}

func TestUnifiedIndex_Delete(t *testing.T) {
	ui := NewUnifiedIndex()

	doc := metadata.Document{
		"category": metadata.String("technology"),
	}
	ui.Set(1, doc)

	// Verify exists
	_, ok := ui.Get(context.Background(), 1)
	require.True(t, ok)

	// Delete
	ui.Delete(1)

	// Verify deleted
	_, ok = ui.Get(context.Background(), 1)
	assert.False(t, ok)
}

func TestUnifiedIndex_CompileFilter_Equal(t *testing.T) {
	ui := NewUnifiedIndex()

	// Add test documents
	ui.Set(1, metadata.Document{"category": metadata.String("tech")})
	ui.Set(2, metadata.Document{"category": metadata.String("science")})
	ui.Set(3, metadata.Document{"category": metadata.String("tech")})

	// Compile filter: category == "tech"
	fs := &metadata.FilterSet{
		Filters: []metadata.Filter{
			{Key: "category", Operator: metadata.OpEqual, Value: metadata.String("tech")},
		},
	}

	bitmap := ui.CompileFilter(fs)
	require.NotNil(t, bitmap)
	assert.Equal(t, uint64(2), bitmap.Cardinality())
	assert.True(t, bitmap.Contains(1))
	assert.False(t, bitmap.Contains(2))
	assert.True(t, bitmap.Contains(3))
}

func TestUnifiedIndex_CompileFilter_In(t *testing.T) {
	ui := NewUnifiedIndex()

	ui.Set(1, metadata.Document{"status": metadata.String("active")})
	ui.Set(2, metadata.Document{"status": metadata.String("pending")})
	ui.Set(3, metadata.Document{"status": metadata.String("inactive")})
	ui.Set(4, metadata.Document{"status": metadata.String("active")})

	// Compile filter: status IN ["active", "pending"]
	fs := &metadata.FilterSet{
		Filters: []metadata.Filter{
			{Key: "status", Operator: metadata.OpIn, Value: metadata.Array([]metadata.Value{metadata.String("active"), metadata.String("pending")})},
		},
	}

	bitmap := ui.CompileFilter(fs)
	require.NotNil(t, bitmap)
	assert.Equal(t, uint64(3), bitmap.Cardinality())
	assert.True(t, bitmap.Contains(1))
	assert.True(t, bitmap.Contains(2))
	assert.False(t, bitmap.Contains(3))
	assert.True(t, bitmap.Contains(4))
}

func TestUnifiedIndex_CompileFilter_MultipleConditions(t *testing.T) {
	ui := NewUnifiedIndex()

	ui.Set(1, metadata.Document{"category": metadata.String("tech"), "year": metadata.Int(2024)})
	ui.Set(2, metadata.Document{"category": metadata.String("tech"), "year": metadata.Int(2023)})
	ui.Set(3, metadata.Document{"category": metadata.String("science"), "year": metadata.Int(2024)})

	// Compile filter: category == "tech" AND year == 2024
	fs := &metadata.FilterSet{
		Filters: []metadata.Filter{
			{Key: "category", Operator: metadata.OpEqual, Value: metadata.String("tech")},
			{Key: "year", Operator: metadata.OpEqual, Value: metadata.Int(2024)},
		},
	}

	bitmap := ui.CompileFilter(fs)
	require.NotNil(t, bitmap)
	assert.Equal(t, uint64(1), bitmap.Cardinality())
	assert.True(t, bitmap.Contains(1))
	assert.False(t, bitmap.Contains(2))
	assert.False(t, bitmap.Contains(3))
}

func TestUnifiedIndex_CompileFilter_NoMatches(t *testing.T) {
	ui := NewUnifiedIndex()

	ui.Set(1, metadata.Document{"category": metadata.String("tech")})

	// Filter with no matches
	fs := &metadata.FilterSet{
		Filters: []metadata.Filter{
			{Key: "category", Operator: metadata.OpEqual, Value: metadata.String("nonexistent")},
		},
	}

	bitmap := ui.CompileFilter(fs)
	require.NotNil(t, bitmap)
	assert.True(t, bitmap.IsEmpty())
}

func TestUnifiedIndex_ScanFilter(t *testing.T) {
	ui := NewUnifiedIndex()

	ui.Set(1, metadata.Document{"score": metadata.Int(100)})
	ui.Set(2, metadata.Document{"score": metadata.Int(50)})
	ui.Set(3, metadata.Document{"score": metadata.Int(150)})

	// Scan filter: score > 75 (not supported by CompileFilter)
	fs := &metadata.FilterSet{
		Filters: []metadata.Filter{
			{Key: "score", Operator: metadata.OpGreaterThan, Value: metadata.Int(75)},
		},
	}

	// CompileFilter should return nil (operator not supported)
	bitmap := ui.CompileFilter(fs)
	assert.Nil(t, bitmap)

	// ScanFilter should work
	ids := ui.ScanFilter(fs)
	assert.Len(t, ids, 2)
	assert.Contains(t, ids, model.RowID(1))
	assert.Contains(t, ids, model.RowID(3))
}

func TestUnifiedIndex_CreateFilterFunc(t *testing.T) {
	ui := NewUnifiedIndex()

	ui.Set(1, metadata.Document{"category": metadata.String("tech")})
	ui.Set(2, metadata.Document{"category": metadata.String("science")})
	ui.Set(3, metadata.Document{"category": metadata.String("tech")})

	// Create filter function
	fs := &metadata.FilterSet{
		Filters: []metadata.Filter{
			{Key: "category", Operator: metadata.OpEqual, Value: metadata.String("tech")},
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
	ui.Set(1, metadata.Document{"category": metadata.String("tech")})

	// Verify inverted index has entry
	fs1 := &metadata.FilterSet{
		Filters: []metadata.Filter{
			{Key: "category", Operator: metadata.OpEqual, Value: metadata.String("tech")},
		},
	}
	bitmap1 := ui.CompileFilter(fs1)
	require.NotNil(t, bitmap1)
	assert.True(t, bitmap1.Contains(1))

	// Update to different category
	ui.Set(1, metadata.Document{"category": metadata.String("science")})

	// Old inverted index entry should be cleaned up
	bitmap2 := ui.CompileFilter(fs1)
	require.NotNil(t, bitmap2)
	assert.False(t, bitmap2.Contains(1)) // No longer in "tech"

	// New category should be indexed
	fs2 := &metadata.FilterSet{
		Filters: []metadata.Filter{
			{Key: "category", Operator: metadata.OpEqual, Value: metadata.String("science")},
		},
	}
	bitmap3 := ui.CompileFilter(fs2)
	require.NotNil(t, bitmap3)
	assert.True(t, bitmap3.Contains(1))
}

func TestUnifiedIndex_InvertedIndexDelete(t *testing.T) {
	ui := NewUnifiedIndex()

	ui.Set(1, metadata.Document{"category": metadata.String("tech")})
	ui.Set(2, metadata.Document{"category": metadata.String("tech")})

	// Both should match
	fs := &metadata.FilterSet{
		Filters: []metadata.Filter{
			{Key: "category", Operator: metadata.OpEqual, Value: metadata.String("tech")},
		},
	}
	bitmap := ui.CompileFilter(fs)
	require.NotNil(t, bitmap)
	assert.Equal(t, uint64(2), bitmap.Cardinality())

	// Delete one
	ui.Delete(1)

	// Only one should remain
	bitmap = ui.CompileFilter(fs)
	require.NotNil(t, bitmap)
	assert.Equal(t, uint64(1), bitmap.Cardinality())
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
	ui.Set(1, metadata.Document{"category": metadata.String("tech"), "year": metadata.Int(2024)})
	ui.Set(2, metadata.Document{"category": metadata.String("science"), "year": metadata.Int(2024)})
	ui.Set(3, metadata.Document{"category": metadata.String("tech"), "year": metadata.Int(2023)})

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

	ui.Set(1, metadata.Document{"key": metadata.String("value")})
	assert.Equal(t, 1, ui.Len())

	ui.Set(2, metadata.Document{"key": metadata.String("value")})
	assert.Equal(t, 2, ui.Len())

	ui.Delete(1)
	assert.Equal(t, 1, ui.Len())
}

// Benchmark CompileFilter vs ScanFilter
func BenchmarkUnifiedIndex_CompileFilter(b *testing.B) {
	ui := NewUnifiedIndex()

	// Add 10k documents
	for i := model.RowID(0); i < 10000; i++ {
		category := "tech"
		if i%3 == 0 {
			category = "science"
		}
		ui.Set(i, metadata.Document{"category": metadata.String(category)})
	}

	fs := &metadata.FilterSet{
		Filters: []metadata.Filter{
			{Key: "category", Operator: metadata.OpEqual, Value: metadata.String("tech")},
		},
	}

	b.ResetTimer()
	for b.Loop() {
		bitmap := ui.CompileFilter(fs)
		_ = bitmap.Cardinality()
	}
}

func BenchmarkUnifiedIndex_ScanFilter(b *testing.B) {
	ui := NewUnifiedIndex()

	// Add 10k documents
	for i := model.RowID(0); i < 10000; i++ {
		score := int64(i % 100)
		ui.Set(i, metadata.Document{"score": metadata.Int(score)})
	}

	fs := &metadata.FilterSet{
		Filters: []metadata.Filter{
			{Key: "score", Operator: metadata.OpGreaterThan, Value: metadata.Int(50)},
		},
	}

	b.ResetTimer()
	for b.Loop() {
		ids := ui.ScanFilter(fs)
		_ = len(ids)
	}
}

// BenchmarkUnifiedIndex_EvaluateFilter_MultiFilter tests lazy execution with multiple filters.
// This measures the improvement from zero-copy first filter and deferred cloning.
func BenchmarkUnifiedIndex_EvaluateFilter_MultiFilter(b *testing.B) {
	ui := NewUnifiedIndex()

	// Add 10k documents with multiple fields
	for i := model.RowID(0); i < 10000; i++ {
		category := "tech"
		if i%3 == 0 {
			category = "science"
		}
		if i%5 == 0 {
			category = "art"
		}
		status := "active"
		if i%4 == 0 {
			status = "inactive"
		}
		tier := "free"
		if i%2 == 0 {
			tier = "paid"
		}
		ui.Set(i, metadata.Document{
			"category": metadata.String(category),
			"status":   metadata.String(status),
			"tier":     metadata.String(tier),
		})
	}

	b.Run("SingleFilter", func(b *testing.B) {
		fs := &metadata.FilterSet{
			Filters: []metadata.Filter{
				{Key: "category", Operator: metadata.OpEqual, Value: metadata.String("tech")},
			},
		}
		b.ResetTimer()
		for b.Loop() {
			bitmap := ui.EvaluateFilter(fs)
			_ = bitmap.Cardinality()
			PutPooledBitmap(bitmap)
		}
	})

	b.Run("TwoFilters", func(b *testing.B) {
		fs := &metadata.FilterSet{
			Filters: []metadata.Filter{
				{Key: "category", Operator: metadata.OpEqual, Value: metadata.String("tech")},
				{Key: "status", Operator: metadata.OpEqual, Value: metadata.String("active")},
			},
		}
		b.ResetTimer()
		for b.Loop() {
			bitmap := ui.EvaluateFilter(fs)
			_ = bitmap.Cardinality()
			PutPooledBitmap(bitmap)
		}
	})

	b.Run("ThreeFilters", func(b *testing.B) {
		fs := &metadata.FilterSet{
			Filters: []metadata.Filter{
				{Key: "category", Operator: metadata.OpEqual, Value: metadata.String("tech")},
				{Key: "status", Operator: metadata.OpEqual, Value: metadata.String("active")},
				{Key: "tier", Operator: metadata.OpEqual, Value: metadata.String("paid")},
			},
		}
		b.ResetTimer()
		for b.Loop() {
			bitmap := ui.EvaluateFilter(fs)
			_ = bitmap.Cardinality()
			PutPooledBitmap(bitmap)
		}
	})

	b.Run("InOperator_MultiValue", func(b *testing.B) {
		fs := &metadata.FilterSet{
			Filters: []metadata.Filter{
				{Key: "category", Operator: metadata.OpIn, Value: metadata.Array([]metadata.Value{
					metadata.String("tech"),
					metadata.String("science"),
				})},
			},
		}
		b.ResetTimer()
		for b.Loop() {
			bitmap := ui.EvaluateFilter(fs)
			_ = bitmap.Cardinality()
			PutPooledBitmap(bitmap)
		}
	})

	b.Run("InOperator_SingleValue", func(b *testing.B) {
		// Single-element In should use zero-copy path
		fs := &metadata.FilterSet{
			Filters: []metadata.Filter{
				{Key: "category", Operator: metadata.OpIn, Value: metadata.Array([]metadata.Value{
					metadata.String("tech"),
				})},
			},
		}
		b.ResetTimer()
		for b.Loop() {
			bitmap := ui.EvaluateFilter(fs)
			_ = bitmap.Cardinality()
			PutPooledBitmap(bitmap)
		}
	})
}
