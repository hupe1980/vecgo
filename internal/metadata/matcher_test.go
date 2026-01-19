package imetadata

import (
	"testing"

	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/model"
)

func TestFilterMatcher_Basic(t *testing.T) {
	ui := NewUnifiedIndex()

	// Add some test data
	for i := uint32(0); i < 1000; i++ {
		doc := metadata.Document{
			"category": metadata.String("A"),
			"price":    metadata.Float(float64(i)),
		}
		if i%2 == 0 {
			doc["category"] = metadata.String("B")
		}
		ui.Set(model.RowID(i), doc)
	}

	// Seal numeric index for range queries
	ui.numeric.Seal()

	t.Run("SingleEquality", func(t *testing.T) {
		fs := &metadata.FilterSet{
			Filters: []metadata.Filter{
				{Key: "category", Operator: metadata.OpEqual, Value: metadata.String("A")},
			},
		}

		scratch := GetMatcherScratch()
		defer PutMatcherScratch(scratch)

		ui.mu.RLock()
		matcher := ui.GetFilterMatcher(fs, scratch)
		ui.mu.RUnlock()
		defer matcher.Release()

		// Should match odd rows (category=A)
		matchCount := 0
		for i := uint32(0); i < 1000; i++ {
			if matcher.Matches(i) {
				matchCount++
				if i%2 == 0 {
					t.Errorf("Row %d matched but has category=B", i)
				}
			}
		}
		if matchCount != 500 {
			t.Errorf("Expected 500 matches, got %d", matchCount)
		}
	})

	t.Run("NumericRange", func(t *testing.T) {
		fs := &metadata.FilterSet{
			Filters: []metadata.Filter{
				{Key: "price", Operator: metadata.OpLessThan, Value: metadata.Float(100)},
			},
		}

		scratch := GetMatcherScratch()
		defer PutMatcherScratch(scratch)

		ui.mu.RLock()
		matcher := ui.GetFilterMatcher(fs, scratch)
		ui.mu.RUnlock()
		defer matcher.Release()

		// Should match rows 0-99
		matchCount := 0
		for i := uint32(0); i < 1000; i++ {
			if matcher.Matches(i) {
				matchCount++
				if i >= 100 {
					t.Errorf("Row %d matched but price >= 100", i)
				}
			}
		}
		if matchCount != 100 {
			t.Errorf("Expected 100 matches, got %d", matchCount)
		}
	})

	t.Run("Composite", func(t *testing.T) {
		fs := &metadata.FilterSet{
			Filters: []metadata.Filter{
				{Key: "category", Operator: metadata.OpEqual, Value: metadata.String("A")},
				{Key: "price", Operator: metadata.OpLessThan, Value: metadata.Float(100)},
			},
		}

		scratch := GetMatcherScratch()
		defer PutMatcherScratch(scratch)

		ui.mu.RLock()
		matcher := ui.GetFilterMatcher(fs, scratch)
		ui.mu.RUnlock()
		defer matcher.Release()

		// Should match odd rows with price < 100
		matchCount := 0
		for i := uint32(0); i < 1000; i++ {
			if matcher.Matches(i) {
				matchCount++
				if i%2 == 0 {
					t.Errorf("Row %d matched but has category=B", i)
				}
				if i >= 100 {
					t.Errorf("Row %d matched but price >= 100", i)
				}
			}
		}
		// Odd numbers from 0-99: 1,3,5,...,99 = 50 numbers
		if matchCount != 50 {
			t.Errorf("Expected 50 matches, got %d", matchCount)
		}
	})

	t.Run("EmptyFilter", func(t *testing.T) {
		fs := &metadata.FilterSet{}

		scratch := GetMatcherScratch()
		defer PutMatcherScratch(scratch)

		ui.mu.RLock()
		matcher := ui.GetFilterMatcher(fs, scratch)
		ui.mu.RUnlock()
		defer matcher.Release()

		// Should match all rows (AlwaysTrueMatcher)
		if _, ok := matcher.(*AlwaysTrueMatcher); !ok {
			t.Errorf("Expected AlwaysTrueMatcher, got %T", matcher)
		}
	})

	t.Run("ImpossibleFilter", func(t *testing.T) {
		fs := &metadata.FilterSet{
			Filters: []metadata.Filter{
				{Key: "nonexistent", Operator: metadata.OpEqual, Value: metadata.String("X")},
			},
		}

		scratch := GetMatcherScratch()
		defer PutMatcherScratch(scratch)

		ui.mu.RLock()
		matcher := ui.GetFilterMatcher(fs, scratch)
		ui.mu.RUnlock()
		defer matcher.Release()

		// Should match nothing (AlwaysFalseMatcher)
		if _, ok := matcher.(*AlwaysFalseMatcher); !ok {
			t.Errorf("Expected AlwaysFalseMatcher, got %T", matcher)
		}
	})
}

func TestMatcherCursor_Basic(t *testing.T) {
	ui := NewUnifiedIndex()

	// Add some test data
	for i := uint32(0); i < 100; i++ {
		doc := metadata.Document{
			"category": metadata.String("A"),
		}
		if i%2 == 0 {
			doc["category"] = metadata.String("B")
		}
		ui.Set(model.RowID(i), doc)
	}

	t.Run("IterateWithMatcher", func(t *testing.T) {
		fs := &metadata.FilterSet{
			Filters: []metadata.Filter{
				{Key: "category", Operator: metadata.OpEqual, Value: metadata.String("A")},
			},
		}

		scratch := GetMatcherScratch()
		defer PutMatcherScratch(scratch)

		ui.mu.RLock()
		cursor := ui.FilterCursorWithMatcher(fs, 100, scratch)
		ui.mu.RUnlock()
		defer cursor.Release()

		// Iterate and collect matches
		var matches []uint32
		cursor.ForEach(func(rowID uint32) bool {
			matches = append(matches, rowID)
			return true
		})

		// Should have 50 odd rows
		if len(matches) != 50 {
			t.Errorf("Expected 50 matches, got %d", len(matches))
		}

		// Verify all are odd
		for _, m := range matches {
			if m%2 == 0 {
				t.Errorf("Row %d is even but matched", m)
			}
		}
	})
}

func BenchmarkFilterMatcher_ZeroAlloc(b *testing.B) {
	ui := NewUnifiedIndex()

	// Add test data
	for i := uint32(0); i < 10000; i++ {
		doc := metadata.Document{
			"category": metadata.String("A"),
			"price":    metadata.Float(float64(i % 100)),
		}
		if i%2 == 0 {
			doc["category"] = metadata.String("B")
		}
		ui.Set(model.RowID(i), doc)
	}
	ui.numeric.Seal()

	fs := &metadata.FilterSet{
		Filters: []metadata.Filter{
			{Key: "category", Operator: metadata.OpEqual, Value: metadata.String("A")},
			{Key: "price", Operator: metadata.OpLessThan, Value: metadata.Float(50)},
		},
	}

	b.Run("OldPath_Closure", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			ui.mu.RLock()
			cursor := ui.FilterCursor(fs, 10000)
			var count int
			cursor.ForEach(func(rowID uint32) bool {
				count++
				return true
			})
			ui.mu.RUnlock()
			_ = count
		}
	})

	b.Run("NewPath_Matcher", func(b *testing.B) {
		b.ReportAllocs()
		scratch := GetMatcherScratch()
		defer PutMatcherScratch(scratch)

		for i := 0; i < b.N; i++ {
			ui.mu.RLock()
			cursor := ui.FilterCursorWithMatcher(fs, 10000, scratch)
			var count int
			cursor.ForEach(func(rowID uint32) bool {
				count++
				return true
			})
			cursor.Release()
			ui.mu.RUnlock()
			_ = count
		}
	})
}
