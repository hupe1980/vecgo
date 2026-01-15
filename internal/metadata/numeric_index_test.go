package imetadata

import (
	"bytes"
	"testing"

	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/model"
)

func TestNumericIndex_Add(t *testing.T) {
	ni := NewNumericIndex()

	// Add some values
	ni.Add("price", metadata.Float(10.5), 1)
	ni.Add("price", metadata.Float(20.0), 2)
	ni.Add("price", metadata.Float(15.0), 3)
	ni.Add("count", metadata.Int(100), 4)
	ni.Add("name", metadata.String("test"), 5) // Should be ignored

	if !ni.HasField("price") {
		t.Error("Expected price field to exist")
	}
	if !ni.HasField("count") {
		t.Error("Expected count field to exist")
	}
	if ni.HasField("name") {
		t.Error("Did not expect name field to exist (string value)")
	}

	if ni.Cardinality() != 4 {
		t.Errorf("Expected cardinality 4, got %d", ni.Cardinality())
	}
	if ni.FieldCount() != 2 {
		t.Errorf("Expected 2 fields, got %d", ni.FieldCount())
	}
}

func TestNumericIndex_QueryRange(t *testing.T) {
	ni := NewNumericIndex()

	// Add values
	for i := 0; i < 100; i++ {
		ni.Add("value", metadata.Int(int64(i)), model.RowID(i))
	}
	ni.Seal() // Sort for efficient queries

	tests := []struct {
		name       string
		min        float64
		max        float64
		includeMin bool
		includeMax bool
		wantCount  int
	}{
		{"All values", 0, 100, true, false, 100},
		{"First 10", 0, 10, true, false, 10},
		{"10-20 inclusive", 10, 20, true, true, 11},
		{"10-20 exclusive", 10, 20, false, false, 9},
		{"Single value", 50, 50, true, true, 1},
		{"Empty range", 50, 49, true, true, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := NewLocalBitmap()
			ni.QueryRange("value", tt.min, tt.max, tt.includeMin, tt.includeMax, result)
			if got := int(result.Cardinality()); got != tt.wantCount {
				t.Errorf("QueryRange() got %d, want %d", got, tt.wantCount)
			}
		})
	}
}

func TestNumericIndex_QueryRangeSIMD(t *testing.T) {
	ni := NewNumericIndex()

	// Add values 0-99
	for i := 0; i < 100; i++ {
		ni.Add("value", metadata.Int(int64(i)), model.RowID(i))
	}
	// Note: QueryRangeSIMD works on unsorted data too
	ni.Seal()

	tests := []struct {
		name      string
		min       float64
		max       float64
		wantCount int
	}{
		{"All values", 0, 99, 100},
		{"First 10", 0, 9, 10},
		{"10-20 inclusive", 10, 20, 11},
		{"Single value", 50, 50, 1},
		{"Empty range", 100, 200, 0},
		{"Negative range", -10, -1, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := NewLocalBitmap()
			ni.QueryRangeSIMD("value", tt.min, tt.max, result)
			if got := int(result.Cardinality()); got != tt.wantCount {
				t.Errorf("QueryRangeSIMD() got %d, want %d", got, tt.wantCount)
			}
		})
	}

	// Verify SIMD and binary search produce same results
	t.Run("SIMD matches BinarySearch", func(t *testing.T) {
		simdResult := NewLocalBitmap()
		bsResult := NewLocalBitmap()

		ni.QueryRangeSIMD("value", 25, 75, simdResult)
		ni.QueryRange("value", 25, 75, true, true, bsResult)

		if simdResult.Cardinality() != bsResult.Cardinality() {
			t.Errorf("SIMD=%d vs BinarySearch=%d", simdResult.Cardinality(), bsResult.Cardinality())
		}
	})
}

func TestNumericIndex_EvaluateFilter(t *testing.T) {
	ni := NewNumericIndex()

	// Add values 0-99
	for i := 0; i < 100; i++ {
		ni.Add("bucket", metadata.Int(int64(i%10)), model.RowID(i)) // Values 0-9, 10 each
	}
	ni.Seal()

	tests := []struct {
		name      string
		op        metadata.Operator
		value     metadata.Value
		wantCount int
	}{
		{"LessThan 5", metadata.OpLessThan, metadata.Int(5), 50},         // 0-4 = 50 rows
		{"LessEqual 5", metadata.OpLessEqual, metadata.Int(5), 60},       // 0-5 = 60 rows
		{"GreaterThan 7", metadata.OpGreaterThan, metadata.Int(7), 20},   // 8-9 = 20 rows
		{"GreaterEqual 7", metadata.OpGreaterEqual, metadata.Int(7), 30}, // 7-9 = 30 rows
		{"NotEqual 5", metadata.OpNotEqual, metadata.Int(5), 90},         // All except 5 = 90 rows
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := metadata.Filter{Key: "bucket", Operator: tt.op, Value: tt.value}
			result := ni.EvaluateFilter(f)
			defer PutPooledBitmap(result)
			if got := int(result.Cardinality()); got != tt.wantCount {
				t.Errorf("EvaluateFilter() got %d, want %d", got, tt.wantCount)
			}
		})
	}
}

func TestNumericIndex_PrefixBitmapEdgeCases(t *testing.T) {
	ni := NewNumericIndex()

	// Low-cardinality: 5 distinct values (0, 2, 4, 6, 8), 20 rows each
	for i := 0; i < 100; i++ {
		val := (i / 20) * 2 // Creates: 20 x 0, 20 x 2, 20 x 4, 20 x 6, 20 x 8
		ni.Add("price", metadata.Int(int64(val)), model.RowID(i))
	}
	ni.Seal()

	// Verify it uses prefix bitmaps (low cardinality)
	if !ni.IsLowCardinality("price") {
		t.Fatal("Expected low-cardinality field to have bitmap index")
	}

	tests := []struct {
		name      string
		op        metadata.Operator
		value     int64
		wantCount int
	}{
		// Edge: filter on non-existent value (between existing values)
		{"LessThan 3 (non-existent)", metadata.OpLessThan, 3, 40},         // 0, 2 = 40
		{"LessEqual 3 (non-existent)", metadata.OpLessEqual, 3, 40},       // 0, 2 = 40
		{"GreaterThan 3 (non-existent)", metadata.OpGreaterThan, 3, 60},   // 4, 6, 8 = 60
		{"GreaterEqual 3 (non-existent)", metadata.OpGreaterEqual, 3, 60}, // 4, 6, 8 = 60
		{"NotEqual 3 (non-existent)", metadata.OpNotEqual, 3, 100},        // All = 100

		// Edge: filter below min value
		{"LessThan -1 (below min)", metadata.OpLessThan, -1, 0},
		{"GreaterThan -1 (below min)", metadata.OpGreaterThan, -1, 100},

		// Edge: filter above max value
		{"LessThan 10 (above max)", metadata.OpLessThan, 10, 100},
		{"GreaterThan 10 (above max)", metadata.OpGreaterThan, 10, 0},

		// Edge: filter on exact boundary values
		{"LessThan 0 (at min)", metadata.OpLessThan, 0, 0},
		{"LessEqual 0 (at min)", metadata.OpLessEqual, 0, 20},
		{"GreaterThan 8 (at max)", metadata.OpGreaterThan, 8, 0},
		{"GreaterEqual 8 (at max)", metadata.OpGreaterEqual, 8, 20},

		// Normal cases
		{"Equal 4", metadata.OpEqual, 4, 20},
		{"LessThan 4", metadata.OpLessThan, 4, 40},         // 0, 2 = 40
		{"LessEqual 4", metadata.OpLessEqual, 4, 60},       // 0, 2, 4 = 60
		{"GreaterThan 4", metadata.OpGreaterThan, 4, 40},   // 6, 8 = 40
		{"GreaterEqual 4", metadata.OpGreaterEqual, 4, 60}, // 4, 6, 8 = 60
		{"NotEqual 4", metadata.OpNotEqual, 4, 80},         // All except 4 = 80
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := metadata.Filter{Key: "price", Operator: tt.op, Value: metadata.Int(tt.value)}
			result := ni.EvaluateFilter(f)
			defer PutPooledBitmap(result)
			if got := int(result.Cardinality()); got != tt.wantCount {
				t.Errorf("EvaluateFilter(%s %d) got %d, want %d", tt.op, tt.value, got, tt.wantCount)
			}
		})
	}
}

func TestNumericIndex_Remove(t *testing.T) {
	ni := NewNumericIndex()

	ni.Add("value", metadata.Int(10), 1)
	ni.Add("value", metadata.Int(20), 2)
	ni.Add("value", metadata.Int(10), 3)
	ni.Seal()

	if ni.Cardinality() != 3 {
		t.Fatalf("Expected 3 entries, got %d", ni.Cardinality())
	}

	// Remove one entry (deferred deletion)
	ni.Remove("value", metadata.Int(10), 1)

	// Cardinality doesn't change until Seal() - pending delete
	if ni.Cardinality() != 3 {
		t.Errorf("Expected 3 entries before Seal (deferred delete), got %d", ni.Cardinality())
	}

	// But QueryRange should exclude pending deletes
	result := NewLocalBitmap()
	ni.QueryRange("value", 10, 10, true, true, result)
	if result.Cardinality() != 1 {
		t.Errorf("Expected 1 match for value=10 (pending delete excluded), got %d", result.Cardinality())
	}
	if !result.Contains(3) {
		t.Error("Expected rowID 3 to match value=10")
	}
	if result.Contains(1) {
		t.Error("Expected rowID 1 to be excluded (pending delete)")
	}

	// After Seal(), cardinality reflects actual entries
	ni.Seal()
	if ni.Cardinality() != 2 {
		t.Errorf("Expected 2 entries after Seal, got %d", ni.Cardinality())
	}
}

func TestNumericIndex_Persistence(t *testing.T) {
	ni := NewNumericIndex()

	// Add values
	ni.Add("price", metadata.Float(10.5), 1)
	ni.Add("price", metadata.Float(20.0), 2)
	ni.Add("count", metadata.Int(100), 3)
	ni.Seal()

	// Write to buffer
	var buf bytes.Buffer
	_, err := ni.WriteTo(&buf)
	if err != nil {
		t.Fatalf("WriteTo failed: %v", err)
	}

	// Read back
	ni2 := NewNumericIndex()
	_, err = ni2.ReadFrom(&buf)
	if err != nil {
		t.Fatalf("ReadFrom failed: %v", err)
	}

	// Verify
	if ni2.FieldCount() != ni.FieldCount() {
		t.Errorf("Field count mismatch: got %d, want %d", ni2.FieldCount(), ni.FieldCount())
	}
	if ni2.Cardinality() != ni.Cardinality() {
		t.Errorf("Cardinality mismatch: got %d, want %d", ni2.Cardinality(), ni.Cardinality())
	}

	// Test query on loaded index
	f := metadata.Filter{Key: "price", Operator: metadata.OpLessThan, Value: metadata.Float(15.0)}
	result := ni2.EvaluateFilter(f)
	defer PutPooledBitmap(result)
	if result.Cardinality() != 1 {
		t.Errorf("Expected 1 match for price<15, got %d", result.Cardinality())
	}
}

func TestUnifiedIndex_NumericFilter(t *testing.T) {
	ui := NewUnifiedIndex()

	// Add documents with numeric fields
	for i := 0; i < 100; i++ {
		ui.Set(model.RowID(i), metadata.Document{
			"bucket": metadata.Int(int64(i % 10)),
			"value":  metadata.Float(float64(i)),
		})
	}

	// Test LessThan filter using NumericIndex
	fs := metadata.NewFilterSet(metadata.Filter{
		Key:      "bucket",
		Operator: metadata.OpLessThan,
		Value:    metadata.Int(3), // 0, 1, 2 = 30 rows
	})

	bitmap := ui.EvaluateFilter(fs)
	defer PutPooledBitmap(bitmap)

	if bitmap.Cardinality() != 30 {
		t.Errorf("Expected 30 matches for bucket<3, got %d", bitmap.Cardinality())
	}

	// Test combined filter
	fs2 := metadata.NewFilterSet(
		metadata.Filter{Key: "bucket", Operator: metadata.OpLessThan, Value: metadata.Int(3)},
		metadata.Filter{Key: "value", Operator: metadata.OpGreaterEqual, Value: metadata.Float(50)},
	)

	bitmap2 := ui.EvaluateFilter(fs2)
	defer PutPooledBitmap(bitmap2)

	// bucket<3: 0,1,2 (at positions 0,1,2,10,11,12,20,21,22,...,90,91,92)
	// value>=50: 50-99
	// Combined: positions 50,51,52,60,61,62,70,71,72,80,81,82,90,91,92 = 15 rows
	if bitmap2.Cardinality() != 15 {
		t.Errorf("Expected 15 matches, got %d", bitmap2.Cardinality())
	}
}

func BenchmarkNumericIndex_QueryRange(b *testing.B) {
	ni := NewNumericIndex()

	// Add 100K values
	for i := 0; i < 100_000; i++ {
		ni.Add("bucket", metadata.Int(int64(i%100)), model.RowID(i))
	}
	ni.Seal()

	result := NewLocalBitmap()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		result.Clear()
		ni.QueryRange("bucket", 0, 5, true, false, result)
	}
	b.ReportMetric(float64(result.Cardinality()), "matches")
}

func BenchmarkNumericIndex_QueryRangeSIMD(b *testing.B) {
	ni := NewNumericIndex()

	// Add 100K values
	for i := 0; i < 100_000; i++ {
		ni.Add("value", metadata.Float(float64(i)), model.RowID(i))
	}
	ni.Seal()

	result := NewLocalBitmap()

	b.Run("BinarySearch_Narrow", func(b *testing.B) {
		// Narrow range: 5% selectivity - binary search should win
		for i := 0; i < b.N; i++ {
			result.Clear()
			ni.QueryRange("value", 0, 5000, true, true, result)
		}
	})

	b.Run("SIMD_Narrow", func(b *testing.B) {
		// Narrow range with SIMD full scan
		for i := 0; i < b.N; i++ {
			result.Clear()
			ni.QueryRangeSIMD("value", 0, 5000, result)
		}
	})

	b.Run("BinarySearch_Wide", func(b *testing.B) {
		// Wide range: 50% selectivity
		for i := 0; i < b.N; i++ {
			result.Clear()
			ni.QueryRange("value", 0, 50000, true, true, result)
		}
	})

	b.Run("SIMD_Wide", func(b *testing.B) {
		// Wide range with SIMD full scan
		for i := 0; i < b.N; i++ {
			result.Clear()
			ni.QueryRangeSIMD("value", 0, 50000, result)
		}
	})

	b.Run("BinarySearch_VeryWide", func(b *testing.B) {
		// Very wide range: 90% selectivity
		for i := 0; i < b.N; i++ {
			result.Clear()
			ni.QueryRange("value", 0, 90000, true, true, result)
		}
	})

	b.Run("SIMD_VeryWide", func(b *testing.B) {
		// Very wide range with SIMD full scan
		for i := 0; i < b.N; i++ {
			result.Clear()
			ni.QueryRangeSIMD("value", 0, 90000, result)
		}
	})
}

func BenchmarkNumericIndex_vs_InvertedScan(b *testing.B) {
	// Test with HIGH cardinality (10K distinct values) - this is where NumericIndex wins
	// The inverted scan has to iterate 10K buckets, while NumericIndex uses binary search
	distinctValues := 10_000
	totalRows := 100_000

	ni := NewNumericIndex()
	ui := NewUnifiedIndex()

	// Add 100K values with 10K distinct buckets
	for i := 0; i < totalRows; i++ {
		doc := metadata.Document{"bucket": metadata.Int(int64(i % distinctValues))}
		ui.Set(model.RowID(i), doc)
		ni.Add("bucket", metadata.Int(int64(i%distinctValues)), model.RowID(i))
	}
	ni.Seal()

	// Query for bucket < 500 (5% of 10K values = 500 values matching)
	f := metadata.Filter{Key: "bucket", Operator: metadata.OpLessThan, Value: metadata.Int(500)}

	b.Run("NumericIndex_HighCardinality", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			result := ni.EvaluateFilter(f)
			PutPooledBitmap(result)
		}
	})

	b.Run("InvertedScan_HighCardinality", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			ui.mu.RLock()
			result := GetPooledBitmap()
			ui.evaluateNumericFilterScanInto(f, result)
			ui.mu.RUnlock()
			PutPooledBitmap(result)
		}
	})
}

func BenchmarkNumericIndex_LowCardinality(b *testing.B) {
	// Test with LOW cardinality - here inverted scan is fine
	distinctValues := 100
	totalRows := 100_000

	ni := NewNumericIndex()
	ui := NewUnifiedIndex()

	for i := 0; i < totalRows; i++ {
		doc := metadata.Document{"bucket": metadata.Int(int64(i % distinctValues))}
		ui.Set(model.RowID(i), doc)
		ni.Add("bucket", metadata.Int(int64(i%distinctValues)), model.RowID(i))
	}
	ni.Seal()

	f := metadata.Filter{Key: "bucket", Operator: metadata.OpLessThan, Value: metadata.Int(5)}

	b.Run("NumericIndex", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			result := ni.EvaluateFilter(f)
			PutPooledBitmap(result)
		}
	})

	b.Run("InvertedScan", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			ui.mu.RLock()
			result := GetPooledBitmap()
			ui.evaluateNumericFilterScanInto(f, result)
			ui.mu.RUnlock()
			PutPooledBitmap(result)
		}
	})
}
