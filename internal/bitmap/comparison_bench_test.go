package bitmap

import (
	"testing"

	"github.com/RoaringBitmap/roaring/v2"
)

// Comparative benchmarks: QueryBitmap vs Roaring Bitmap
// Run with: go test -bench=. -benchmem ./internal/bitmap/

// ==============================================================================
// AddRange / AddMany comparison
// ==============================================================================

func BenchmarkComparison_AddRange_QueryBitmap(b *testing.B) {
	pool := NewQueryBitmapPool(100000)
	qb := pool.Get()
	defer pool.Put(qb)

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		qb.Clear()
		qb.AddRange(0, 10000)
	}
}

func BenchmarkComparison_AddRange_Roaring(b *testing.B) {
	rb := roaring.New()

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		rb.Clear()
		rb.AddRange(0, 10000)
	}
}

// ==============================================================================
// AND operation comparison
// ==============================================================================

func BenchmarkComparison_And_QueryBitmap(b *testing.B) {
	pool := NewQueryBitmapPool(100000)
	a := pool.Get()
	x := pool.Get()
	defer pool.Put(a)
	defer pool.Put(x)

	x.Clear()
	x.AddRange(5000, 15000)

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		a.Clear()
		a.AddRange(0, 10000)
		a.And(x)
	}
}

func BenchmarkComparison_And_Roaring(b *testing.B) {
	a := roaring.New()
	x := roaring.New()
	x.AddRange(5000, 15000)

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		a.Clear()
		a.AddRange(0, 10000)
		a.And(x)
	}
}

// ==============================================================================
// Cardinality (popcount) comparison
// ==============================================================================

func BenchmarkComparison_Cardinality_QueryBitmap(b *testing.B) {
	qb := New(100000)
	qb.AddRange(0, 50000)

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		qb.cardinality = -1
		_ = qb.Cardinality()
	}
}

func BenchmarkComparison_Cardinality_Roaring(b *testing.B) {
	rb := roaring.New()
	rb.AddRange(0, 50000)

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = rb.GetCardinality()
	}
}

// ==============================================================================
// Iteration comparison
// ==============================================================================

func BenchmarkComparison_ForEach_10K_QueryBitmap(b *testing.B) {
	qb := New(100000)
	qb.AddRange(0, 10000)

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		count := 0
		qb.ForEach(func(id uint32) bool {
			count++
			return true
		})
	}
}

func BenchmarkComparison_ForEach_10K_Roaring(b *testing.B) {
	rb := roaring.New()
	rb.AddRange(0, 10000)

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		count := 0
		rb.Iterate(func(id uint32) bool {
			count++
			return true
		})
	}
}

// ==============================================================================
// ToArray/ToSlice comparison
// ==============================================================================

func BenchmarkComparison_ToSlice_10K_QueryBitmap(b *testing.B) {
	qb := New(100000)
	qb.AddRange(0, 10000)
	scratch := make([]uint32, 0, 10000)

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = qb.ToSlice(scratch[:0])
	}
}

func BenchmarkComparison_ToSlice_10K_Roaring(b *testing.B) {
	rb := roaring.New()
	rb.AddRange(0, 10000)

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = rb.ToArray()
	}
}

// ==============================================================================
// Pool vs New comparison (allocation overhead)
// ==============================================================================

func BenchmarkComparison_Pool_QueryBitmap(b *testing.B) {
	pool := NewQueryBitmapPool(100000)

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		qb := pool.Get()
		qb.Clear()
		qb.AddRange(0, 10000)
		pool.Put(qb)
	}
}

func BenchmarkComparison_New_Roaring(b *testing.B) {
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		rb := roaring.New()
		rb.AddRange(0, 10000)
	}
}

// ==============================================================================
// Complex filter scenario: simulate filtered search
// ==============================================================================

func BenchmarkComparison_FilteredSearch_QueryBitmap(b *testing.B) {
	pool := NewQueryBitmapPool(100000)

	// Simulate: numeric filter AND categorical filter
	numericFilter := pool.Get()
	numericFilter.Clear()
	numericFilter.AddRange(10000, 50000) // 40K matches

	categoryFilter := pool.Get()
	categoryFilter.Clear()
	categoryFilter.AddRange(20000, 60000) // 40K matches

	scratch := make([]uint32, 0, 50000)

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		// Get a working bitmap
		result := pool.Get()
		result.Clear()

		// Copy numeric filter
		result.CopyFrom(numericFilter)

		// AND with category filter
		result.And(categoryFilter)

		// Extract matching IDs
		_ = result.ToSlice(scratch[:0])

		pool.Put(result)
	}

	pool.Put(numericFilter)
	pool.Put(categoryFilter)
}

func BenchmarkComparison_FilteredSearch_Roaring(b *testing.B) {
	// Simulate: numeric filter AND categorical filter
	numericFilter := roaring.New()
	numericFilter.AddRange(10000, 50000)

	categoryFilter := roaring.New()
	categoryFilter.AddRange(20000, 60000)

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		// Clone numeric filter
		result := numericFilter.Clone()

		// AND with category filter
		result.And(categoryFilter)

		// Extract matching IDs
		_ = result.ToArray()
	}
}

// ==============================================================================
// Memory efficiency: large bitmap operations
// ==============================================================================

func BenchmarkComparison_LargeBitmap_100K_QueryBitmap(b *testing.B) {
	pool := NewQueryBitmapPool(1000000)
	a := pool.Get()
	x := pool.Get()
	defer pool.Put(a)
	defer pool.Put(x)

	x.Clear()
	x.AddRange(50000, 150000)

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		a.Clear()
		a.AddRange(0, 100000)
		a.And(x)
		_ = a.Cardinality()
	}
}

func BenchmarkComparison_LargeBitmap_100K_Roaring(b *testing.B) {
	a := roaring.New()
	x := roaring.New()
	x.AddRange(50000, 150000)

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		a.Clear()
		a.AddRange(0, 100000)
		a.And(x)
		_ = a.GetCardinality()
	}
}
