package bitmap

import (
	"math/rand"
	"testing"
)

func TestQueryBitmap_Basic(t *testing.T) {
	qb := New(1000)

	// Test Add
	if !qb.Add(100) {
		t.Error("Add should return true for new bit")
	}
	if qb.Add(100) {
		t.Error("Add should return false for existing bit")
	}

	// Test Contains
	if !qb.Contains(100) {
		t.Error("Contains should return true for set bit")
	}
	if qb.Contains(200) {
		t.Error("Contains should return false for unset bit")
	}

	// Test Cardinality
	if c := qb.Cardinality(); c != 1 {
		t.Errorf("Cardinality = %d, want 1", c)
	}

	// Test Clear
	qb.Clear()
	if !qb.IsEmpty() {
		t.Error("IsEmpty should return true after Clear")
	}
}

func TestQueryBitmap_AddRange(t *testing.T) {
	tests := []struct {
		name       string
		start, end uint32
		wantCard   int
	}{
		{"single word", 0, 64, 64},
		{"cross word", 60, 70, 10},
		{"multiple words", 0, 200, 200},
		{"mid-word", 10, 50, 40},
		{"empty range", 100, 100, 0},
		{"large range", 0, 1000, 1000},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			qb := New(10000)
			qb.AddRange(tt.start, tt.end)

			if c := qb.Cardinality(); c != tt.wantCard {
				t.Errorf("Cardinality = %d, want %d", c, tt.wantCard)
			}

			// Verify each bit in range is set
			for i := tt.start; i < tt.end; i++ {
				if !qb.Contains(i) {
					t.Errorf("Contains(%d) = false, want true", i)
				}
			}

			// Verify bits outside range are not set
			if tt.start > 0 && qb.Contains(tt.start-1) {
				t.Errorf("Contains(%d) = true, want false", tt.start-1)
			}
			if tt.end < 10000 && qb.Contains(tt.end) {
				t.Errorf("Contains(%d) = true, want false", tt.end)
			}
		})
	}
}

func TestQueryBitmap_And(t *testing.T) {
	a := New(1000)
	b := New(1000)

	a.AddRange(100, 300) // [100, 300)
	b.AddRange(200, 400) // [200, 400)

	a.And(b)

	// Intersection should be [200, 300)
	if c := a.Cardinality(); c != 100 {
		t.Errorf("Cardinality after And = %d, want 100", c)
	}

	for i := uint32(200); i < 300; i++ {
		if !a.Contains(i) {
			t.Errorf("Contains(%d) = false, want true", i)
		}
	}

	// Outside intersection
	if a.Contains(100) {
		t.Error("Contains(100) = true after And, want false")
	}
	if a.Contains(350) {
		t.Error("Contains(350) = true after And, want false")
	}
}

func TestQueryBitmap_Or(t *testing.T) {
	a := New(1000)
	b := New(1000)

	a.AddRange(100, 200)
	b.AddRange(150, 250)

	a.Or(b)

	// Union should be [100, 250)
	if c := a.Cardinality(); c != 150 {
		t.Errorf("Cardinality after Or = %d, want 150", c)
	}

	for i := uint32(100); i < 250; i++ {
		if !a.Contains(i) {
			t.Errorf("Contains(%d) = false, want true", i)
		}
	}
}

func TestQueryBitmap_AndNot(t *testing.T) {
	a := New(1000)
	b := New(1000)

	a.AddRange(100, 300)
	b.AddRange(200, 400)

	a.AndNot(b)

	// Difference should be [100, 200)
	if c := a.Cardinality(); c != 100 {
		t.Errorf("Cardinality after AndNot = %d, want 100", c)
	}

	for i := uint32(100); i < 200; i++ {
		if !a.Contains(i) {
			t.Errorf("Contains(%d) = false, want true", i)
		}
	}
	for i := uint32(200); i < 300; i++ {
		if a.Contains(i) {
			t.Errorf("Contains(%d) = true, want false (removed by AndNot)", i)
		}
	}
}

func TestQueryBitmap_ForEach(t *testing.T) {
	qb := New(1000)
	qb.Add(10)
	qb.Add(100)
	qb.Add(500)

	var collected []uint32
	qb.ForEach(func(id uint32) bool {
		collected = append(collected, id)
		return true
	})

	if len(collected) != 3 {
		t.Errorf("ForEach collected %d items, want 3", len(collected))
	}

	// Should be in sorted order
	expected := []uint32{10, 100, 500}
	for i, v := range expected {
		if collected[i] != v {
			t.Errorf("collected[%d] = %d, want %d", i, collected[i], v)
		}
	}
}

func TestQueryBitmap_ToSlice(t *testing.T) {
	qb := New(1000)
	qb.Add(5)
	qb.Add(50)
	qb.Add(500)

	scratch := make([]uint32, 0, 10)
	result := qb.ToSlice(scratch)

	if len(result) != 3 {
		t.Errorf("ToSlice returned %d items, want 3", len(result))
	}

	expected := []uint32{5, 50, 500}
	for i, v := range expected {
		if result[i] != v {
			t.Errorf("result[%d] = %d, want %d", i, result[i], v)
		}
	}
}

func TestQueryBitmap_NextSet(t *testing.T) {
	qb := New(1000)
	qb.Add(0)
	qb.Add(5)
	qb.Add(64) // Cross word boundary
	qb.Add(100)
	qb.Add(500)

	tests := []struct {
		start uint32
		want  uint32
		ok    bool
	}{
		{0, 0, true},
		{1, 5, true},
		{5, 5, true},
		{6, 64, true},
		{64, 64, true},
		{65, 100, true},
		{100, 100, true},
		{101, 500, true},
		{500, 500, true},
		{501, 0, false}, // No more bits
		{999, 0, false},
	}

	for _, tt := range tests {
		got, ok := qb.NextSet(tt.start)
		if ok != tt.ok || (ok && got != tt.want) {
			t.Errorf("NextSet(%d) = (%d, %v), want (%d, %v)", tt.start, got, ok, tt.want, tt.ok)
		}
	}

	// Test iteration pattern
	var collected []uint32
	for i, ok := qb.NextSet(0); ok; i, ok = qb.NextSet(i + 1) {
		collected = append(collected, i)
	}
	expected := []uint32{0, 5, 64, 100, 500}
	if len(collected) != len(expected) {
		t.Errorf("Iteration collected %d items, want %d", len(collected), len(expected))
	}
	for i, v := range expected {
		if collected[i] != v {
			t.Errorf("collected[%d] = %d, want %d", i, collected[i], v)
		}
	}
}

func TestQueryBitmap_NextSetMany(t *testing.T) {
	qb := New(1000)
	qb.AddRange(0, 10)
	qb.Add(64)
	qb.Add(100)
	qb.Add(500)

	// Test with buffer larger than set bits
	buffer := make([]uint32, 20)
	lastIdx, result := qb.NextSetMany(0, buffer)

	if len(result) != 13 {
		t.Errorf("NextSetMany returned %d items, want 13", len(result))
	}
	if lastIdx != 500 {
		t.Errorf("lastIdx = %d, want 500", lastIdx)
	}

	// Test with small buffer (batch iteration)
	buffer = make([]uint32, 5)
	var allBits []uint32
	i := uint32(0)
	for {
		lastIdx, batch := qb.NextSetMany(i, buffer)
		if len(batch) == 0 {
			break
		}
		allBits = append(allBits, batch...)
		i = lastIdx + 1
	}

	if len(allBits) != 13 {
		t.Errorf("Batch iteration collected %d items, want 13", len(allBits))
	}
}

func TestQueryBitmap_Rank(t *testing.T) {
	qb := New(1000)
	qb.Add(0)
	qb.Add(5)
	qb.Add(64)
	qb.Add(100)

	tests := []struct {
		i    uint32
		want int
	}{
		{0, 1},   // Bit 0 is set
		{4, 1},   // Only bit 0 before bit 4
		{5, 2},   // Bits 0, 5
		{63, 2},  // Still bits 0, 5
		{64, 3},  // Bits 0, 5, 64
		{100, 4}, // All four bits
		{999, 4}, // Same as Cardinality
	}

	for _, tt := range tests {
		got := qb.Rank(tt.i)
		if got != tt.want {
			t.Errorf("Rank(%d) = %d, want %d", tt.i, got, tt.want)
		}
	}
}

func TestQueryBitmap_Select(t *testing.T) {
	qb := New(1000)
	qb.Add(0)
	qb.Add(5)
	qb.Add(64)
	qb.Add(100)

	tests := []struct {
		j    int
		want uint32
		ok   bool
	}{
		{0, 0, true},
		{1, 5, true},
		{2, 64, true},
		{3, 100, true},
		{4, 0, false}, // Out of range
		{-1, 0, false},
	}

	for _, tt := range tests {
		got, ok := qb.Select(tt.j)
		if ok != tt.ok || (ok && got != tt.want) {
			t.Errorf("Select(%d) = (%d, %v), want (%d, %v)", tt.j, got, ok, tt.want, tt.ok)
		}
	}
}

func TestQueryBitmap_IntersectionCount(t *testing.T) {
	a := New(1000)
	b := New(1000)

	a.AddRange(100, 300) // [100, 300)
	b.AddRange(200, 400) // [200, 400)

	// Intersection is [200, 300) = 100 bits
	if c := a.IntersectionCount(b); c != 100 {
		t.Errorf("IntersectionCount = %d, want 100", c)
	}

	// Original bitmaps should be unchanged
	if c := a.Cardinality(); c != 200 {
		t.Errorf("a.Cardinality = %d after IntersectionCount, want 200", c)
	}
	if c := b.Cardinality(); c != 200 {
		t.Errorf("b.Cardinality = %d after IntersectionCount, want 200", c)
	}
}

func TestQueryBitmap_UnionCount(t *testing.T) {
	a := New(1000)
	b := New(1000)

	a.AddRange(100, 200) // [100, 200)
	b.AddRange(150, 250) // [150, 250)

	// Union is [100, 250) = 150 bits
	if c := a.UnionCount(b); c != 150 {
		t.Errorf("UnionCount = %d, want 150", c)
	}

	// Original bitmaps should be unchanged
	if c := a.Cardinality(); c != 100 {
		t.Errorf("a.Cardinality = %d after UnionCount, want 100", c)
	}
}

func TestQueryBitmapPool(t *testing.T) {
	pool := NewQueryBitmapPool(10000)

	// Get and use
	qb := pool.Get()
	qb.Clear()
	qb.AddRange(0, 1000)

	if c := qb.Cardinality(); c != 1000 {
		t.Errorf("Cardinality = %d, want 1000", c)
	}

	// Return to pool
	pool.Put(qb)

	// Get again - should be cleared
	qb2 := pool.Get()
	qb2.Clear()
	if !qb2.IsEmpty() {
		t.Error("Pooled bitmap should be reusable after Clear")
	}

	pool.Put(qb2)
}

// Benchmarks

func BenchmarkQueryBitmap_AddRange(b *testing.B) {
	pool := NewQueryBitmapPool(100000)
	qb := pool.Get()
	defer pool.Put(qb)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		qb.Clear()
		qb.AddRange(0, 10000)
	}
}

func BenchmarkQueryBitmap_And(b *testing.B) {
	pool := NewQueryBitmapPool(100000)
	a := pool.Get()
	x := pool.Get()
	defer pool.Put(a)
	defer pool.Put(x)

	x.Clear()
	x.AddRange(5000, 15000)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		a.Clear()
		a.AddRange(0, 10000)
		a.And(x)
	}
}

func BenchmarkQueryBitmap_Cardinality(b *testing.B) {
	qb := New(100000)
	qb.AddRange(0, 50000)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		qb.cardinality = -1
		_ = qb.Cardinality()
	}
}

func BenchmarkQueryBitmap_ForEach_10K(b *testing.B) {
	qb := New(100000)
	qb.AddRange(0, 10000)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		count := 0
		qb.ForEach(func(id uint32) bool {
			count++
			return true
		})
	}
}

func BenchmarkQueryBitmap_Sparse(b *testing.B) {
	qb := New(100000)
	rng := rand.New(rand.NewSource(42))
	for i := 0; i < 1000; i++ {
		qb.Add(uint32(rng.Intn(100000)))
	}

	b.Run("Cardinality", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			qb.cardinality = -1
			_ = qb.Cardinality()
		}
	})

	b.Run("ForEach", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			count := 0
			qb.ForEach(func(id uint32) bool {
				count++
				return true
			})
		}
	})
}

func BenchmarkQueryBitmapPool(b *testing.B) {
	pool := NewQueryBitmapPool(100000)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		qb := pool.Get()
		qb.Clear()
		qb.AddRange(0, 10000)
		pool.Put(qb)
	}
}

func BenchmarkQueryBitmap_NextSet(b *testing.B) {
	qb := New(100000)
	qb.AddRange(0, 10000)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		count := 0
		for j, ok := qb.NextSet(0); ok; j, ok = qb.NextSet(j + 1) {
			count++
		}
	}
}

func BenchmarkQueryBitmap_NextSetMany(b *testing.B) {
	qb := New(100000)
	qb.AddRange(0, 10000)
	buffer := make([]uint32, 256)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx := uint32(0)
		for {
			lastIdx, batch := qb.NextSetMany(idx, buffer)
			if len(batch) == 0 {
				break
			}
			idx = lastIdx + 1
		}
	}
}

func BenchmarkQueryBitmap_Rank(b *testing.B) {
	qb := New(100000)
	qb.AddRange(0, 50000)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = qb.Rank(75000)
	}
}

func BenchmarkQueryBitmap_IntersectionCount(b *testing.B) {
	a := New(100000)
	x := New(100000)
	a.AddRange(0, 50000)
	x.AddRange(25000, 75000)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = a.IntersectionCount(x)
	}
}

func TestQueryBitmap_ActiveBlocks(t *testing.T) {
	// Test that activeBlocks is correctly maintained
	qb := New(10000)

	// Add bits in block 0 (bits 0-511)
	qb.Add(100)
	if !qb.isBlockActive(0) {
		t.Error("Block 0 should be active after adding bit 100")
	}

	// Add bits in block 5 (bits 2560-3071)
	qb.Add(2600)
	if !qb.isBlockActive(5) {
		t.Error("Block 5 should be active after adding bit 2600")
	}

	// Block 3 should be inactive
	if qb.isBlockActive(3) {
		t.Error("Block 3 should be inactive")
	}

	// Test Clear only clears active blocks
	qb.Clear()
	if !qb.IsEmpty() {
		t.Error("Should be empty after Clear")
	}

	// Test activeBlocks with And operation
	a := New(10000)
	bm := New(10000)
	a.AddRange(0, 1000)    // Blocks 0, 1
	bm.AddRange(500, 1500) // Blocks 0, 1, 2
	a.And(bm)

	// After AND, only the intersection remains
	if a.Cardinality() != 500 {
		t.Errorf("Cardinality after AND = %d, want 500", a.Cardinality())
	}

	// Test Clone preserves activeBlocks
	c := a.Clone()
	if c.Cardinality() != a.Cardinality() {
		t.Errorf("Clone cardinality mismatch")
	}

	// Test sparse iteration efficiency (shouldn't iterate all blocks)
	sparse := New(100000) // ~196 blocks
	sparse.Add(0)
	sparse.Add(99000) // Block 193

	count := 0
	sparse.ForEach(func(i uint32) bool {
		count++
		return true
	})
	if count != 2 {
		t.Errorf("ForEach count = %d, want 2", count)
	}

	// Test ActiveBlockCount
	if sparse.ActiveBlockCount() != 2 {
		t.Errorf("ActiveBlockCount = %d, want 2", sparse.ActiveBlockCount())
	}

	// Test Density
	dense := New(1000)
	dense.AddRange(0, 500)
	if d := dense.Density(); d < 0.49 || d > 0.51 {
		t.Errorf("Density = %f, want ~0.5", d)
	}
}

func BenchmarkQueryBitmap_ActiveBlocksClear(b *testing.B) {
	// Compare Clear performance for sparse vs dense
	b.Run("Sparse_Clear", func(b *testing.B) {
		qb := New(100000)
		// Add 100 bits scattered across the bitmap
		for i := uint32(0); i < 100; i++ {
			qb.Add(i * 1000)
		}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			qb.Clear()
			for j := uint32(0); j < 100; j++ {
				qb.Add(j * 1000)
			}
		}
	})

	b.Run("Dense_Clear", func(b *testing.B) {
		qb := New(100000)
		qb.AddRange(0, 50000)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			qb.Clear()
			qb.AddRange(0, 50000)
		}
	})
}
