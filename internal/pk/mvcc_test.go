package pk

import (
	"sync"
	"testing"

	"github.com/hupe1980/vecgo/model"
)

func TestIndex_Basic(t *testing.T) {
	idx := New()
	id := model.ID(1)
	loc := model.Location{SegmentID: 1, RowID: 100}
	lsn := uint64(10)

	// Get empty
	if _, ok := idx.Get(id, lsn); ok {
		t.Fatal("expected exists=false")
	}

	// Upsert
	idx.Upsert(id, loc, lsn)

	// Get visible
	if got, ok := idx.Get(id, lsn); !ok || got != loc {
		t.Fatalf("expected %v, got %v", loc, got)
	}

	// Get older snapshot
	if _, ok := idx.Get(id, lsn-1); ok {
		t.Fatal("expected exists=false for older snapshot")
	}

	// Upsert newer version
	newLoc := model.Location{SegmentID: 2, RowID: 200}
	idx.Upsert(id, newLoc, lsn+5)

	// Check old snapshot
	if got, ok := idx.Get(id, lsn); !ok || got != loc {
		t.Fatalf("expected old %v, got %v", loc, got)
	}

	// Check new snapshot
	if got, ok := idx.Get(id, lsn+5); !ok || got != newLoc {
		t.Fatalf("expected new %v, got %v", newLoc, got)
	}

	// Delete
	idx.Delete(id, lsn+10)

	// Check deleted
	if _, ok := idx.Get(id, lsn+10); ok {
		t.Fatal("expected exists=false after delete")
	}

	// Check history still exists
	if got, ok := idx.Get(id, lsn+5); !ok || got != newLoc {
		t.Fatalf("expected history %v, got %v", newLoc, got)
	}
}

func TestIndex_ZeroID(t *testing.T) {
	idx := New()
	loc := model.Location{SegmentID: 1, RowID: 100}

	// ID=0 should be rejected
	if _, ok := idx.Upsert(0, loc, 10); ok {
		t.Fatal("expected Upsert to return false for ID=0")
	}

	if _, ok := idx.Get(0, 10); ok {
		t.Fatal("expected Get to return false for ID=0")
	}

	if _, ok := idx.Delete(0, 10); ok {
		t.Fatal("expected Delete to return false for ID=0")
	}
}

func TestIndex_UpsertReturnsOldValue(t *testing.T) {
	idx := New()
	id := model.ID(1)
	loc1 := model.Location{SegmentID: 1, RowID: 100}
	loc2 := model.Location{SegmentID: 2, RowID: 200}

	// First insert should return empty location
	old, existed := idx.Upsert(id, loc1, 10)
	if existed {
		t.Fatal("expected existed=false for new entry")
	}
	if old != (model.Location{}) {
		t.Fatalf("expected empty location, got %v", old)
	}

	// Update at same LSN should return old value
	old, existed = idx.Upsert(id, loc2, 10)
	if !existed {
		t.Fatal("expected existed=true for update")
	}
	if old != loc1 {
		t.Fatalf("expected old=%v, got %v", loc1, old)
	}

	// Update at newer LSN (prepend case)
	loc3 := model.Location{SegmentID: 3, RowID: 300}
	old, existed = idx.Upsert(id, loc3, 20)
	if !existed {
		t.Fatal("expected existed=true for prepend")
	}
	if old != loc2 {
		t.Fatalf("expected old=%v, got %v", loc2, old)
	}
}

func TestIndex_Count(t *testing.T) {
	idx := New()

	if idx.Count() != 0 {
		t.Fatalf("expected count=0, got %d", idx.Count())
	}

	// Insert 3 entries
	idx.Upsert(1, model.Location{SegmentID: 1, RowID: 1}, 10)
	idx.Upsert(2, model.Location{SegmentID: 1, RowID: 2}, 10)
	idx.Upsert(3, model.Location{SegmentID: 1, RowID: 3}, 10)

	if idx.Count() != 3 {
		t.Fatalf("expected count=3, got %d", idx.Count())
	}

	// Delete one
	idx.Delete(2, 20)
	if idx.Count() != 2 {
		t.Fatalf("expected count=2 after delete, got %d", idx.Count())
	}

	// Delete again (no-op)
	idx.Delete(2, 25)
	if idx.Count() != 2 {
		t.Fatalf("expected count=2 after double delete, got %d", idx.Count())
	}
}

func TestIndex_MaxID(t *testing.T) {
	idx := New()

	if idx.MaxID() != 0 {
		t.Fatalf("expected maxID=0 for empty index, got %d", idx.MaxID())
	}

	// Insert scattered IDs
	idx.Upsert(100, model.Location{SegmentID: 1, RowID: 1}, 10)
	if idx.MaxID() != 100 {
		t.Fatalf("expected maxID=100, got %d", idx.MaxID())
	}

	idx.Upsert(50, model.Location{SegmentID: 1, RowID: 2}, 10)
	if idx.MaxID() != 100 {
		t.Fatalf("expected maxID=100 (unchanged), got %d", idx.MaxID())
	}

	// Large ID requiring new page
	idx.Upsert(200000, model.Location{SegmentID: 1, RowID: 3}, 10)
	if idx.MaxID() != 200000 {
		t.Fatalf("expected maxID=200000, got %d", idx.MaxID())
	}
}

func TestIndex_Grow(t *testing.T) {
	idx := New()

	// ID that requires page growth (e.g., > 65536)
	// Page size is 1<<16 = 65536.
	// ID 100,000 should be on page 1.
	id := model.ID(100000)
	loc := model.Location{SegmentID: 1, RowID: 1}

	idx.Upsert(id, loc, 1)

	if got, ok := idx.Get(id, 1); !ok || got != loc {
		t.Fatalf("expected %v, got %v", loc, got)
	}

	// Fill gaps
	id2 := model.ID(100001)
	idx.Upsert(id2, loc, 1)
	if _, ok := idx.Get(id2, 1); !ok {
		t.Fatal("expected id2")
	}
}

func TestIndex_Scan(t *testing.T) {
	idx := New()
	// Insert scattered IDs
	fullMap := make(map[model.ID]model.Location)

	// Page 0
	idx.Upsert(1, model.Location{SegmentID: 1, RowID: 1}, 10)
	fullMap[1] = model.Location{SegmentID: 1, RowID: 1}

	// Page 1
	idHigh := model.ID(70000)
	idx.Upsert(idHigh, model.Location{SegmentID: 1, RowID: 2}, 10)
	fullMap[idHigh] = model.Location{SegmentID: 1, RowID: 2}

	count := 0
	for id, loc := range idx.Scan(10) {
		count++
		expected, ok := fullMap[id]
		if !ok {
			t.Fatalf("unexpected id %d", id)
		}
		if expected != loc {
			t.Fatalf("id %d: mismatch loc", id)
		}
	}

	if count != 2 {
		t.Fatalf("expected 2 items, got %d", count)
	}
}

func TestIndex_ScanSkipsDeleted(t *testing.T) {
	idx := New()

	idx.Upsert(1, model.Location{SegmentID: 1, RowID: 1}, 10)
	idx.Upsert(2, model.Location{SegmentID: 1, RowID: 2}, 10)
	idx.Upsert(3, model.Location{SegmentID: 1, RowID: 3}, 10)
	idx.Delete(2, 20)

	// Scan at LSN 20 should skip ID 2
	count := 0
	for range idx.Scan(20) {
		count++
	}
	if count != 2 {
		t.Fatalf("expected 2 items (skipping deleted), got %d", count)
	}

	// Scan at LSN 15 should include ID 2 (before deletion)
	count = 0
	for range idx.Scan(15) {
		count++
	}
	if count != 3 {
		t.Fatalf("expected 3 items at old LSN, got %d", count)
	}
}

func TestIndex_ConcurrentReadsWrites(t *testing.T) {
	idx := New()
	const numGoroutines = 100
	const opsPerGoroutine = 100

	var wg sync.WaitGroup
	wg.Add(numGoroutines * 2)

	// Writers
	for i := 0; i < numGoroutines; i++ {
		go func(base int) {
			defer wg.Done()
			for j := 0; j < opsPerGoroutine; j++ {
				id := model.ID(base*opsPerGoroutine + j + 1)
				loc := model.Location{SegmentID: model.SegmentID(base), RowID: model.RowID(j)}
				lsn := uint64(base*opsPerGoroutine + j + 1)
				idx.Upsert(id, loc, lsn)
			}
		}(i)
	}

	// Readers
	for i := 0; i < numGoroutines; i++ {
		go func(base int) {
			defer wg.Done()
			for j := 0; j < opsPerGoroutine; j++ {
				id := model.ID(base*opsPerGoroutine + j + 1)
				// Read at high LSN to see all writes
				idx.Get(id, ^uint64(0))
			}
		}(i)
	}

	wg.Wait()

	// Verify count is as expected
	expectedCount := numGoroutines * opsPerGoroutine
	if idx.Count() != expectedCount {
		t.Fatalf("expected count=%d, got %d", expectedCount, idx.Count())
	}
}

func TestIndex_MVCCVersionChain(t *testing.T) {
	idx := New()
	id := model.ID(1)

	// Create version chain: LSN 10 -> LSN 20 -> LSN 30
	idx.Upsert(id, model.Location{SegmentID: 1, RowID: 10}, 10)
	idx.Upsert(id, model.Location{SegmentID: 1, RowID: 20}, 20)
	idx.Upsert(id, model.Location{SegmentID: 1, RowID: 30}, 30)

	// Query at different LSNs
	tests := []struct {
		lsn         uint64
		expectedRow model.RowID
		exists      bool
	}{
		{5, 0, false},   // Before first version
		{10, 10, true},  // Exact match LSN 10
		{15, 10, true},  // Between 10 and 20
		{20, 20, true},  // Exact match LSN 20
		{25, 20, true},  // Between 20 and 30
		{30, 30, true},  // Exact match LSN 30
		{100, 30, true}, // After all versions
	}

	for _, tt := range tests {
		loc, ok := idx.Get(id, tt.lsn)
		if ok != tt.exists {
			t.Errorf("LSN %d: expected exists=%v, got %v", tt.lsn, tt.exists, ok)
		}
		if ok && loc.RowID != tt.expectedRow {
			t.Errorf("LSN %d: expected RowID=%d, got %d", tt.lsn, tt.expectedRow, loc.RowID)
		}
	}
}

func TestIndex_InsertOldVersion(t *testing.T) {
	idx := New()
	id := model.ID(1)

	// Insert at LSN 30 first
	idx.Upsert(id, model.Location{SegmentID: 1, RowID: 30}, 30)

	// Then insert at LSN 10 (older version via COW path)
	idx.Upsert(id, model.Location{SegmentID: 1, RowID: 10}, 10)

	// Query should see both versions
	loc, ok := idx.Get(id, 30)
	if !ok || loc.RowID != 30 {
		t.Fatalf("expected RowID=30, got %v", loc)
	}

	loc, ok = idx.Get(id, 10)
	if !ok || loc.RowID != 10 {
		t.Fatalf("expected RowID=10, got %v", loc)
	}
}
