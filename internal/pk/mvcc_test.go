package pk

import (
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
