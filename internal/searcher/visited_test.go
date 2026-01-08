package searcher

import (
	"math/rand"
	"testing"
	"time"

	"github.com/hupe1980/vecgo/model"
)

func TestVisitedSet_Basic(t *testing.T) {
	v := NewVisitedSet(64)

	ids := []model.RowID{0, 1, 63, 64, 100, 1000}

	// Initial state
	for _, id := range ids {
		if v.Visited(id) {
			t.Errorf("ID %d should not be visited yet", id)
		}
	}

	// Visit
	for _, id := range ids {
		v.Visit(id)
	}

	// Check
	for _, id := range ids {
		if !v.Visited(id) {
			t.Errorf("ID %d should be visited", id)
		}
	}

	// Ensure unvisited ones are still unvisited
	if v.Visited(2) {
		t.Error("ID 2 should not be visited")
	}

	// Visit again (idempotent)
	v.Visit(0)
	if !v.Visited(0) {
		t.Error("ID 0 should still be visited")
	}
}

func TestVisitedSet_Reset(t *testing.T) {
	v := NewVisitedSet(10)

	v.Visit(5)
	v.Visit(128) // Triggers grow

	if !v.Visited(5) || !v.Visited(128) {
		t.Fatal("Visit failed")
	}

	v.Reset()

	if v.Visited(5) {
		t.Error("ID 5 should be cleared after Reset")
	}
	if v.Visited(128) {
		t.Error("ID 128 should be cleared after Reset")
	}

	// Check dirty list is empty (internal implementation detail, but good for whitebox testing if access allowed,
	// here we just rely on behavior)
}

func TestVisitedSet_EnsureCapacity(t *testing.T) {
	v := NewVisitedSet(10)

	// Pre-grow
	v.EnsureCapacity(1000)

	// Should not panic or fail
	v.Visit(999)
	if !v.Visited(999) {
		t.Error("Visit 999 failed")
	}
}

func TestVisitedSet_Fuzz(t *testing.T) {
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	v := NewVisitedSet(10)

	// Visit 100 random IDs
	visited := make(map[model.RowID]bool)
	for i := 0; i < 100; i++ {
		id := model.RowID(rng.Intn(5000))
		v.Visit(id)
		visited[id] = true
	}

	// Verify
	for i := 0; i < 5000; i++ {
		id := model.RowID(i)
		want := visited[id]
		got := v.Visited(id)
		if want != got {
			t.Fatalf("ID %d: want %v, got %v", id, want, got)
		}
	}

	v.Reset()

	for i := 0; i < 5000; i++ {
		if v.Visited(model.RowID(i)) {
			t.Fatalf("ID %d should be cleared", i)
		}
	}
}
