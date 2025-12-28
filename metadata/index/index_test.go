package index

import (
	"testing"

	"github.com/hupe1980/vecgo/metadata"
)

func TestInvertedIndex_CompileEqAndIn(t *testing.T) {
	ix := New()
	ix.Add(1, metadata.Document{"category": metadata.String("tech"), "status": metadata.String("active")})
	ix.Add(2, metadata.Document{"category": metadata.String("sports"), "status": metadata.String("active")})
	ix.Add(3, metadata.Document{"category": metadata.String("tech"), "status": metadata.String("inactive")})

	fn, ok := ix.Compile(metadata.NewFilterSet(
		metadata.Filter{Key: "category", Operator: metadata.OpEqual, Value: metadata.String("tech")},
		metadata.Filter{Key: "status", Operator: metadata.OpIn, Value: metadata.Array([]metadata.Value{metadata.String("active")})},
	))
	if !ok {
		t.Fatalf("expected compile ok")
	}
	if !fn(1) {
		t.Fatalf("expected id=1 to match")
	}
	if fn(2) {
		t.Fatalf("expected id=2 to not match")
	}
	if fn(3) {
		t.Fatalf("expected id=3 to not match")
	}
}

func TestInvertedIndex_UpdateAndRemove(t *testing.T) {
	ix := New()
	oldDoc := metadata.Document{"category": metadata.String("tech")}
	ix.Add(1, oldDoc)

	fn, ok := ix.Compile(metadata.NewFilterSet(
		metadata.Filter{Key: "category", Operator: metadata.OpEqual, Value: metadata.String("tech")},
	))
	if !ok || !fn(1) {
		t.Fatalf("expected id=1 to match")
	}

	newDoc := metadata.Document{"category": metadata.String("sports")}
	ix.Update(1, oldDoc, newDoc)

	fnTech, ok := ix.Compile(metadata.NewFilterSet(
		metadata.Filter{Key: "category", Operator: metadata.OpEqual, Value: metadata.String("tech")},
	))
	if !ok {
		t.Fatalf("expected compile ok")
	}
	if fnTech(1) {
		t.Fatalf("expected id=1 to not match tech after update")
	}

	fnSports, ok := ix.Compile(metadata.NewFilterSet(
		metadata.Filter{Key: "category", Operator: metadata.OpEqual, Value: metadata.String("sports")},
	))
	if !ok || !fnSports(1) {
		t.Fatalf("expected id=1 to match sports after update")
	}

	ix.Remove(1, newDoc)
	fnSports2, ok := ix.Compile(metadata.NewFilterSet(
		metadata.Filter{Key: "category", Operator: metadata.OpEqual, Value: metadata.String("sports")},
	))
	if !ok {
		t.Fatalf("expected compile ok")
	}
	if fnSports2(1) {
		t.Fatalf("expected id=1 to not match after remove")
	}
}
