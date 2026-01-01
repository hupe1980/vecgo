package engine

import (
	"context"
	"testing"

	"github.com/hupe1980/vecgo/codec"
	"github.com/hupe1980/vecgo/core"
	"github.com/hupe1980/vecgo/index"
	"github.com/hupe1980/vecgo/index/flat"
	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/wal"
)

func TestCoordinator_InsertUpdateDelete(t *testing.T) {
	ctx := context.Background()

	idx, err := flat.New(func(o *flat.Options) {
		o.Dimension = 3
		o.DistanceType = index.DistanceTypeSquaredL2
	})
	if err != nil {
		t.Fatalf("flat.New failed: %v", err)
	}

	dataStore := NewMapStore[string]()
	metaStore := metadata.NewUnifiedIndex()

	walLog, err := wal.New(func(o *wal.Options) {
		o.Path = t.TempDir()
		o.DurabilityMode = wal.DurabilityAsync
	})
	if err != nil {
		t.Fatalf("wal.New failed: %v", err)
	}
	defer walLog.Close()

	coord, err := New(idx, dataStore, metaStore, walLog, codec.Default, WithDimension(3))
	if err != nil {
		t.Fatalf("engine.New failed: %v", err)
	}

	id, err := coord.Insert(ctx, []float32{1, 2, 3}, "a", metadata.Metadata{"k": metadata.String("v")})
	if err != nil {
		t.Fatalf("Insert failed: %v", err)
	}

	if got, ok := dataStore.Get(core.LocalID(id)); !ok || got != "a" {
		t.Fatalf("dataStore.Get: got=%q ok=%v", got, ok)
	}
	if got, ok := metaStore.Get(core.LocalID(id)); !ok {
		t.Fatalf("metaStore.Get: ok=%v got=%v", ok, got)
	} else if s, ok := got["k"].AsString(); !ok || s != "v" {
		t.Fatalf("metaStore.Get: k=%v ok=%v", got["k"], ok)
	}

	// Update with nil metadata should NOT overwrite existing metadata.
	if err := coord.Update(ctx, id, []float32{3, 2, 1}, "b", nil); err != nil {
		t.Fatalf("Update failed: %v", err)
	}
	if got, ok := dataStore.Get(core.LocalID(id)); !ok || got != "b" {
		t.Fatalf("dataStore.Get after update: got=%q ok=%v", got, ok)
	}
	if got, ok := metaStore.Get(core.LocalID(id)); !ok {
		t.Fatalf("metaStore.Get after update(nil): ok=%v got=%v", ok, got)
	} else if s, ok := got["k"].AsString(); !ok || s != "v" {
		t.Fatalf("metaStore.Get after update(nil): k=%v ok=%v", got["k"], ok)
	}

	if err := coord.Delete(ctx, id); err != nil {
		t.Fatalf("Delete failed: %v", err)
	}
	if _, ok := dataStore.Get(core.LocalID(id)); ok {
		t.Fatalf("expected data deleted")
	}
}
