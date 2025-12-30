package engine

import (
	"context"
	"testing"

	"github.com/hupe1980/vecgo/codec"
	"github.com/hupe1980/vecgo/index"
	"github.com/hupe1980/vecgo/index/flat"
	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/wal"
)

func BenchmarkCoordinatorInsert(b *testing.B) {
	idx, err := flat.New(func(o *flat.Options) {
		o.Dimension = 3
		o.DistanceType = index.DistanceTypeSquaredL2
	})
	if err != nil {
		b.Fatal(err)
	}
	dataStore := NewMapStore[string]()
	metaStore := metadata.NewUnifiedIndex()

	walLog, err := wal.New(func(o *wal.Options) {
		o.Path = b.TempDir()
		o.DurabilityMode = wal.DurabilityAsync
	})
	if err != nil {
		b.Fatal(err)
	}
	defer walLog.Close()

	coord, err := New(idx, dataStore, metaStore, walLog, codec.Default, WithDimension(3))
	if err != nil {
		b.Fatal(err)
	}

	ctx := context.Background()
	b.ResetTimer()
	for b.Loop() {
		_, _ = coord.Insert(ctx, []float32{1, 2, 3}, "x", nil)
	}
}
