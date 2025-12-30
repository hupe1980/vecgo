package hnsw

import (
	"context"
	"testing"

	"github.com/hupe1980/vecgo/index"
	"github.com/hupe1980/vecgo/testutil"
)

func BenchmarkKNNSearchAlloc(b *testing.B) {
	dim := 128
	count := 10000
	
	// Create index
	idx, err := New(func(o *Options) {
		o.Dimension = dim
		o.M = 16
		o.EF = 100
		o.DistanceType = index.DistanceTypeSquaredL2
	})
	if err != nil {
		b.Fatal(err)
	}

	// Insert vectors
	ctx := context.Background()
	rng := testutil.NewRNG(0)
	vectors := rng.UniformVectors(count, dim)
	
	for _, vec := range vectors {
		if _, err := idx.Insert(ctx, vec); err != nil {
			b.Fatal(err)
		}
	}

	query := rng.UniformVectors(1, dim)[0]

	b.ResetTimer()
	b.ReportAllocs()

	b.Run("KNNSearch", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, err := idx.KNNSearch(ctx, query, 10, nil)
			if err != nil {
				b.Fatal(err)
			}
		}
	})

	b.Run("KNNSearchStream", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			iter := idx.KNNSearchStream(ctx, query, 10, nil)
			for _, err := range iter {
				if err != nil {
					b.Fatal(err)
				}
			}
		}
	})
}
