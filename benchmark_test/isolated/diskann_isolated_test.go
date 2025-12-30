package isolated

import (
	"context"
	"os"
	"testing"

	"github.com/hupe1980/vecgo/index"
	"github.com/hupe1980/vecgo/index/diskann"
	"github.com/hupe1980/vecgo/testutil"
)

func BenchmarkDiskANNIsolatedInsert(b *testing.B) {
	dim := 128

	b.Run("Direct", func(b *testing.B) {
		tmpDir, err := os.MkdirTemp("", "diskann_bench_insert")
		if err != nil {
			b.Fatal(err)
		}
		defer os.RemoveAll(tmpDir)

		idx, err := diskann.New(dim, index.DistanceTypeSquaredL2, tmpDir, nil)
		if err != nil {
			b.Fatal(err)
		}
		defer idx.Close()

		rng := testutil.NewRNG(42)
		ctx := context.Background()

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			vec := rng.UnitVector(dim)
			if _, err := idx.Insert(ctx, vec); err != nil {
				b.Fatal(err)
			}
		}
	})
}

func BenchmarkDiskANNIsolatedSearch(b *testing.B) {
	dim := 128
	size := 10000
	k := 10

	tmpDir, err := os.MkdirTemp("", "diskann_bench_search")
	if err != nil {
		b.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	idx, err := diskann.New(dim, index.DistanceTypeSquaredL2, tmpDir, nil)
	if err != nil {
		b.Fatal(err)
	}
	defer idx.Close()

	rng := testutil.NewRNG(42)
	ctx := context.Background()
	vectors := make([][]float32, size)

	// Pre-fill
	for i := 0; i < size; i++ {
		vec := rng.UnitVector(dim)
		vectors[i] = vec
		if _, err := idx.Insert(ctx, vec); err != nil {
			b.Fatal(err)
		}
	}

	query := rng.UnitVector(dim)
	groundTruth := testutil.BruteForceSearch(vectors, query, k)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		results, err := idx.KNNSearch(ctx, query, k, nil)
		if err != nil {
			b.Fatal(err)
		}

		if i == 0 {
			recall := testutil.ComputeRecall(groundTruth, results)
			b.ReportMetric(recall, "recall")
		}
	}
}
