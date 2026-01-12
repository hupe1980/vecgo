package benchmark_test

import (
	"testing"

	"github.com/hupe1980/vecgo"
)

// BenchmarkInsertMicro is a micro-benchmark isolating pure insert path.
// Uses huge memtable to prevent any flush during benchmark.
func BenchmarkInsertMicro(b *testing.B) {
	dim := 128
	batchSize := 1000

	b.Run("Deferred", func(b *testing.B) {
		dir := b.TempDir()
		// Use very large memtable (16GB) to prevent any flush
		e, err := vecgo.Open(dir, vecgo.Create(dim, vecgo.MetricL2),
			vecgo.WithCompactionThreshold(1<<40),
			vecgo.WithFlushConfig(vecgo.FlushConfig{MaxMemTableSize: 1 << 34}), // 16GB memtable
			vecgo.WithMemoryLimit(0),
		)
		if err != nil {
			b.Fatal(err)
		}
		defer e.Close()

		vec := make([]float32, dim)
		vectors := make([][]float32, batchSize)
		for i := range batchSize {
			vectors[i] = vec
		}

		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i += batchSize {
			count := batchSize
			if i+count > b.N {
				count = b.N - i
			}
			e.BatchInsertDeferred(vectors[:count], nil, nil)
		}
	})
}
