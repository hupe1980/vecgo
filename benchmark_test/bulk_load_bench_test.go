package benchmark_test

import (
	"testing"

	"github.com/hupe1980/vecgo"
)

// BenchmarkBulkLoad compares standard batch insert with deferred bulk insert.
// Run with: go test -bench=BenchmarkBulkLoad ./benchmark_test/... -benchmem
func BenchmarkBulkLoad(b *testing.B) {
	dim := 128
	batchSize := 1000

	b.Run("Standard", func(b *testing.B) {
		dir := b.TempDir()
		// Default config
		e, _ := vecgo.Open(vecgo.Local(dir), vecgo.Create(dim, vecgo.MetricL2))
		defer e.Close()

		vec := make([]float32, dim)
		vectors := make([][]float32, batchSize)
		for i := 0; i < batchSize; i++ {
			vectors[i] = vec
		}

		b.ResetTimer()
		for i := 0; i < b.N; i += batchSize {
			count := batchSize
			if i+count > b.N {
				count = b.N - i
			}
			e.BatchInsert(vectors[:count], nil, nil)
		}
	})

	b.Run("Deferred", func(b *testing.B) {
		dir := b.TempDir()
		// Disable compaction and set large memtable to isolate pure ingest speed
		// In production, you'd want to tune these based on your memory budget
		e, _ := vecgo.Open(vecgo.Local(dir), vecgo.Create(dim, vecgo.MetricL2),
			vecgo.WithCompactionThreshold(1<<30),
			vecgo.WithFlushConfig(vecgo.FlushConfig{MaxMemTableSize: 1 << 30}), // 1GB memtable
			vecgo.WithMemoryLimit(0),                                           // Unlimited memory for benchmark
		)
		defer e.Close()

		vec := make([]float32, dim)
		vectors := make([][]float32, batchSize)
		for i := 0; i < batchSize; i++ {
			vectors[i] = vec
		}

		b.ResetTimer()
		for i := 0; i < b.N; i += batchSize {
			count := batchSize
			if i+count > b.N {
				count = b.N - i
			}
			e.BatchInsertDeferred(vectors[:count], nil, nil)
		}
	})
}
