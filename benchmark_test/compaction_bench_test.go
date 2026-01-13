package benchmark_test

import (
	"testing"

	"github.com/hupe1980/vecgo"
	"github.com/hupe1980/vecgo/testutil"
)

func BenchmarkCompaction_Pressure(b *testing.B) {
	b.ReportAllocs()

	dir := b.TempDir()

	// Configure engine for high compaction pressure:
	// - Small MemTable (flush often)
	// - Low compaction threshold (compact often)
	eng, err := vecgo.Open(vecgo.Local(dir), vecgo.Create(128, vecgo.MetricL2),
		vecgo.WithFlushConfig(vecgo.FlushConfig{
			MaxMemTableSize: 1 * 1024 * 1024, // 1MB MemTable
		}),
		vecgo.WithCompactionThreshold(2), // Compact when 2 segments exist
	)
	if err != nil {
		b.Fatal(err)
	}
	defer eng.Close()

	rng := testutil.NewRNG(1)
	vec := make([]float32, 128)
	rng.FillUniform(vec)

	b.ResetTimer()

	// We run for b.N iterations.
	// Each iteration inserts a vector.
	// Compaction happens in background.
	for i := 0; i < b.N; i++ {
		if _, err := eng.Insert(vec, nil, nil); err != nil {
			b.Fatal(err)
		}
	}
}
