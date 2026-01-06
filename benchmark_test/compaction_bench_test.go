package benchmark_test

import (
	"testing"

	"github.com/hupe1980/vecgo"
	"github.com/hupe1980/vecgo/engine"
	"github.com/hupe1980/vecgo/model"
	"github.com/hupe1980/vecgo/testutil"
)

func BenchmarkCompaction_Pressure(b *testing.B) {
	b.ReportAllocs()

	dir := b.TempDir()

	// Configure engine for high compaction pressure:
	// - Small MemTable (flush often)
	// - Low compaction threshold (compact often)
	eng, err := vecgo.Open(dir, 128, vecgo.MetricL2,
		engine.WithFlushConfig(engine.FlushConfig{
			MaxMemTableSize: 1 * 1024 * 1024, // 1MB MemTable
		}),
		engine.WithCompactionThreshold(2), // Compact when 2 segments exist
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
		pk := model.PKUint64(uint64(i))
		if err := eng.Insert(pk, vec, nil, nil); err != nil {
			b.Fatal(err)
		}
	}
}
