package wal

import (
	"testing"
	"time"

	"github.com/hupe1980/vecgo/core"
	"github.com/hupe1980/vecgo/metadata"
)

// BenchmarkDurabilityModes compares write latency across different durability modes.
func BenchmarkDurabilityAsync(b *testing.B) {
	benchmarkDurability(b, DurabilityAsync)
}

func BenchmarkDurabilityGroupCommit(b *testing.B) {
	benchmarkDurability(b, DurabilityGroupCommit)
}

func BenchmarkDurabilitySync(b *testing.B) {
	benchmarkDurability(b, DurabilitySync)
}

func benchmarkDurability(b *testing.B, mode DurabilityMode) {
	tmpDir := b.TempDir()

	w, err := New(func(o *Options) {
		o.Path = tmpDir
		o.DurabilityMode = mode
		o.GroupCommitInterval = 10 * time.Millisecond
		o.GroupCommitMaxOps = 100
		o.Compress = false
	})
	if err != nil {
		b.Fatal(err)
	}
	defer w.Close()

	vec := []float32{1.0, 2.0, 3.0}
	data := []byte("test data")
	meta := metadata.Metadata{"key": metadata.String("value")}

	b.ResetTimer()
	for i := 0; b.Loop(); i++ {
		id := core.LocalID(i)
		if err := w.LogPrepareInsert(id, vec, data, meta); err != nil {
			b.Fatal(err)
		}
		if err := w.LogCommitInsert(id); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkGroupCommitBatchSizes measures throughput with different batch sizes.
func BenchmarkGroupCommitBatchSize10(b *testing.B) {
	benchmarkGroupCommitBatchSize(b, 10)
}

func BenchmarkGroupCommitBatchSize50(b *testing.B) {
	benchmarkGroupCommitBatchSize(b, 50)
}

func BenchmarkGroupCommitBatchSize100(b *testing.B) {
	benchmarkGroupCommitBatchSize(b, 100)
}

func BenchmarkGroupCommitBatchSize500(b *testing.B) {
	benchmarkGroupCommitBatchSize(b, 500)
}

func benchmarkGroupCommitBatchSize(b *testing.B, batchSize int) {
	tmpDir := b.TempDir()

	w, err := New(func(o *Options) {
		o.Path = tmpDir
		o.DurabilityMode = DurabilityGroupCommit
		o.GroupCommitInterval = 100 * time.Millisecond // Long interval to test batch size
		o.GroupCommitMaxOps = batchSize
		o.Compress = false
	})
	if err != nil {
		b.Fatal(err)
	}
	defer w.Close()

	vec := []float32{1.0, 2.0, 3.0}
	data := []byte("test data")
	meta := metadata.Metadata{"key": metadata.String("value")}

	b.ResetTimer()
	for i := 0; b.Loop(); i++ {
		id := core.LocalID(i)
		if err := w.LogPrepareInsert(id, vec, data, meta); err != nil {
			b.Fatal(err)
		}
		if err := w.LogCommitInsert(id); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkGroupCommitIntervals measures impact of different fsync intervals.
func BenchmarkGroupCommitInterval1ms(b *testing.B) {
	benchmarkGroupCommitInterval(b, 1*time.Millisecond)
}

func BenchmarkGroupCommitInterval10ms(b *testing.B) {
	benchmarkGroupCommitInterval(b, 10*time.Millisecond)
}

func BenchmarkGroupCommitInterval50ms(b *testing.B) {
	benchmarkGroupCommitInterval(b, 50*time.Millisecond)
}

func BenchmarkGroupCommitInterval100ms(b *testing.B) {
	benchmarkGroupCommitInterval(b, 100*time.Millisecond)
}

func benchmarkGroupCommitInterval(b *testing.B, interval time.Duration) {
	tmpDir := b.TempDir()

	w, err := New(func(o *Options) {
		o.Path = tmpDir
		o.DurabilityMode = DurabilityGroupCommit
		o.GroupCommitInterval = interval
		o.GroupCommitMaxOps = 1000 // High threshold so interval is the trigger
		o.Compress = false
	})
	if err != nil {
		b.Fatal(err)
	}
	defer w.Close()

	vec := []float32{1.0, 2.0, 3.0}
	data := []byte("test data")
	meta := metadata.Metadata{"key": metadata.String("value")}

	b.ResetTimer()
	for i := 0; b.Loop(); i++ {
		id := core.LocalID(i)
		if err := w.LogPrepareInsert(id, vec, data, meta); err != nil {
			b.Fatal(err)
		}
		if err := w.LogCommitInsert(id); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkParallelWrites measures concurrent write throughput.
func BenchmarkParallelWritesAsync(b *testing.B) {
	benchmarkParallelWrites(b, DurabilityAsync)
}

func BenchmarkParallelWritesGroupCommit(b *testing.B) {
	benchmarkParallelWrites(b, DurabilityGroupCommit)
}

func BenchmarkParallelWritesSync(b *testing.B) {
	benchmarkParallelWrites(b, DurabilitySync)
}

func benchmarkParallelWrites(b *testing.B, mode DurabilityMode) {
	tmpDir := b.TempDir()

	w, err := New(func(o *Options) {
		o.Path = tmpDir
		o.DurabilityMode = mode
		o.GroupCommitInterval = 10 * time.Millisecond
		o.GroupCommitMaxOps = 100
		o.Compress = false
	})
	if err != nil {
		b.Fatal(err)
	}
	defer w.Close()

	vec := []float32{1.0, 2.0, 3.0}
	data := []byte("test data")
	meta := metadata.Metadata{"key": metadata.String("value")}

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		var i uint64
		for pb.Next() {
			i++
			id := core.LocalID(i)
			if err := w.LogPrepareInsert(id, vec, data, meta); err != nil {
				b.Fatal(err)
			}
			if err := w.LogCommitInsert(id); err != nil {
				b.Fatal(err)
			}
		}
	})
}

// BenchmarkRecoveryWithGroupCommit measures recovery time with group commit.
func BenchmarkRecoveryWithGroupCommit(b *testing.B) {
	tmpDir := b.TempDir()

	// Pre-populate WAL with 10k entries
	{
		w, err := New(func(o *Options) {
			o.Path = tmpDir
			o.DurabilityMode = DurabilityGroupCommit
			o.Compress = false
		})
		if err != nil {
			b.Fatal(err)
		}

		vec := []float32{1.0, 2.0, 3.0}
		data := []byte("test data")
		meta := metadata.Metadata{"key": metadata.String("value")}

		for i := 0; i < 10000; i++ {
			id := core.LocalID(i)
			_ = w.LogPrepareInsert(id, vec, data, meta)
			_ = w.LogCommitInsert(id)
		}
		w.Close()
	}

	b.ResetTimer()
	for b.Loop() {
		// Measure replay time
		w2, err := New(func(o *Options) {
			o.Path = tmpDir
			o.DurabilityMode = DurabilityGroupCommit
		})
		if err != nil {
			b.Fatal(err)
		}
		applyFunc := func(e Entry) error { return nil }
		_ = w2.ReplayCommitted(applyFunc)
		w2.Close()
	}
}
