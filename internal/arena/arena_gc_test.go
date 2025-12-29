package arena

import (
	"runtime"
	"testing"
)

// BenchmarkArenaReuse simulates real HNSW usage: create arena once, allocate many times, reset between builds.
// This is the CORRECT way to benchmark arena - mimics actual usage pattern.
func BenchmarkArenaReuse(b *testing.B) {
	a := New(DefaultChunkSize)
	defer a.Free()

	runtime.GC()
	var m1 runtime.MemStats
	runtime.ReadMemStats(&m1)

	b.ResetTimer()
	b.ReportAllocs()
	for b.Loop() {
		for j := 0; j < 1000; j++ {
			_ = a.AllocUint32Slice(16)
		}
		a.Reset()
	}

	b.StopTimer()
	runtime.GC()
	var m2 runtime.MemStats
	runtime.ReadMemStats(&m2)
	b.ReportMetric(float64(m2.NumGC-m1.NumGC), "gcs")
	b.ReportMetric(float64(m2.PauseTotalNs-m1.PauseTotalNs)/1e6, "gc_pause_ms")
}

// BenchmarkStandardHeap forces heap allocation (prevents compiler optimization).
// Fair comparison to arena - same allocation pattern, but GC-managed.
func BenchmarkStandardHeap(b *testing.B) {
	runtime.GC()
	var m1 runtime.MemStats
	runtime.ReadMemStats(&m1)

	b.ResetTimer()
	b.ReportAllocs()
	for b.Loop() {
		for j := 0; j < 1000; j++ {
			s := make([]uint32, 16)
			runtime.KeepAlive(s) // Force heap allocation
		}
	}

	b.StopTimer()
	runtime.GC()
	var m2 runtime.MemStats
	runtime.ReadMemStats(&m2)
	b.ReportMetric(float64(m2.NumGC-m1.NumGC), "gcs")
	b.ReportMetric(float64(m2.PauseTotalNs-m1.PauseTotalNs)/1e6, "gc_pause_ms")
}

// BenchmarkArenaOneShotBuild simulates building one HNSW index from scratch.
// Creates arena, allocates, frees once. More realistic than repeated alloc/free.
func BenchmarkArenaOneShotBuild(b *testing.B) {
	runtime.GC()
	var m1 runtime.MemStats
	runtime.ReadMemStats(&m1)

	b.ResetTimer()
	b.ReportAllocs()
	for b.Loop() {
		a := New(DefaultChunkSize)
		for j := 0; j < 1000; j++ {
			_ = a.AllocUint32Slice(16)
		}
		a.Free()
	}

	b.StopTimer()
	runtime.GC()
	var m2 runtime.MemStats
	runtime.ReadMemStats(&m2)
	b.ReportMetric(float64(m2.NumGC-m1.NumGC), "gcs")
	b.ReportMetric(float64(m2.PauseTotalNs-m1.PauseTotalNs)/1e6, "gc_pause_ms")
}

// BenchmarkStandardOneShotBuild simulates standard allocation for one build cycle.
func BenchmarkStandardOneShotBuild(b *testing.B) {
	runtime.GC()
	var m1 runtime.MemStats
	runtime.ReadMemStats(&m1)

	b.ResetTimer()
	b.ReportAllocs()
	for b.Loop() {
		for j := 0; j < 1000; j++ {
			s := make([]uint32, 16)
			runtime.KeepAlive(s)
		}
		// Force GC between iterations (simulates index lifetime)
		runtime.GC()
	}

	b.StopTimer()
	runtime.GC()
	var m2 runtime.MemStats
	runtime.ReadMemStats(&m2)
	b.ReportMetric(float64(m2.NumGC-m1.NumGC), "gcs")
	b.ReportMetric(float64(m2.PauseTotalNs-m1.PauseTotalNs)/1e6, "gc_pause_ms")
}
