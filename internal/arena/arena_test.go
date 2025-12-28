package arena

import (
	"fmt"
	"runtime"
	"sync"
	"testing"
	"unsafe"
)

func TestArena_New(t *testing.T) {
	t.Run("default chunk size", func(t *testing.T) {
		a := New(0)
		defer a.Free()

		if a.chunkSize != DefaultChunkSize {
			t.Errorf("expected chunkSize=%d, got %d", DefaultChunkSize, a.chunkSize)
		}
		if a.alignment != DefaultAlignment {
			t.Errorf("expected alignment=%d, got %d", DefaultAlignment, a.alignment)
		}
		if a.current.Load() == nil {
			t.Error("current chunk should not be nil")
		}
	})

	t.Run("custom chunk size", func(t *testing.T) {
		customSize := 4096
		a := New(customSize)
		defer a.Free()

		if a.chunkSize != customSize {
			t.Errorf("expected chunkSize=%d, got %d", customSize, a.chunkSize)
		}
	})
}

func TestArena_AllocBytes(t *testing.T) {
	t.Run("basic allocation", func(t *testing.T) {
		a := New(1024)
		defer a.Free()

		slice := a.AllocBytes(100)
		// Slice may be slightly larger due to alignment
		if len(slice) < 100 {
			t.Errorf("expected length>=100, got %d", len(slice))
		}

		// Verify zero-initialization
		for i, b := range slice {
			if b != 0 {
				t.Errorf("byte at index %d not zero: %d", i, b)
			}
		}
	})

	t.Run("zero size", func(t *testing.T) {
		a := New(1024)
		defer a.Free()

		slice := a.AllocBytes(0)
		if slice != nil {
			t.Error("expected nil for zero size")
		}
	})

	t.Run("alignment", func(t *testing.T) {
		a := New(1024)
		defer a.Free()

		sizes := []int{1, 3, 5, 7, 9, 15, 17}
		for _, size := range sizes {
			slice := a.AllocBytes(size)
			if slice == nil {
				t.Fatalf("allocation failed for size=%d", size)
			}

			ptr := uintptr(unsafe.Pointer(&slice[0]))
			if ptr%uintptr(DefaultAlignment) != 0 {
				t.Errorf("size=%d ptr=%x not aligned", size, ptr)
			}
		}
	})

	t.Run("multiple chunks", func(t *testing.T) {
		chunkSize := 128
		a := New(chunkSize)
		defer a.Free()

		for i := 0; i < 10; i++ {
			slice := a.AllocBytes(64)
			if slice == nil {
				t.Fatalf("allocation %d failed", i)
			}
		}

		stats := a.Stats()
		if stats.ChunksAllocated <= 1 {
			t.Error("expected multiple chunks")
		}
	})
}

func TestArena_AllocUint32Slice(t *testing.T) {
	t.Run("basic allocation", func(t *testing.T) {
		a := New(1024)
		defer a.Free()

		slice := a.AllocUint32Slice(10)
		if len(slice) != 0 {
			t.Errorf("expected length=0, got %d", len(slice))
		}
		if cap(slice) != 10 {
			t.Errorf("expected capacity=10, got %d", cap(slice))
		}

		slice = append(slice, 1, 2, 3, 4, 5)
		if len(slice) != 5 {
			t.Errorf("expected length=5 after append, got %d", len(slice))
		}
	})

	t.Run("zero capacity", func(t *testing.T) {
		a := New(1024)
		defer a.Free()

		slice := a.AllocUint32Slice(0)
		if slice != nil {
			t.Error("expected nil for zero capacity")
		}
	})

	t.Run("multiple allocations", func(t *testing.T) {
		a := New(1024)
		defer a.Free()

		slices := make([][]uint32, 10)
		for i := range slices {
			slices[i] = a.AllocUint32Slice(5)
			slices[i] = append(slices[i], uint32(i))
		}

		for i, slice := range slices {
			if len(slice) != 1 || slice[0] != uint32(i) {
				t.Errorf("slice %d has wrong value", i)
			}
		}
	})
}

func TestArena_AllocFloat32Slice(t *testing.T) {
	t.Run("basic allocation", func(t *testing.T) {
		a := New(1024)
		defer a.Free()

		slice := a.AllocFloat32Slice(10)
		if len(slice) != 0 {
			t.Errorf("expected length=0, got %d", len(slice))
		}
		if cap(slice) != 10 {
			t.Errorf("expected capacity=10, got %d", cap(slice))
		}

		slice = append(slice, 1.5, 2.5, 3.5)
		if len(slice) != 3 {
			t.Errorf("expected length=3 after append, got %d", len(slice))
		}
	})
}

func TestArena_Stats(t *testing.T) {
	t.Run("initial stats", func(t *testing.T) {
		a := New(1024)
		defer a.Free()

		stats := a.Stats()
		if stats.ChunksAllocated != 1 {
			t.Errorf("expected ChunksAllocated=1, got %d", stats.ChunksAllocated)
		}
		if stats.BytesReserved != 1024 {
			t.Errorf("expected BytesReserved=1024, got %d", stats.BytesReserved)
		}
		if stats.BytesUsed != 0 {
			t.Errorf("expected BytesUsed=0, got %d", stats.BytesUsed)
		}
	})

	t.Run("after allocations", func(t *testing.T) {
		a := New(1024)
		defer a.Free()

		a.AllocBytes(100)
		a.AllocBytes(200)
		a.AllocUint32Slice(10)

		stats := a.Stats()
		if stats.BytesUsed != 340 {
			t.Errorf("expected BytesUsed=340, got %d", stats.BytesUsed)
		}
		if stats.TotalAllocs != 3 {
			t.Errorf("expected TotalAllocs=3, got %d", stats.TotalAllocs)
		}
	})
}

func TestArena_Free(t *testing.T) {
	a := New(1024)

	a.AllocBytes(100)
	a.AllocBytes(200)

	statsBefore := a.Stats()
	if statsBefore.BytesUsed == 0 {
		t.Error("expected BytesUsed > 0 before free")
	}

	a.Free()

	statsAfter := a.Stats()
	if statsAfter.ActiveChunks != 0 {
		t.Errorf("expected ActiveChunks=0 after free, got %d", statsAfter.ActiveChunks)
	}
}

func TestArena_Reset(t *testing.T) {
	t.Run("basic reset", func(t *testing.T) {
		a := New(1024)
		defer a.Free()

		a.AllocBytes(100)
		a.AllocBytes(200)

		statsBefore := a.Stats()
		allocsBefore := statsBefore.TotalAllocs

		a.Reset()

		statsAfter := a.Stats()
		if statsAfter.BytesUsed != 0 {
			t.Errorf("expected BytesUsed=0 after reset, got %d", statsAfter.BytesUsed)
		}
		if statsAfter.TotalAllocs != allocsBefore {
			t.Error("alloc count should be preserved after reset")
		}
		if statsAfter.ActiveChunks != 1 {
			t.Errorf("expected ActiveChunks=1 after reset, got %d", statsAfter.ActiveChunks)
		}
	})

	t.Run("reset after multiple chunks", func(t *testing.T) {
		a := New(256)
		defer a.Free()

		for i := 0; i < 10; i++ {
			a.AllocBytes(128)
		}

		statsBefore := a.Stats()
		if statsBefore.ActiveChunks <= 1 {
			t.Error("expected multiple chunks before reset")
		}

		a.Reset()

		statsAfter := a.Stats()
		if statsAfter.ActiveChunks != 1 {
			t.Errorf("expected ActiveChunks=1 after reset, got %d", statsAfter.ActiveChunks)
		}
	})
}

func TestArena_Usage(t *testing.T) {
	a := New(1000)
	defer a.Free()

	usage := a.Usage()
	if usage > 1.0 {
		t.Errorf("initial usage should be near 0%%, got %.2f%%", usage)
	}

	a.AllocBytes(500)
	usage = a.Usage()
	if usage < 45.0 || usage > 55.0 {
		t.Errorf("expected usage around 50%%, got %.2f%%", usage)
	}
}

func TestArena_Concurrent(t *testing.T) {
	a := New(DefaultChunkSize)
	defer a.Free()

	const goroutines = 100
	const allocsPerGoroutine = 100

	var wg sync.WaitGroup
	wg.Add(goroutines)

	for i := 0; i < goroutines; i++ {
		go func() {
			defer wg.Done()
			for j := 0; j < allocsPerGoroutine; j++ {
				slice := a.AllocUint32Slice(16)
				slice = append(slice, uint32(j))
				runtime.KeepAlive(slice)
			}
		}()
	}

	wg.Wait()

	stats := a.Stats()
	if stats.TotalAllocs != goroutines*allocsPerGoroutine {
		t.Errorf("expected TotalAllocs=%d, got %d",
			goroutines*allocsPerGoroutine, stats.TotalAllocs)
	}
}

func BenchmarkArena_AllocBytes(b *testing.B) {
	sizes := []int{16, 64, 256, 1024}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("size=%d", size), func(b *testing.B) {
			a := New(DefaultChunkSize)
			defer a.Free()

			b.ResetTimer()
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				_ = a.AllocBytes(size)
			}
		})
	}
}

func BenchmarkArena_AllocUint32Slice(b *testing.B) {
	capacities := []int{8, 16, 32, 64}

	for _, cap := range capacities {
		b.Run(fmt.Sprintf("cap=%d", cap), func(b *testing.B) {
			a := New(DefaultChunkSize)
			defer a.Free()

			b.ResetTimer()
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				_ = a.AllocUint32Slice(cap)
			}
		})
	}
}

func BenchmarkArena_vs_Make(b *testing.B) {
	b.Run("arena", func(b *testing.B) {
		a := New(DefaultChunkSize)
		defer a.Free()

		b.ResetTimer()
		b.ReportAllocs()

		for i := 0; i < b.N; i++ {
			_ = a.AllocUint32Slice(16)
		}
	})

	b.Run("make", func(b *testing.B) {
		b.ResetTimer()
		b.ReportAllocs()

		for i := 0; i < b.N; i++ {
			_ = make([]uint32, 0, 16)
		}
	})
}

func BenchmarkArena_ConcurrentAllocs(b *testing.B) {
	a := New(DefaultChunkSize)
	defer a.Free()

	b.ResetTimer()
	b.ReportAllocs()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			_ = a.AllocUint32Slice(16)
		}
	})
}
