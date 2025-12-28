// Package arena provides a custom memory arena allocator optimized for HNSW graph construction.
//
// # Concurrency Model
//
// Arena supports concurrent allocations (AllocBytes, AllocUint32Slice, AllocFloat32Slice)
// but does NOT support concurrent Reset/Free operations. The typical usage pattern is:
//   - Create arena during HNSW index construction
//   - Allocate from multiple goroutines during parallel inserts (SAFE)
//   - Call Close() once when index is destroyed (NOT concurrent with allocations)
//
// # Memory Management
//
// Arena allocates memory in large chunks (1 MiB default) and uses lock-free CAS for
// fast allocation. Memory is not returned to the OS until Free() is called, making it
// ideal for bulk allocation patterns like HNSW graph construction.
//
// # HNSW Usage Recommendations
//
//  1. Create one arena per HNSW index (not per-operation)
//  2. Store arena reference in HNSW struct
//  3. Use AllocUint32Slice for adjacency lists (excellent cache locality)
//  4. Call Free() in HNSW's Close() method for cleanup
//  5. Do NOT call Reset/Free while allocations are happening
package arena

import (
	"fmt"
	"sync"
	"sync/atomic"
	"unsafe"
)

const (
	DefaultChunkSize = 1024 * 1024
	DefaultAlignment = 8
)

// Stats tracks arena memory usage metrics.
//
// Note on semantics:
//   - BytesReserved: total memory reserved from OS (via make([]byte))
//   - BytesUsed: actual bytes requested by allocations (before alignment)
//   - BytesWasted: padding added for alignment
//   - ActiveChunks: number of chunks currently held
//   - TotalAllocs: cumulative allocation count
type Stats struct {
	ChunksAllocated uint64 // Historical: total chunks ever created
	BytesReserved   uint64 // Current: total memory reserved (replaces BytesAllocated)
	BytesUsed       uint64 // Current: actual bytes used
	BytesWasted     uint64 // Current: alignment padding
	ActiveChunks    uint64 // Current: active chunk count
	TotalAllocs     uint64 // Historical: total allocations
}

type chunk struct {
	data   []byte
	offset atomic.Int64 // MUST be atomic - accessed concurrently without locks
}

type Arena struct {
	chunkSize int
	alignment int
	chunks    []*chunk
	current   atomic.Pointer[chunk]
	mu        sync.Mutex
	stats     Stats
}

func New(chunkSize int) *Arena {
	if chunkSize <= 0 {
		chunkSize = DefaultChunkSize
	}

	a := &Arena{
		chunkSize: chunkSize,
		alignment: DefaultAlignment,
		chunks:    make([]*chunk, 0, 16),
	}

	a.allocateChunk()
	return a
}

func (a *Arena) allocateChunk() {
	newChunk := &chunk{
		data: make([]byte, a.chunkSize),
		// offset is atomic.Int64 - zero value is correct
	}

	a.mu.Lock()
	a.chunks = append(a.chunks, newChunk)
	a.stats.ChunksAllocated++
	a.stats.BytesReserved += uint64(a.chunkSize)
	a.stats.ActiveChunks++
	a.mu.Unlock()

	a.current.Store(newChunk)
}

func (a *Arena) align(size int) int {
	mask := a.alignment - 1
	return (size + mask) & ^mask
}

func (a *Arena) AllocBytes(size int) []byte {
	if size <= 0 {
		return nil
	}

	alignedSize := a.align(size)

	for {
		curr := a.current.Load()
		if curr == nil {
			a.allocateChunk()
			continue
		}

		oldOffset := curr.offset.Load()
		newOffset := oldOffset + int64(alignedSize)

		if newOffset <= int64(len(curr.data)) {
			// Fast path: CAS allocation from current chunk (lock-free)
			if curr.offset.CompareAndSwap(oldOffset, newOffset) {
				// Update stats under lock (stats are not on hot path)
				a.mu.Lock()
				a.stats.BytesUsed += uint64(size)
				a.stats.BytesWasted += uint64(alignedSize - size)
				a.stats.TotalAllocs++
				a.mu.Unlock()

				return curr.data[oldOffset:newOffset:newOffset]
			}
			// CAS failed - another goroutine won, retry
			continue
		}

		// Current chunk full - need new chunk (slow path)
		// Use CAS on current to ensure only one goroutine allocates
		if a.current.CompareAndSwap(curr, nil) {
			a.allocateChunk()
		}
		// Either we allocated or another goroutine did - retry from top
	}
}

func (a *Arena) AllocUint32Slice(capacity int) []uint32 {
	if capacity <= 0 {
		return nil
	}

	size := capacity * int(unsafe.Sizeof(uint32(0)))
	bytes := a.AllocBytes(size)

	return unsafe.Slice((*uint32)(unsafe.Pointer(&bytes[0])), capacity)[:0:capacity]
}

func (a *Arena) AllocFloat32Slice(capacity int) []float32 {
	if capacity <= 0 {
		return nil
	}

	size := capacity * int(unsafe.Sizeof(float32(0)))
	bytes := a.AllocBytes(size)

	return unsafe.Slice((*float32)(unsafe.Pointer(&bytes[0])), capacity)[:0:capacity]
}

func (a *Arena) Stats() Stats {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.stats
}

// Free releases all arena memory back to the garbage collector.
//
// IMPORTANT:
//  1. Do NOT call Free concurrently with allocations
//  2. Memory is eligible for GC but not immediately returned to OS
//  3. All slices allocated from this arena become invalid after Free
//  4. Typical usage: defer arena.Free() or call in Close() method
//
// After Free(), the arena cannot be reused. Create a new arena instead.
func (a *Arena) Free() {
	a.mu.Lock()
	defer a.mu.Unlock()

	// Clear chunk references to enable GC
	for i := range a.chunks {
		a.chunks[i] = nil
	}
	a.chunks = nil
	a.current.Store(nil)

	// Update stats to reflect freed state
	a.stats.ActiveChunks = 0
	a.stats.BytesReserved = 0
	a.stats.BytesUsed = 0
	a.stats.BytesWasted = 0
}

// Reset clears all allocations and releases extra chunks, keeping only the first chunk.
//
// IMPORTANT:
//  1. Do NOT call Reset concurrently with allocations
//  2. All slices allocated before Reset become invalid
//  3. Useful for reusing arena across multiple independent build phases
//
// Reset is more efficient than Free + New when you need to reuse the arena.
func (a *Arena) Reset() {
	a.mu.Lock()
	defer a.mu.Unlock()

	// Reset all chunk offsets to zero
	for _, c := range a.chunks {
		c.offset.Store(0)
	}

	if len(a.chunks) > 0 {
		firstChunk := a.chunks[0]
		// Free extra chunks (keep first for reuse)
		for i := 1; i < len(a.chunks); i++ {
			a.chunks[i] = nil
		}
		a.chunks = a.chunks[:1]
		a.current.Store(firstChunk)

		// Update stats - only first chunk remains
		a.stats.ActiveChunks = 1
		a.stats.BytesReserved = uint64(a.chunkSize)
	}

	// Clear usage stats (historical counts like ChunksAllocated/TotalAllocs unchanged)
	a.stats.BytesUsed = 0
	a.stats.BytesWasted = 0
}

func (a *Arena) Usage() float64 {
	stats := a.Stats()
	if stats.BytesReserved == 0 {
		return 0
	}
	return float64(stats.BytesUsed) / float64(stats.BytesReserved) * 100
}

func (a *Arena) String() string {
	stats := a.Stats()
	return fmt.Sprintf(
		"Arena{chunks: %d, reserved: %.2f MB, used: %.2f MB, wasted: %.2f KB, usage: %.1f%%, allocs: %d}",
		stats.ActiveChunks,
		float64(stats.BytesReserved)/(1024*1024),
		float64(stats.BytesUsed)/(1024*1024),
		float64(stats.BytesWasted)/1024,
		a.Usage(),
		stats.TotalAllocs,
	)
}
