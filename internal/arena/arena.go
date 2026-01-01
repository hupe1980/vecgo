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
	"errors"
	"fmt"
	"math/bits"
	"runtime"
	"sync"
	"sync/atomic"
	"unsafe"
)

var (
	ErrMaxChunksExceeded = errors.New("arena: max chunks exceeded")
	ErrAllocationFailed  = errors.New("arena: allocation failed")
)

const (
	DefaultChunkSize = 1024 * 1024
	DefaultAlignment = 8
	MaxChunks        = 65536 // Limit to 64GB addressable space with 1MB chunks
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

// Ref represents a safe reference to an arena allocation.
// It includes the generation ID to detect stale references.
type Ref struct {
	Gen    uint32
	Offset uint64
}

type atomicStats struct {
	ChunksAllocated atomic.Uint64
	BytesReserved   atomic.Uint64
	BytesUsed       atomic.Uint64
	BytesWasted     atomic.Uint64
	ActiveChunks    atomic.Uint64
	TotalAllocs     atomic.Uint64
}

type chunk struct {
	data   []byte
	offset atomic.Int64 // MUST be atomic - accessed concurrently without locks
	index  uint32       // Index of this chunk in the arena
}

type Arena struct {
	chunkSize  int
	chunkBits  int    // Power of 2 exponent for chunk size
	chunkMask  uint64 // Mask for offset within chunk
	alignment  int
	chunks     [MaxChunks]atomic.Pointer[chunk] // Fixed-size array to avoid slice race conditions
	chunkCount atomic.Uint32                    // Number of active chunks (protected by mu)
	current    atomic.Pointer[chunk]
	mu         sync.Mutex
	stats      atomicStats
	refs       atomic.Int64  // Reference count for safety
	generation atomic.Uint32 // Generation counter to detect stale offsets
}

func New(chunkSize int) (*Arena, error) {
	if chunkSize <= 0 {
		chunkSize = DefaultChunkSize
	}

	// Round up to next power of 2 for efficient bitwise operations
	chunkBits := bits.Len(uint(chunkSize - 1))
	// If chunkSize is already a power of 2, bits.Len(chunkSize-1) is correct.
	// Example: 1024 (10000000000) -> 1023 (1111111111) -> Len=10. 1<<10 = 1024. Correct.
	// Example: 1025 -> 1024 -> Len=11. 1<<11 = 2048. Correct.

	chunkSize = 1 << chunkBits
	chunkMask := uint64(chunkSize - 1)

	a := &Arena{
		chunkSize: chunkSize,
		chunkBits: chunkBits,
		chunkMask: chunkMask,
		alignment: DefaultAlignment,
	}
	// Initialize generation to 1 so 0 is invalid
	a.generation.Store(1)

	if err := a.allocateChunk(); err != nil {
		return nil, err
	}
	// Reserve offset 0 as null
	if _, _, err := a.Alloc(1); err != nil {
		return nil, err
	}
	return a, nil
}

// IncRef increments the reference count.
// Call this when starting a long-running operation that uses the arena.
func (a *Arena) IncRef() {
	a.refs.Add(1)
}

// DecRef decrements the reference count.
// Call this when finished with the arena.
func (a *Arena) DecRef() {
	a.refs.Add(-1)
}

// Generation returns the current generation of the arena.
func (a *Arena) Generation() uint32 {
	return a.generation.Load()
}

// GetSafe returns a pointer to the data at the given reference.
// It validates the generation and returns nil if the reference is stale.
func (a *Arena) GetSafe(ref Ref) unsafe.Pointer {
	if ref.Gen != a.generation.Load() {
		return nil
	}
	return a.Get(ref.Offset)
}

func (a *Arena) allocateChunk() error {
	a.mu.Lock()
	defer a.mu.Unlock()

	idx := a.chunkCount.Load()

	if idx >= MaxChunks {
		// This is a critical failure for the arena.
		return ErrMaxChunksExceeded
	}

	newChunk := &chunk{
		data:  make([]byte, a.chunkSize),
		index: idx,
	}

	// Store in array using atomic pointer (though lock protects against concurrent allocateChunk,
	// Get() is lock-free and needs to see the pointer safely)
	a.chunks[idx].Store(newChunk)

	// Update stats
	a.stats.ChunksAllocated.Add(1)
	a.stats.BytesReserved.Add(uint64(a.chunkSize))
	a.stats.ActiveChunks.Add(1)

	// Make visible to Alloc
	a.current.Store(newChunk)

	// Update count for next allocation
	a.chunkCount.Add(1)
	return nil
}

// Alloc allocates memory and returns the global offset and the byte slice.
// The global offset can be used with Get() to retrieve the pointer later.
func (a *Arena) Alloc(size int) (uint64, []byte, error) {
	return a.alloc(size, a.alignment)
}

func (a *Arena) alloc(size int, align int) (uint64, []byte, error) {
	if size <= 0 {
		return 0, nil, nil
	}

	mask := align - 1
	alignedSize := (size + mask) & ^mask

	for {
		curr := a.current.Load()
		if curr == nil {
			continue
		}

		oldOffset := curr.offset.Load()
		newOffset := oldOffset + int64(alignedSize)

		if newOffset <= int64(len(curr.data)) {
			if curr.offset.CompareAndSwap(oldOffset, newOffset) {
				a.stats.BytesUsed.Add(uint64(size))
				a.stats.BytesWasted.Add(uint64(alignedSize - size))
				a.stats.TotalAllocs.Add(1)

				// Calculate global offset
				// GlobalOffset = (ChunkIndex << ChunkBits) | ChunkOffset
				// Ensure ChunkOffset fits in ChunkMask
				if uint64(oldOffset) > a.chunkMask {
					return 0, nil, fmt.Errorf("arena: offset exceeds chunk mask")
				}
				globalOffset := (uint64(curr.index) << a.chunkBits) | uint64(oldOffset)
				return globalOffset, curr.data[oldOffset:newOffset:newOffset], nil
			}
			continue
		}

		// Current chunk is full. Try to allocate a new one.
		// We use a lock to ensure only one goroutine allocates a new chunk.
		// But we must be careful not to block readers.
		// The `allocateChunk` method locks `a.mu`.

		// Optimization: Check if someone else already allocated a new chunk
		// by checking if a.current changed.
		if a.current.Load() != curr {
			continue
		}

		// Try to set current to nil to signal we are allocating
		if !a.current.CompareAndSwap(curr, nil) {
			continue
		}

		if err := a.allocateChunk(); err != nil {
			// Restore current if allocation failed so others can try (though if max chunks exceeded, they will also fail)
			// Actually if allocateChunk failed, we are in trouble.
			// We should probably restore the old current so other threads don't spin on nil?
			// But if max chunks exceeded, we can't allocate more.
			// Let's just return error.
			// But we need to restore a.current to something non-nil or handle the nil case in the loop?
			// The loop checks `if curr == nil { continue }`.
			// If we leave it nil, everyone spins.
			// We should restore it.
			a.current.Store(curr)
			return 0, nil, err
		}
	}
}

// Get returns an unsafe.Pointer to the memory at the given global offset.
// It performs no bounds checking.
func (a *Arena) Get(offset uint64) unsafe.Pointer {
	chunkIdx := offset >> a.chunkBits
	chunkOffset := offset & a.chunkMask

	// Bounds check
	if chunkIdx >= uint64(a.chunkCount.Load()) {
		panic("arena: stale offset")
	}

	// Use atomic load to safely read the chunk pointer without locks.
	// Since chunks are only appended and never removed/moved during allocation,
	// and we use a fixed-size array, this is safe.
	// Note: We don't check bounds of chunks array because chunkIdx comes from
	// a valid offset returned by Alloc, which ensures chunkIdx < MaxChunks.
	c := a.chunks[chunkIdx].Load()

	if c == nil {
		panic("Arena.Get: chunk is nil")
	}

	// In a correct program, c should never be nil if offset is valid.
	// However, if offset is garbage, this might panic or return garbage.
	return unsafe.Add(unsafe.Pointer(&c.data[0]), chunkOffset)
}

// AllocPointer allocates memory for a struct of the given size and alignment.
func (a *Arena) AllocPointer(size, align int) (unsafe.Pointer, error) {
	if align <= 0 {
		align = a.alignment
	}
	_, bytes, err := a.alloc(size, align)
	if err != nil {
		return nil, err
	}
	return unsafe.Pointer(&bytes[0]), nil
}

func (a *Arena) AllocBytes(size int) ([]byte, error) {
	_, bytes, err := a.Alloc(size)
	return bytes, err
}

func (a *Arena) AllocUint32Slice(capacity int) ([]uint32, error) {
	if capacity <= 0 {
		return nil, nil
	}

	size := capacity * int(unsafe.Sizeof(uint32(0)))
	_, bytes, err := a.Alloc(size)
	if err != nil {
		return nil, err
	}

	return unsafe.Slice((*uint32)(unsafe.Pointer(&bytes[0])), capacity)[:0:capacity], nil
}

func (a *Arena) AllocFloat32Slice(capacity int) ([]float32, error) {
	if capacity <= 0 {
		return nil, nil
	}

	size := capacity * int(unsafe.Sizeof(float32(0)))
	_, bytes, err := a.Alloc(size)
	if err != nil {
		return nil, err
	}

	return unsafe.Slice((*float32)(unsafe.Pointer(&bytes[0])), capacity)[:0:capacity], nil
}

func (a *Arena) Stats() Stats {
	return Stats{
		ChunksAllocated: a.stats.ChunksAllocated.Load(),
		BytesReserved:   a.stats.BytesReserved.Load(),
		BytesUsed:       a.stats.BytesUsed.Load(),
		BytesWasted:     a.stats.BytesWasted.Load(),
		ActiveChunks:    a.stats.ActiveChunks.Load(),
		TotalAllocs:     a.stats.TotalAllocs.Load(),
	}
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
	// Wait for references to drop
	for a.refs.Load() > 0 {
		runtime.Gosched()
	}

	a.mu.Lock()
	defer a.mu.Unlock()

	// Increment generation to invalidate old references
	a.generation.Add(1)

	// Clear chunk references to enable GC
	count := a.chunkCount.Load()
	for i := 0; i < int(count); i++ {
		a.chunks[i].Store(nil)
	}
	a.chunkCount.Store(0)
	a.current.Store(nil)

	// Update stats to reflect freed state
	a.stats.ActiveChunks.Store(0)
	a.stats.BytesReserved.Store(0)
	a.stats.BytesUsed.Store(0)
	a.stats.BytesWasted.Store(0)
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
	// Wait for references to drop
	for a.refs.Load() > 0 {
		runtime.Gosched()
	}

	a.mu.Lock()
	defer a.mu.Unlock()

	// Increment generation to invalidate old references
	a.generation.Add(1)

	count := a.chunkCount.Load()
	if count > 0 {
		firstChunk := a.chunks[0].Load()
		// Reset offset of first chunk
		firstChunk.offset.Store(0)

		// Free extra chunks (keep first for reuse)
		for i := 1; i < int(count); i++ {
			a.chunks[i].Store(nil)
		}
		a.chunkCount.Store(1)
		a.current.Store(firstChunk)

		// Update stats - only first chunk remains
		a.stats.ActiveChunks.Store(1)
		a.stats.BytesReserved.Store(uint64(a.chunkSize))
	}

	// Clear usage stats (historical counts like ChunksAllocated/TotalAllocs unchanged)
	a.stats.BytesUsed.Store(0)
	a.stats.BytesWasted.Store(0)
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
