package arena

import (
	"context"
	"errors"
	"fmt"
	"math/bits"
	"runtime"
	"sync"
	"sync/atomic"
	"unsafe"

	"github.com/hupe1980/vecgo/internal/conv"
	"github.com/hupe1980/vecgo/internal/mmap"
)

// Stats are always tracked. The overhead is negligible (~0.5ns per alloc)
// and the visibility is valuable for debugging and monitoring.

// ArenaState represents the lifecycle state of an arena.
type ArenaState int32

const (
	// Building means the arena accepts allocations and writes.
	Building ArenaState = iota
	// Frozen means the arena is read-only. No more allocations or writes.
	// This is the state during query phase for maximum performance.
	Frozen
)

// MemoryAcquirer is an interface for acquiring memory.
type MemoryAcquirer interface {
	AcquireMemory(ctx context.Context, amount int64) error
	ReleaseMemory(amount int64)
}

var (
	// ErrMaxChunksExceeded is returned when the arena exceeds the maximum number of chunks.
	ErrMaxChunksExceeded = errors.New("arena: max chunks exceeded")
	// ErrAllocationFailed is returned when an allocation fails.
	ErrAllocationFailed = errors.New("arena: allocation failed")
)

const (
	// DefaultChunkSize is the default size of a chunk (4MB).
	// Larger chunks reduce CAS contention and syscall overhead.
	// Apple Silicon and modern x86 benefit from larger linear memory regions.
	DefaultChunkSize = 4 * 1024 * 1024
	// DefaultAlignment is the default memory alignment (8 bytes).
	DefaultAlignment = 8
	// MaxChunks limits the number of chunks to prevent excessive memory usage.
	// Limit to 256GB addressable space with 4MB chunks.
	MaxChunks = 65536
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

type atomicStats struct {
	ChunksAllocated atomic.Uint64
	BytesReserved   atomic.Uint64
	BytesUsed       atomic.Uint64
	BytesWasted     atomic.Uint64
	ActiveChunks    atomic.Uint64
	TotalAllocs     atomic.Uint64
}

type chunk struct {
	data    []byte
	mapping *mmap.Mapping // Holds the off-heap mapping (if applicable)
	offset  atomic.Int64  // MUST be atomic - accessed concurrently without locks
	index   uint32        // Index of this chunk in the arena
}

// Arena is a memory arena allocator.
//
// MEMORY MODEL:
// - Arena is append-only (monotonic). Memory is never reused.
// - Once allocated, offsets remain valid until Free().
// - Arena can be frozen for read-only access (query phase).
// - This design matches DiskANN's memory semantics.
//
// LIFECYCLE:
// - Building: allocations and writes allowed
// - Frozen: read-only, no allocations (optimal for queries)
// - Freed: all memory released, arena unusable
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
	refs       atomic.Int64 // Reference count for safety
	state      atomic.Int32 // ArenaState: Building or Frozen
	acquirer   MemoryAcquirer
}

// Option is a configuration option for Arena.
type Option func(*Arena)

// WithMemoryAcquirer sets the memory acquirer for the arena.
func WithMemoryAcquirer(acquirer MemoryAcquirer) Option {
	return func(a *Arena) {
		a.acquirer = acquirer
	}
}

// New creates a new Arena with the given chunk size.
func New(chunkSize int, opts ...Option) (*Arena, error) {
	if chunkSize <= 0 {
		chunkSize = DefaultChunkSize
	}

	// Round up to next power of 2 for efficient bitwise operations
	chunkBits := bits.Len(uint(chunkSize - 1))
	// If chunkSize is already a power of 2, bits.Len(chunkSize-1) is correct.
	// Example: 1024 (10000000000) -> 1023 (1111111111) -> Len=10. 1<<10 = 1024. Correct.
	// Example: 1025 -> 1024 -> Len=11. 1<<11 = 2048. Correct.

	chunkSize = 1 << chunkBits

	// Validate that global offset encoding won't overflow:
	// globalOffset = (chunkIndex << chunkBits) | localOffset
	// chunkIndex can be up to MaxChunks-1, requiring bits.Len(MaxChunks-1) bits
	// Total bits needed: chunkBits + bits.Len(MaxChunks-1) must be < 64
	maxChunkIndexBits := bits.Len(uint(MaxChunks - 1))
	if chunkBits+maxChunkIndexBits >= 64 {
		return nil, fmt.Errorf("arena: chunk size %d with MaxChunks %d would overflow 64-bit offset encoding", chunkSize, MaxChunks)
	}

	chunkMask, err := conv.IntToUint64(chunkSize - 1)
	if err != nil {
		return nil, err
	}

	a := &Arena{
		chunkSize: chunkSize,
		chunkBits: chunkBits,
		chunkMask: chunkMask,
		alignment: DefaultAlignment,
	}

	for _, opt := range opts {
		opt(a)
	}

	// Arena starts in Building state
	a.state.Store(int32(Building))

	if err := a.allocateChunk(context.Background()); err != nil {
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

// State returns the current state of the arena.
func (a *Arena) State() ArenaState {
	return ArenaState(a.state.Load())
}

// Freeze transitions the arena to read-only state.
// After freezing:
// - No more allocations are allowed
// - All reads are lock-free and safe
// - This is the optimal state for query phase
//
// Freeze is idempotent - calling it multiple times is safe.
func (a *Arena) Freeze() {
	a.state.Store(int32(Frozen))
}

// IsFrozen returns true if the arena is in read-only state.
func (a *Arena) IsFrozen() bool {
	return ArenaState(a.state.Load()) == Frozen
}

func (a *Arena) allocateChunk(ctx context.Context) error {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.allocateChunkLocked(ctx)
}

func (a *Arena) allocateChunkLocked(ctx context.Context) error {
	idx := a.chunkCount.Load()
	// fmt.Printf("DEBUG: Allocating chunk %d (Max: %d)\n", idx, MaxChunks)

	if idx >= MaxChunks {
		// This is a critical failure for the arena.
		return ErrMaxChunksExceeded
	}

	// Acquire memory if acquirer is set
	if a.acquirer != nil {
		chunkSize64 := int64(a.chunkSize)
		// Respect caller's context - they control timeouts
		if err := a.acquirer.AcquireMemory(ctx, chunkSize64); err != nil {
			return err
		}
	}

	// Use off-heap anonymous mapping to avoid GC pressure for large graphs.
	// On Unix: mmap with MAP_ANONYMOUS (lazy allocation, demand-paged).
	// On Windows: VirtualAlloc with MEM_COMMIT (demand-paged).
	//
	// If this fails, the most common cause is missing Close() calls on HNSW indexes,
	// which prevents arena memory from being released. Ensure all HNSW instances
	// are closed with defer idx.Close() after creation.
	mapping, err := mmap.MapAnon(a.chunkSize)
	if err != nil {
		if a.acquirer != nil {
			a.acquirer.ReleaseMemory(int64(a.chunkSize))
		}
		return fmt.Errorf("failed to allocate arena chunk (%d bytes): %w (hint: ensure HNSW.Close() is called to release memory)", a.chunkSize, err)
	}

	newChunk := &chunk{
		data:    mapping.Bytes(),
		mapping: mapping,
		index:   idx,
	}

	// CRITICAL ORDERING for publication safety:
	// 1. Store chunk pointer in array
	// 2. Update current for allocators
	// 3. Increment count (makes chunk visible to readers)
	//
	// This ensures any reader that sees chunkCount > idx will
	// also see the chunk pointer (due to seq-cst atomics).
	a.chunks[idx].Store(newChunk)
	a.current.Store(newChunk)

	// Update stats before making visible
	a.stats.ChunksAllocated.Add(1)
	a.stats.BytesReserved.Add(uint64(a.chunkSize))
	a.stats.ActiveChunks.Add(1)

	// Finally, increment count to make chunk visible to Get()
	a.chunkCount.Add(1)

	return nil
}

// Alloc allocates memory and returns the global offset and the byte slice.
// The global offset can be used with Get() to retrieve the pointer later.
//
// Returns error if the arena is frozen (read-only state).
func (a *Arena) Alloc(size int) (uint64, []byte, error) {
	if size <= 0 {
		return 0, nil, nil
	}

	// Check if arena is frozen
	if ArenaState(a.state.Load()) == Frozen {
		return 0, nil, errors.New("arena is frozen (read-only)")
	}

	mask := a.alignment - 1
	alignedSize := (size + mask) & ^mask

	for {
		curr := a.current.Load()
		if curr == nil {
			return 0, nil, errors.New("arena is closed")
		}

		offset, data, ok := a.tryAllocInChunk(curr, size, alignedSize)
		if ok {
			return offset, data, nil
		}

		// Current chunk is full. Try to allocate a new one.
		if a.current.Load() != curr {
			continue
		}

		a.mu.Lock()
		if a.current.Load() != curr {
			a.mu.Unlock()
			continue
		}

		if err := a.allocateChunkLocked(context.Background()); err != nil {
			a.mu.Unlock()
			return 0, nil, err
		}
		a.mu.Unlock()
	}
}

// tryAllocInChunk is the fast path for allocation.
func (a *Arena) tryAllocInChunk(curr *chunk, size, alignedSize int) (uint64, []byte, bool) {
	oldOffset := curr.offset.Load()
	newOffset := oldOffset + int64(alignedSize)

	if newOffset > int64(len(curr.data)) {
		return 0, nil, false
	}

	// Publish the allocation via CAS.
	// Memory is guaranteed zero from mmap (MAP_ANON/MEM_COMMIT).
	// Visibility to other goroutines is established by the atomic stores
	// that follow allocation (e.g., storing count, then pointer).
	// NOTE: Do NOT reuse arena memory after Reset() for live data structures.
	if !curr.offset.CompareAndSwap(oldOffset, newOffset) {
		return 0, nil, false
	}

	// Update stats (atomic, lock-free)
	a.stats.BytesUsed.Add(uint64(size))
	a.stats.BytesWasted.Add(uint64(alignedSize - size))
	a.stats.TotalAllocs.Add(1)

	// Calculate global offset
	globalOffset := (uint64(curr.index) << a.chunkBits) | uint64(oldOffset)
	data := curr.data[oldOffset:newOffset:newOffset]

	return globalOffset, data, true
}

// Get returns an unsafe.Pointer to the memory at the given global offset.
// Returns nil if the offset is invalid (out of bounds).
//
// SAFETY: With monotonic arena semantics, once an offset is returned from
// Alloc(), it remains valid until Free(). No generation checking is needed.
// This is the same memory model used by DiskANN.
func (a *Arena) Get(offset uint64) unsafe.Pointer {
	chunkIdx := offset >> a.chunkBits
	chunkOffset := offset & a.chunkMask

	// Bounds check against current chunk count
	count := a.chunkCount.Load()
	if chunkIdx >= uint64(count) {
		return nil
	}

	// Load chunk pointer (atomic for safe concurrent access)
	c := a.chunks[chunkIdx].Load()
	if c == nil {
		return nil
	}

	// Bounds check within chunk
	if chunkOffset >= uint64(len(c.data)) {
		return nil
	}

	return unsafe.Add(unsafe.Pointer(&c.data[0]), chunkOffset)
}

// AllocPointer allocates memory for a struct of the given size.
// The returned pointer is aligned to DefaultAlignment (8 bytes).
func (a *Arena) AllocPointer(size int) (unsafe.Pointer, error) {
	_, bytes, err := a.Alloc(size)
	if err != nil {
		return nil, err
	}
	return unsafe.Pointer(&bytes[0]), nil
}

// AllocBytes allocates a byte slice of the given size.
func (a *Arena) AllocBytes(size int) ([]byte, error) {
	_, bytes, err := a.Alloc(size)
	return bytes, err
}

// AllocUint32Slice allocates a uint32 slice of the given capacity.
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

// AllocFloat32Slice allocates a float32 slice of the given capacity.
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

// Stats returns the current arena statistics.
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
//  1. Do NOT call Free concurrently with allocations OR Get() calls
//  2. Memory is eligible for GC but not immediately returned to OS
//  3. All slices allocated from this arena become invalid after Free
//  4. All offsets become invalid - Get() on old offsets is undefined behavior
//  5. Typical usage: defer arena.Free() or call in Close() method
//
// After Free(), the arena cannot be reused. Create a new arena instead.
func (a *Arena) Free() {
	// Wait for references to drop
	for a.refs.Load() > 0 {
		runtime.Gosched()
	}

	a.mu.Lock()
	defer a.mu.Unlock()

	// Release memory if acquirer is set
	if a.acquirer != nil {
		bytesReserved := a.stats.BytesReserved.Load()
		if bytesReserved > 0 {
			bytesReserved64 := int64(bytesReserved)
			a.acquirer.ReleaseMemory(bytesReserved64)
		}
	}

	// Mark arena as freed
	a.state.Store(int32(Frozen))

	// Clear chunk references and unmap memory
	count := a.chunkCount.Load()
	for i := uint32(0); i < count; i++ {
		chunk := a.chunks[i].Load()
		if chunk != nil && chunk.mapping != nil {
			// Unmap off-heap memory
			_ = chunk.mapping.Close()
		}
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

// Usage returns the memory usage percentage.
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
