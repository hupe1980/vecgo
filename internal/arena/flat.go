package arena

import (
	"errors"
	"sync/atomic"
)

var (
	ErrArenaFull = errors.New("arena is full")
)

// FlatArena is a contiguous memory arena.
// It is designed to be mmap-able.
// It uses uint64 offsets for addressing.
type FlatArena struct {
	buf []byte
	ptr atomic.Uint64 // Current allocation offset
}

// NewFlat creates a new FlatArena with the given initial size.
func NewFlat(size int) *FlatArena {
	a := &FlatArena{
		buf: make([]byte, size),
	}
	// Reserve offset 0 as invalid/null
	a.ptr.Store(1)
	return a
}

// NewFlatFromBytes creates a FlatArena wrapping an existing byte slice.
func NewFlatFromBytes(buf []byte) *FlatArena {
	a := &FlatArena{
		buf: buf,
	}
	// Assume the buffer is full or managed externally?
	// For mmap loading, we might set ptr to len(buf).
	// But if we want to append, we need to know where the used data ends.
	// For now, let's assume we start at 0 if we are building, or user sets it.
	return a
}

// Alloc allocates size bytes and returns the offset.
// It returns ErrArenaFull if there is not enough space.
func (a *FlatArena) Alloc(size uint64) (uint64, error) {
	// We need to ensure alignment?
	// Let's align to 8 bytes for uint64 access.
	const align = 8

	for {
		cur := a.ptr.Load()
		padding := (align - (cur % align)) % align
		next := cur + padding + size

		// Check bounds (optimistic read of buf len)
		// Note: buf len might change if we grow, but we need a lock for that.
		// For now, let's assume fixed size or lock for alloc if dynamic.
		// To be safe and simple:

		if int(next) > len(a.buf) {
			return 0, ErrArenaFull
		}

		if a.ptr.CompareAndSwap(cur, next) {
			return cur + padding, nil
		}
	}
}

// Get returns a slice of the arena at the given offset and size.
// WARNING: The returned slice is valid only as long as the arena buffer is not reallocated.
func (a *FlatArena) Get(offset uint64, size uint64) []byte {
	return a.buf[offset : offset+size]
}

// Buffer returns the underlying byte slice.
func (a *FlatArena) Buffer() []byte {
	return a.buf
}

// Size returns the current used size.
func (a *FlatArena) Size() uint64 {
	return a.ptr.Load()
}

// SetSize sets the current used size.
func (a *FlatArena) SetSize(size uint64) {
	a.ptr.Store(size)
}

// Reset resets the allocator to the beginning.
func (a *FlatArena) Reset() {
	a.ptr.Store(0)
}
