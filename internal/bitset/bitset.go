package bitset

import (
	"encoding/binary"
	"io"
	"math/bits"
	"sync"
	"sync/atomic"
)

// BitSet is a thread-safe bitset.
type BitSet struct {
	data []atomic.Uint64
	size uint64
	mu   sync.RWMutex
}

// New creates a new BitSet with the given size (in bits).
func New(size uint64) *BitSet {
	words := (size + 63) / 64
	return &BitSet{
		data: make([]atomic.Uint64, int(words)),
		size: size,
	}
}

// Set sets the bit at the given index.
func (b *BitSet) Set(i uint64) {
	b.mu.RLock()
	defer b.mu.RUnlock()

	if i >= b.size {
		return // Or panic/grow
	}
	idx := i / 64
	mask := uint64(1 << (i % 64))
	b.data[idx].Or(mask)
}

// Unset clears the bit at the given index.
func (b *BitSet) Unset(i uint64) {
	b.mu.RLock()
	defer b.mu.RUnlock()

	if i >= b.size {
		return
	}
	idx := i / 64
	mask := uint64(1 << (i % 64))
	b.data[idx].And(^mask)
}

// Test returns true if the bit at the given index is set.
func (b *BitSet) Test(i uint64) bool {
	b.mu.RLock()
	defer b.mu.RUnlock()

	if i >= b.size {
		return false
	}
	idx := i / 64
	mask := uint64(1 << (i % 64))
	return (b.data[idx].Load() & mask) != 0
}

// Grow ensures the bitset can hold at least size bits.
func (b *BitSet) Grow(size uint64) {
	b.mu.Lock()
	defer b.mu.Unlock()

	if size <= b.size {
		return
	}
	newWords := (size + 63) / 64
	if int(newWords) > len(b.data) {
		newData := make([]atomic.Uint64, int(newWords))
		copy(newData, b.data)
		b.data = newData
	}
	b.size = size
}

// WriteTo writes the bitset to the writer.
func (b *BitSet) WriteTo(w io.Writer) (int64, error) {
	b.mu.RLock()
	defer b.mu.RUnlock()

	// Write size
	if err := binary.Write(w, binary.LittleEndian, uint64(b.size)); err != nil {
		return 0, err
	}
	n := int64(8)

	// Write data
	for i := range b.data {
		val := b.data[i].Load()
		if err := binary.Write(w, binary.LittleEndian, val); err != nil {
			return n, err
		}
		n += 8
	}
	return n, nil
}

// ReadFrom reads the bitset from the reader.
func (b *BitSet) ReadFrom(r io.Reader) (int64, error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	// Read size
	var size uint64
	if err := binary.Read(r, binary.LittleEndian, &size); err != nil {
		return 0, err
	}
	b.size = size
	n := int64(8)

	words := (b.size + 63) / 64
	b.data = make([]atomic.Uint64, int(words))

	for i := 0; i < int(words); i++ {
		var val uint64
		if err := binary.Read(r, binary.LittleEndian, &val); err != nil {
			return n, err
		}
		b.data[i].Store(val)
		n += 8
	}
	return n, nil
}

// Count returns the number of set bits.
func (b *BitSet) Count() int {
	b.mu.RLock()
	defer b.mu.RUnlock()

	count := 0
	for i := range b.data {
		count += bits.OnesCount64(b.data[i].Load())
	}
	return count
}

// ClearAll clears all bits in the bitset.
func (b *BitSet) ClearAll() {
	b.mu.Lock()
	defer b.mu.Unlock()

	for i := range b.data {
		b.data[i].Store(0)
	}
}

// Len returns the size of the bitset in bits.
func (b *BitSet) Len() uint64 {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.size
}
