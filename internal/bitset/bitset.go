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
	size int
	mu   sync.RWMutex
}

// New creates a new BitSet with the given size (in bits).
func New(size int) *BitSet {
	words := (size + 63) / 64
	return &BitSet{
		data: make([]atomic.Uint64, words),
		size: size,
	}
}

// Set sets the bit at the given index.
func (b *BitSet) Set(i int) {
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
func (b *BitSet) Unset(i int) {
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
func (b *BitSet) Test(i int) bool {
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
func (b *BitSet) Grow(size int) {
	b.mu.Lock()
	defer b.mu.Unlock()

	if size <= b.size {
		return
	}
	newWords := (size + 63) / 64
	if newWords > len(b.data) {
		newData := make([]atomic.Uint64, newWords)
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
	if err := binary.Write(w, binary.LittleEndian, int64(b.size)); err != nil {
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
	var size int64
	if err := binary.Read(r, binary.LittleEndian, &size); err != nil {
		return 0, err
	}
	b.size = int(size)
	n := int64(8)

	words := (b.size + 63) / 64
	b.data = make([]atomic.Uint64, words)

	for i := 0; i < words; i++ {
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
