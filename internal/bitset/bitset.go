package bitset

import (
	"encoding/binary"
	"io"
	"math/bits"
	"sync/atomic"
)

const (
	// segmentBits determines the size of each segment.
	// 16 bits = 65536 bits per segment.
	segmentBits = 16
	segmentSize = 1 << segmentBits // 65536 bits
	segmentMask = segmentSize - 1

	// wordsPerSegment is the number of uint64 words in a segment.
	// 65536 bits / 64 bits/word = 1024 words.
	wordsPerSegment = segmentSize / 64
)

// BitSegment is a fixed-size segment of the bitset.
type BitSegment [wordsPerSegment]atomic.Uint64

// BitSet is a thread-safe, lock-free, segmented bitset.
type BitSet struct {
	segments atomic.Pointer[[]*BitSegment]
	size     atomic.Uint64
}

// New creates a new BitSet with the given size (in bits).
func New(size uint64) *BitSet {
	b := &BitSet{}
	b.size.Store(size)
	b.growSegments(size)
	return b
}

// growSegments ensures enough segments exist for the given size.
func (b *BitSet) growSegments(size uint64) {
	if size == 0 {
		return
	}
	targetIdx := int((size - 1) >> segmentBits)

	// Fast path
	segments := b.segments.Load()
	if segments != nil && len(*segments) > targetIdx && (*segments)[targetIdx] != nil {
		return
	}

	// Slow path: CAS loop
	for {
		oldSegments := b.segments.Load()
		var newSegments []*BitSegment
		currentLen := 0
		if oldSegments != nil {
			currentLen = len(*oldSegments)
		}

		if targetIdx < currentLen && (*oldSegments)[targetIdx] != nil {
			return // Already grown
		}

		newLen := max(targetIdx+1, currentLen)
		newSegments = make([]*BitSegment, newLen)

		if oldSegments != nil {
			copy(newSegments, *oldSegments)
		}

		// Allocate missing segments
		// Optimization: Only iterate over the new range.
		// We assume 0..currentLen-1 are already populated or intentionally nil.
		for i := currentLen; i < newLen; i++ {
			if newSegments[i] == nil {
				newSegments[i] = new(BitSegment)
			}
		}

		newSegmentsPtr := new([]*BitSegment)
		*newSegmentsPtr = newSegments

		if b.segments.CompareAndSwap(oldSegments, newSegmentsPtr) {
			return
		}
	}
}

// Set sets the bit at the given index.
func (b *BitSet) Set(i uint64) {
	limit := b.size.Load()
	if i >= limit {
		return
	}

	segIdx := int(i >> segmentBits)
	segments := b.segments.Load()
	if segments == nil || segIdx >= len(*segments) {
		return
	}

	seg := (*segments)[segIdx]
	if seg == nil {
		return
	}

	offset := i & segmentMask
	wordIdx := offset / 64
	bitMask := uint64(1 << (offset % 64))

	seg[wordIdx].Or(bitMask)
}

// TestAndSet sets the bit at the given index and returns true if it was ALREADY set.
func (b *BitSet) TestAndSet(i uint64) bool {
	limit := b.size.Load()
	if i >= limit {
		return false
	}

	segIdx := int(i >> segmentBits)
	segments := b.segments.Load()
	if segments == nil || segIdx >= len(*segments) {
		return false
	}

	seg := (*segments)[segIdx]
	if seg == nil {
		return false
	}

	offset := i & segmentMask
	wordIdx := offset / 64
	bitMask := uint64(1 << (offset % 64))

	// Optimistic check
	prev := seg[wordIdx].Load()
	if (prev & bitMask) != 0 {
		return true
	}

	// Optimistic check failed, try atomic OR
	// We need to know if WE set it.
	for {
		oldVal := seg[wordIdx].Load()
		if (oldVal & bitMask) != 0 {
			return true // Already set
		}
		newVal := oldVal | bitMask
		if seg[wordIdx].CompareAndSwap(oldVal, newVal) {
			return false // We set it
		}
	}
}

// Unset clears the bit at the given index.
func (b *BitSet) Unset(i uint64) {
	limit := b.size.Load()
	if i >= limit {
		return
	}

	segIdx := int(i >> segmentBits)
	segments := b.segments.Load()
	if segments == nil || segIdx >= len(*segments) {
		return
	}

	seg := (*segments)[segIdx]
	if seg == nil {
		return
	}

	offset := i & segmentMask
	wordIdx := offset / 64
	bitMask := uint64(1 << (offset % 64))

	seg[wordIdx].And(^bitMask)
}

// Test returns true if the bit at the given index is set.
func (b *BitSet) Test(i uint64) bool {
	limit := b.size.Load()
	if i >= limit {
		return false
	}

	segIdx := int(i >> segmentBits)
	segments := b.segments.Load()
	if segments == nil || segIdx >= len(*segments) {
		return false
	}

	seg := (*segments)[segIdx]
	if seg == nil {
		return false
	}

	offset := i & segmentMask
	wordIdx := offset / 64
	bitMask := uint64(1 << (offset % 64))

	return (seg[wordIdx].Load() & bitMask) != 0
}

// NextSetBit returns the index of the next set bit starting from i (inclusive).
// Returns -1 if no bit is set after i.
func (b *BitSet) NextSetBit(i uint64) int64 {
	limit := b.size.Load()
	if i >= limit {
		return -1
	}

	segments := b.segments.Load()
	if segments == nil {
		return -1
	}

	// 1. Check the word containing i
	segIdx := int(i >> segmentBits)
	if segIdx >= len(*segments) {
		return -1
	}

	offset := i & segmentMask
	wordIdx := int(offset / 64)
	bitOffset := int(offset % 64)

	seg := (*segments)[segIdx]
	if seg != nil {
		val := seg[wordIdx].Load()
		// Mask out bits before bitOffset
		val &= ^((1 << bitOffset) - 1)
		if val != 0 {
			return int64(uint64(segIdx)*segmentSize + uint64(wordIdx)*64 + uint64(bits.TrailingZeros64(val)))
		}
	}

	// 2. Check remaining words in the current segment
	if seg != nil {
		for w := wordIdx + 1; w < wordsPerSegment; w++ {
			val := seg[w].Load()
			if val != 0 {
				return int64(uint64(segIdx)*segmentSize + uint64(w)*64 + uint64(bits.TrailingZeros64(val)))
			}
		}
	}

	// 3. Check remaining segments
	for s := segIdx + 1; s < len(*segments); s++ {
		seg = (*segments)[s]
		if seg == nil {
			continue
		}
		for w := 0; w < wordsPerSegment; w++ {
			val := seg[w].Load()
			if val != 0 {
				return int64(uint64(s)*segmentSize + uint64(w)*64 + uint64(bits.TrailingZeros64(val)))
			}
		}
	}

	return -1
}

// Grow ensures the bitset can hold at least size bits.
func (b *BitSet) Grow(size uint64) {
	b.growSegments(size)
	for {
		cur := b.size.Load()
		if size <= cur {
			break
		}
		if b.size.CompareAndSwap(cur, size) {
			break
		}
	}
}

// WriteTo writes the bitset to the writer.
func (b *BitSet) WriteTo(w io.Writer) (int64, error) {
	size := b.size.Load()
	if err := binary.Write(w, binary.LittleEndian, size); err != nil {
		return 0, err
	}
	n := int64(8)

	segments := b.segments.Load()
	if segments == nil {
		return n, nil
	}

	numWords := (size + 63) / 64

	for i := uint64(0); i < numWords; i++ {
		bitIdx := i * 64
		segIdx := int(bitIdx >> segmentBits)

		var val uint64
		if segIdx < len(*segments) {
			seg := (*segments)[segIdx]
			if seg != nil {
				offset := bitIdx & segmentMask
				wordIdx := offset / 64
				val = seg[wordIdx].Load()
			}
		}

		if err := binary.Write(w, binary.LittleEndian, val); err != nil {
			return n, err
		}
		n += 8
	}
	return n, nil
}

// ReadFrom reads the bitset from the reader.
func (b *BitSet) ReadFrom(r io.Reader) (int64, error) {
	var size uint64
	if err := binary.Read(r, binary.LittleEndian, &size); err != nil {
		return 0, err
	}
	// Fix: Grow segments BEFORE updating size to ensure readers don't see out-of-bounds segments
	b.growSegments(size)
	b.size.Store(size)

	n := int64(8)
	numWords := (size + 63) / 64

	segments := b.segments.Load()

	for i := uint64(0); i < numWords; i++ {
		var val uint64
		if err := binary.Read(r, binary.LittleEndian, &val); err != nil {
			return n, err
		}

		bitIdx := i * 64
		segIdx := int(bitIdx >> segmentBits)
		if segments != nil && segIdx < len(*segments) {
			seg := (*segments)[segIdx]
			if seg != nil {
				offset := bitIdx & segmentMask
				wordIdx := offset / 64
				seg[wordIdx].Store(val)
			}
		}
		n += 8
	}
	return n, nil
}

// Count returns the number of set bits.
func (b *BitSet) Count() int {
	count := 0
	segments := b.segments.Load()
	if segments == nil {
		return 0
	}

	// Fast path for single segment
	if len(*segments) == 1 {
		seg := (*segments)[0]
		if seg != nil {
			for i := 0; i < wordsPerSegment; i++ {
				val := seg[i].Load()
				if val != 0 {
					count += bits.OnesCount64(val)
				}
			}
		}
		return count
	}

	size := b.size.Load()
	numWords := (size + 63) / 64
	currentWord := uint64(0)

	for _, seg := range *segments {
		if currentWord >= numWords {
			break
		}
		if seg == nil {
			currentWord += wordsPerSegment
			continue
		}

		limit := wordsPerSegment
		if remaining := int(numWords - currentWord); remaining < limit {
			limit = remaining
		}

		for i := 0; i < limit; i++ {
			val := seg[i].Load()
			if val != 0 {
				count += bits.OnesCount64(val)
			}
		}
		currentWord += wordsPerSegment
	}
	return count
}

// ClearAll clears all bits in the bitset.
func (b *BitSet) ClearAll() {
	segments := b.segments.Load()
	if segments == nil {
		return
	}
	for _, seg := range *segments {
		if seg != nil {
			for i := range seg {
				seg[i].Store(0)
			}
		}
	}
}

// Len returns the size of the bitset in bits.
func (b *BitSet) Len() uint64 {
	return b.size.Load()
}

// FastBitSet is a non-thread-safe bitset optimized for reuse in pools.
// It uses a dirty list to allow O(K) reset where K is the number of set bits.
type FastBitSet struct {
	bits  []uint64
	dirty []uint64
}

// NewFast creates a new FastBitSet.
func NewFast(capacity int) *FastBitSet {
	// capacity is number of nodes.
	// bits needed = (capacity + 63) / 64
	return &FastBitSet{
		bits:  make([]uint64, (capacity+63)/64),
		dirty: make([]uint64, 0, 128), // Initial capacity for dirty list
	}
}

// Set marks a bit as set.
func (b *FastBitSet) Set(id uint64) {
	wordIdx := int(id >> 6)
	bitMask := uint64(1) << (id & 63)

	if wordIdx >= len(b.bits) {
		b.grow(wordIdx + 1)
	}

	if b.bits[wordIdx]&bitMask == 0 {
		b.bits[wordIdx] |= bitMask
		b.dirty = append(b.dirty, id)
	}
}

// TestAndSet sets the bit and returns true if it was already set.
func (b *FastBitSet) TestAndSet(id uint64) bool {
	wordIdx := int(id >> 6)
	bitMask := uint64(1) << (id & 63)

	if wordIdx >= len(b.bits) {
		b.grow(wordIdx + 1)
	}

	if b.bits[wordIdx]&bitMask != 0 {
		return true
	}

	b.bits[wordIdx] |= bitMask
	b.dirty = append(b.dirty, id)
	return false
}

// Test returns true if the bit is set.
func (b *FastBitSet) Test(id uint64) bool {
	wordIdx := int(id >> 6)
	if wordIdx >= len(b.bits) {
		return false
	}
	return b.bits[wordIdx]&(uint64(1)<<(id&63)) != 0
}

// Reset clears all set bits.
func (b *FastBitSet) Reset() {
	for _, id := range b.dirty {
		wordIdx := int(id >> 6)
		bitMask := uint64(1) << (id & 63)
		b.bits[wordIdx] &^= bitMask
	}
	b.dirty = b.dirty[:0]
}

func (b *FastBitSet) grow(newLen int) {
	currentLen := len(b.bits)
	newCap := currentLen * 2
	if newCap < newLen {
		newCap = newLen
	}

	newBits := make([]uint64, newCap)
	copy(newBits, b.bits)
	b.bits = newBits
}
