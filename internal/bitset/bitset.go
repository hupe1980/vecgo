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
	// Calculate required number of segments
	// (size + segmentSize - 1) / segmentSize
	// But since indices are 0-based, if size is 1, we need index 0.
	// If size is 65537, we need index 1.
	// So we need segment index (size-1) >> segmentBits to exist.

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

		newLen := targetIdx + 1
		if newLen < currentLen {
			newLen = currentLen
		}
		newSegments = make([]*BitSegment, newLen)

		if oldSegments != nil {
			copy(newSegments, *oldSegments)
		}

		// Allocate missing segments
		for i := 0; i < newLen; i++ {
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
	// Check bounds?
	// If i >= size, we should probably ignore or grow?
	// The original implementation returned if i >= size.
	// But usually we want to grow if we are setting a bit.
	// However, `tombstones` usually has a fixed size or grows explicitly.
	// Let's stick to "return if out of bounds" to match original behavior,
	// BUT `Grow` updates `size`.

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

	// Offset within segment
	offset := i & segmentMask
	wordIdx := offset / 64
	bitMask := uint64(1 << (offset % 64))

	seg[wordIdx].Or(bitMask)
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

// Grow ensures the bitset can hold at least size bits.
func (b *BitSet) Grow(size uint64) {
	// Update size first or last?
	// If we update size first, readers might see out of bounds segments.
	// So grow segments first.
	b.growSegments(size)

	// Update size if larger
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
	// Snapshot size
	size := b.size.Load()
	if err := binary.Write(w, binary.LittleEndian, size); err != nil {
		return 0, err
	}
	n := int64(8)

	segments := b.segments.Load()
	if segments == nil {
		return n, nil
	}

	// We need to write exactly `size` bits worth of data?
	// The original implementation wrote `words` based on size.
	// We should replicate that format for compatibility.

	numWords := (size + 63) / 64

	// Iterate words
	for i := uint64(0); i < numWords; i++ {
		// Find segment
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
	// Read size
	var size uint64
	if err := binary.Read(r, binary.LittleEndian, &size); err != nil {
		return 0, err
	}
	b.size.Store(size)
	b.growSegments(size)

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

	size := b.size.Load()
	numWords := (size + 63) / 64

	for i := uint64(0); i < numWords; i++ {
		bitIdx := i * 64
		segIdx := int(bitIdx >> segmentBits)

		if segIdx < len(*segments) {
			seg := (*segments)[segIdx]
			if seg != nil {
				offset := bitIdx & segmentMask
				wordIdx := offset / 64
				val := seg[wordIdx].Load()
				count += bits.OnesCount64(val)
			}
		}
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
