// Package container implements container data structures.
package container

import (
	"sync"
	"sync/atomic"
)

const (
	// segmentBits determines the size of each segment.
	// 16 bits = 65536 items per segment.
	segmentBits = 16
	segmentSize = 1 << segmentBits
	segmentMask = segmentSize - 1
)

// SegmentedArray is a thread-safe, lock-free, segmented array.
// It supports append-only growth and random access.
type SegmentedArray[T any] struct {
	segments atomic.Pointer[[]*Segment[T]]
	mu       sync.Mutex // Protects growth
}

// Segment is a fixed-size array of items.
type Segment[T any] struct {
	items [segmentSize]T
}

// NewSegmentedArray creates a new SegmentedArray.
func NewSegmentedArray[T any]() *SegmentedArray[T] {
	sa := &SegmentedArray[T]{}
	// Initialize with empty segments slice
	segments := make([]*Segment[T], 0)
	sa.segments.Store(&segments)
	return sa
}

// Get returns the item at the given index.
// Returns zero value if index is out of bounds or segment not allocated.
func (sa *SegmentedArray[T]) Get(index uint32) (T, bool) {
	segments := sa.segments.Load()
	if segments == nil {
		var zero T
		return zero, false
	}
	segIdx := int(index >> segmentBits)
	if segIdx >= len(*segments) {
		var zero T
		return zero, false
	}
	seg := (*segments)[segIdx]
	if seg == nil {
		var zero T
		return zero, false
	}
	return seg.items[index&segmentMask], true
}

// Set sets the item at the given index.
// It grows the array if necessary.
func (sa *SegmentedArray[T]) Set(index uint32, value T) {
	segIdx := int(index >> segmentBits)

	// Fast path: check if segment exists
	segments := sa.segments.Load()
	if segments != nil && segIdx < len(*segments) && (*segments)[segIdx] != nil {
		(*segments)[segIdx].items[index&segmentMask] = value
		return
	}

	// Slow path: grow
	sa.mu.Lock()
	defer sa.mu.Unlock()

	// Reload under lock
	segments = sa.segments.Load()
	var currentSegments []*Segment[T]
	if segments != nil {
		currentSegments = *segments
	}

	// Check again
	if segIdx < len(currentSegments) && currentSegments[segIdx] != nil {
		currentSegments[segIdx].items[index&segmentMask] = value
		return
	}

	// Grow slice if needed
	newSegments := currentSegments
	if segIdx >= len(newSegments) {
		// Create new slice with capacity
		grown := make([]*Segment[T], segIdx+1)
		copy(grown, newSegments)
		newSegments = grown
	}

	// Allocate segment if needed
	if newSegments[segIdx] == nil {
		newSegments[segIdx] = &Segment[T]{}
	}

	// Store value
	newSegments[segIdx].items[index&segmentMask] = value

	// Publish new segments
	sa.segments.Store(&newSegments)
}
