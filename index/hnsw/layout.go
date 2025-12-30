package hnsw

import (
	"encoding/binary"
	"sync/atomic"
	"unsafe"
)

// Node layout constants
const (
	nodeIDOffset    = 0
	nodeLevelOffset = 8
	// 4 bytes padding to align to 16 bytes
	nodeHeaderSize = 16
)

// nodeLayout manages the memory layout of nodes in the flat arena.
//
// New Layout for Lock-Free / COW support:
// [0-7]   ID (uint64)
// [8-11]  Level (uint32)
// [12-15] Padding (unused)
// [16...] Link Array: (Level+1) * 8 bytes.
//
//	Each entry is an atomic offset (uint64) pointing to the neighbor list for that layer.
//
// Neighbor List Layout (stored elsewhere in arena, pointed to by Link Array):
// [0-3]   Count (uint32)
// [4-7]   Capacity (uint32) - strictly speaking redundant if we know M/M0, but useful for debugging/safety
// [8...]  Neighbors (uint64 * Capacity)
type nodeLayout struct {
	M  int
	M0 int
}

func newNodeLayout(M int) *nodeLayout {
	return &nodeLayout{
		M:  M,
		M0: M * 2,
	}
}

// InitialSize returns the size required for a new node allocation.
// This includes the header, the link array, and the initial empty neighbor lists.
func (l *nodeLayout) InitialSize(level int) uint32 {
	// Header
	size := uint32(nodeHeaderSize)

	// Link Array: (level + 1) * 8 bytes
	size += uint32(level+1) * 8

	// Initial Neighbor Lists (inline for locality on creation)
	// Layer 0
	size += l.neighborListSize(l.M0)
	// Layers 1..level
	if level > 0 {
		size += uint32(level) * l.neighborListSize(l.M)
	}

	return size
}

func (l *nodeLayout) neighborListSize(capacity int) uint32 {
	return 8 + uint32(capacity)*8 // 4 byte count + 4 byte cap + neighbors
}

// Initialize writes the initial structure of the node into the provided buffer.
// It returns the offsets for the neighbor lists so they can be populated or linked.
func (l *nodeLayout) Initialize(data []byte, id uint64, level int) {
	// 1. Write Header
	binary.LittleEndian.PutUint64(data[nodeIDOffset:], id)
	binary.LittleEndian.PutUint32(data[nodeLevelOffset:], uint32(level))

	// 2. Calculate offsets
	linkArrayStart := uint32(nodeHeaderSize)
	// The first neighbor list starts immediately after the link array
	currentListOffset := linkArrayStart + uint32(level+1)*8

	// 3. Initialize Link Array and Neighbor Lists
	for layer := 0; layer <= level; layer++ {
		// Write the offset to the link array
		// Note: This offset is relative to the node start?
		// No, for the arena, we usually want absolute offsets or relative to node.
		// If we use relative to node, we can't easily move lists.
		// Let's assume these are RELATIVE offsets from the node start for now,
		// because the node itself might be moved? No, nodes are fixed in arena.
		// Actually, if we want to support COW, the new list will be allocated
		// somewhere else in the arena. So these MUST be offsets relative to the
		// Arena start, OR we need the node's base address to resolve relative offsets.
		//
		// Given the `memtable` design, `data` passed here is a slice of the arena.
		// But `Initialize` doesn't know the absolute address of `data` in the arena.
		//
		// We have two choices:
		// A) Store relative offsets from node start. (Hard if list moves far away)
		// B) Store "Arena IDs" or absolute offsets.
		//
		// Let's assume the caller handles the "absolute vs relative" logic.
		// For this function, we will write the *initial* relative offsets assuming
		// contiguous allocation.
		//
		// WAIT: The `memtable` returns a pointer (or index) to the node.
		// If we want to update the link array later to point to a new location,
		// that new location will be an absolute offset in the arena.
		// So the Link Array MUST store Absolute Arena Offsets.
		//
		// However, `Initialize` only sees `data []byte`. It doesn't know the node's
		// offset in the arena.
		//
		// Refactor: `Initialize` should take `nodeBaseOffset uint32`.
		// But for now, let's just write the structure and let the caller fill the Link Array
		// if it needs absolute addresses.
		//
		// Actually, let's make `Initialize` just set up the local structure.
		// The caller (in `hnsw.go`) knows the node's position.
		//
		// Let's write the *relative* offsets for now, and the caller can adjust them
		// if it wants absolute.
		// OR, simpler: The Link Array is initialized to 0, and the caller sets it up.
		//
		// Let's go with: Initialize sets up the neighbor list headers (count/cap)
		// at the expected inline positions.

		capacity := l.M
		if layer == 0 {
			capacity = l.M0
		}

		// Initialize the list header at currentListOffset
		// Count = 0
		binary.LittleEndian.PutUint32(data[currentListOffset:], 0)
		// Capacity
		binary.LittleEndian.PutUint32(data[currentListOffset+4:], uint32(capacity))

		// We don't write the link array here because we don't know the base offset.
		// The caller must do:
		// offset := nodeBase + currentListOffset
		// atomic.StoreUint64(&linkArray[layer], offset)

		currentListOffset += l.neighborListSize(capacity)
	}
}

// Accessors

func (l *nodeLayout) getID(data []byte) uint64 {
	return binary.LittleEndian.Uint64(data[nodeIDOffset:])
}

func (l *nodeLayout) getLevel(data []byte) int {
	return int(binary.LittleEndian.Uint32(data[nodeLevelOffset:]))
}

// GetLinkAddress returns the offset in `data` where the link for `layer` is stored.
func (l *nodeLayout) GetLinkAddress(layer int) uint32 {
	return uint32(nodeHeaderSize + layer*8)
}

// Helper to calculate the offset of the initial inline neighbor list for a layer.
// Used during initialization.
func (l *nodeLayout) InitialNeighborListOffset(level int, layer int) uint32 {
	offset := uint32(nodeHeaderSize) + uint32(level+1)*8

	// Add previous layers
	for i := 0; i < layer; i++ {
		cap := l.M
		if i == 0 {
			cap = l.M0
		}
		offset += l.neighborListSize(cap)
	}
	return offset
}

// Neighbor List Accessors (operate on the list data directly)

func (l *nodeLayout) getListCount(listData []byte) uint32 {
	return binary.LittleEndian.Uint32(listData[0:])
}

func (l *nodeLayout) setListCount(listData []byte, count uint32) {
	binary.LittleEndian.PutUint32(listData[0:], count)
}

func (l *nodeLayout) getListCapacity(listData []byte) uint32 {
	return binary.LittleEndian.Uint32(listData[4:])
}

func (l *nodeLayout) getListNeighbors(listData []byte) []uint64 {
	count := l.getListCount(listData)
	// Unsafe slice for zero-copy?
	// For now, safe copy to avoid escaping pointers to arena if not careful.
	// But for performance, we want zero-copy.
	// Let's return a slice backed by the arena.

	// Offset 8 is where neighbors start
	ptr := unsafe.Pointer(&listData[8])
	return unsafe.Slice((*uint64)(ptr), count)
}

// Atomic helpers for the Link Array

func (l *nodeLayout) AtomicLoadLink(data []byte, layer int) uint64 {
	offset := l.GetLinkAddress(layer)
	// Must be 64-bit aligned, which it is (16 + layer*8)
	ptr := (*uint64)(unsafe.Pointer(&data[offset]))
	return atomic.LoadUint64(ptr)
}

func (l *nodeLayout) AtomicStoreLink(data []byte, layer int, listOffset uint64) {
	offset := l.GetLinkAddress(layer)
	ptr := (*uint64)(unsafe.Pointer(&data[offset]))
	atomic.StoreUint64(ptr, listOffset)
}
