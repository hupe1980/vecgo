package hnsw

import (
	"encoding/binary"
	"unsafe"
)

// Node layout constants
const (
	nodeIDOffset    = 0
	nodeLevelOffset = 8
	nodeHeaderSize  = 12
)

// nodeLayout manages the memory layout of nodes in the flat arena.
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

// Size returns the total size in bytes required for a node of the given level.
func (l *nodeLayout) Size(level int) uint32 {
	// Header: ID (8) + Level (4)
	size := uint32(nodeHeaderSize)

	// Layer 0: Count (4) + Neighbors (M0 * 8)
	size += 4 + uint32(l.M0)*8

	// Layers 1..level: Count (4) + Neighbors (M * 8)
	if level > 0 {
		size += uint32(level) * (4 + uint32(l.M)*8)
	}

	return size
}

// offsets returns the byte offsets for a specific layer's count and neighbors.
// It assumes the node starts at 0. Add the node's base offset to the result.
func (l *nodeLayout) layerOffsets(level int, layer int) (countOffset uint32, neighborsOffset uint32) {
	offset := uint32(nodeHeaderSize)

	// Layer 0
	if layer == 0 {
		return offset, offset + 4
	}

	// Skip Layer 0
	offset += 4 + uint32(l.M0)*8

	// Skip intermediate layers
	// Each intermediate layer is: 4 + M*8
	layerSize := 4 + uint32(l.M)*8
	offset += uint32(layer-1) * layerSize

	return offset, offset + 4
}

// Accessor helpers (using unsafe for speed, or binary.LittleEndian)
// We'll use binary.LittleEndian for safety first, optimize later.

func (l *nodeLayout) getID(data []byte) uint64 {
	return binary.LittleEndian.Uint64(data[nodeIDOffset:])
}

func (l *nodeLayout) getLevel(data []byte) int {
	return int(binary.LittleEndian.Uint32(data[nodeLevelOffset:]))
}

func (l *nodeLayout) getLayerCount(data []byte, layer int) uint32 {
	// We need to calculate offset dynamically because nodes vary in size?
	// No, the layout is fixed for a given M.
	// But we need to know the node's structure.
	// Wait, layerOffsets depends on the layer we are accessing, not the node's max level.
	// The node's max level determines its total size, but the offsets for layer 0, 1, 2 are constant relative to the start.

	countOff, _ := l.layerOffsets(0, layer) // 0 is dummy here, layerOffsets doesn't use level
	return binary.LittleEndian.Uint32(data[countOff:])
}

func (l *nodeLayout) setLayerCount(data []byte, layer int, count uint32) {
	countOff, _ := l.layerOffsets(0, layer)
	binary.LittleEndian.PutUint32(data[countOff:], count)
}

func (l *nodeLayout) getNeighbors(data []byte, level int, layer int) []uint64 {
	count := l.getLayerCount(data, layer)
	_, neighborsOff := l.layerOffsets(level, layer)

	res := make([]uint64, count)
	for i := uint32(0); i < count; i++ {
		res[i] = binary.LittleEndian.Uint64(data[neighborsOff+i*8:])
	}
	return res
}

// Unsafe version for zero-copy (if needed later)
func (l *nodeLayout) getNeighborsUnsafe(data []byte, layer int) []uint32 {
	_, neighborsOff := l.layerOffsets(0, layer)
	ptr := unsafe.Pointer(&data[neighborsOff])

	// Determine capacity based on layer
	cap := l.M
	if layer == 0 {
		cap = l.M0
	}

	return unsafe.Slice((*uint32)(ptr), cap)
}
