package hnsw

import (
	"math"
	"sync/atomic"
	"unsafe"

	"github.com/hupe1980/vecgo/internal/arena"
	"github.com/hupe1980/vecgo/internal/core"
)

// NodeOffsetSegment is a fixed-size array of node offsets.
type NodeOffsetSegment [nodeSegmentSize]atomic.Uint32

// Neighbor represents a connection to another node with its distance.
type Neighbor struct {
	ID   core.LocalID
	Dist float32
}

// AsUint64 converts Neighbor to uint64 for atomic storage.
func (n Neighbor) AsUint64() uint64 {
	return uint64(n.ID)<<32 | uint64(math.Float32bits(n.Dist))
}

// NeighborFromUint64 converts uint64 back to Neighbor.
func NeighborFromUint64(v uint64) Neighbor {
	return Neighbor{
		ID:   core.LocalID(v >> 32),
		Dist: math.Float32frombits(uint32(v)),
	}
}

// Node is a view over the arena data.
// It does not store data itself, but provides methods to access data in the arena.
type Node struct {
	Offset uint32
	Gen    uint32
}

// Layout:
// [0:4] Level (uint32)
// [4:...] Connections

// Connection Block Layout:
// [0:4] Count (uint32)
// [4:...] Neighbors (uint64 array)

func (n Node) Level(a *arena.Arena) int {
	ptr := a.GetSafe(arena.Ref{Gen: n.Gen, Offset: n.Offset})
	if ptr == nil {
		// Stale reference or invalid offset
		return -1
	}
	return int(*(*uint32)(ptr))
}

func (n Node) setLevel(a *arena.Arena, level int) {
	ptr := a.GetSafe(arena.Ref{Gen: n.Gen, Offset: n.Offset})
	if ptr == nil {
		return
	}
	*(*uint32)(ptr) = uint32(level)
}

// connectionBlockOffset calculates the offset of the connection block for the given layer.
func (n Node) connectionBlockOffset(layer int, m, m0 int) uint32 {
	// Base offset after Level
	offset := n.Offset + 4

	if layer == 0 {
		return offset
	}

	// Skip layer 0
	// Layer 0 size: 4 (count) + m0 * 8 (neighbors)
	offset += 4 + uint32(m0)*8

	// Skip layers 1 to layer-1
	// Each layer size: 4 (padding) + 4 (count) + m * 8 (neighbors)
	if layer > 1 {
		offset += uint32(layer-1) * (4 + 4 + uint32(m)*8)
	}

	// Add padding for current layer to align neighbors to 8 bytes
	offset += 4

	return offset
}

// GetConnectionsRaw returns the neighbors for the given layer as raw uint64s.
// It returns a slice backed by the arena memory.
// WARNING: The slice is valid only until the arena is reset/freed.
// Concurrent access: Safe for concurrent reads if updates are atomic.
func (n Node) GetConnectionsRaw(a *arena.Arena, layer int, m, m0 int) []uint64 {
	blockOffset := n.connectionBlockOffset(layer, m, m0)
	ptr := a.GetSafe(arena.Ref{Gen: n.Gen, Offset: blockOffset})
	if ptr == nil {
		return nil
	}

	count := atomic.LoadUint32((*uint32)(ptr))
	neighborsPtr := unsafe.Pointer(uintptr(ptr) + 4)

	return unsafe.Slice((*uint64)(neighborsPtr), count)
}

func (n Node) GetConnection(a *arena.Arena, layer int, index int, m, m0 int) Neighbor {
	blockOffset := n.connectionBlockOffset(layer, m, m0)
	neighborOffset := blockOffset + 4 + uint32(index)*8

	ptr := a.GetSafe(arena.Ref{Gen: n.Gen, Offset: neighborOffset})
	if ptr == nil {
		return Neighbor{}
	}
	return NeighborFromUint64(atomic.LoadUint64((*uint64)(ptr)))
}

func (n Node) SetConnection(a *arena.Arena, layer int, index int, neighbor Neighbor, m, m0 int) {
	blockOffset := n.connectionBlockOffset(layer, m, m0)
	// Neighbors start at offset + 4
	// Index * 8
	neighborOffset := blockOffset + 4 + uint32(index)*8

	ptr := a.GetSafe(arena.Ref{Gen: n.Gen, Offset: neighborOffset})
	if ptr == nil {
		return
	}
	atomic.StoreUint64((*uint64)(ptr), neighbor.AsUint64())
}

func (n Node) SetCount(a *arena.Arena, layer int, count int, m, m0 int) {
	blockOffset := n.connectionBlockOffset(layer, m, m0)
	ptr := a.GetSafe(arena.Ref{Gen: n.Gen, Offset: blockOffset})
	if ptr == nil {
		return
	}
	atomic.StoreUint32((*uint32)(ptr), uint32(count))
}

func (n Node) GetCount(a *arena.Arena, layer int, m, m0 int) int {
	blockOffset := n.connectionBlockOffset(layer, m, m0)
	ptr := a.GetSafe(arena.Ref{Gen: n.Gen, Offset: blockOffset})
	if ptr == nil {
		return 0
	}
	return int(atomic.LoadUint32((*uint32)(ptr)))
}

// Init initializes the node in the arena.
func (n Node) Init(a *arena.Arena, level int, m, m0 int) {
	n.setLevel(a, level)
	// Counts are 0 by default (arena is zeroed? No, arena is not zeroed if reused!)
	// Arena `Alloc` returns zeroed memory?
	// `make([]byte)` returns zeroed memory.
	// If we reuse arena (Reset), we don't zero it.
	// So we MUST initialize counts to 0.

	for i := 0; i <= level; i++ {
		n.SetCount(a, i, 0, m, m0)
	}
}

// AllocNode allocates a new node in the arena and returns it.
func AllocNode(a *arena.Arena, level int, m, m0 int) (Node, error) {
	// Calculate size
	// Header: 4
	// Level 0: 4 + m0 * 8
	// Level > 0: 4 (padding) + 4 (count) + m * 8

	size := 4 + (4 + m0*8) + level*(4+4+m*8)

	offset, _, err := a.Alloc(size)
	if err != nil {
		return Node{}, err
	}
	node := Node{Offset: offset, Gen: a.Generation()}
	node.Init(a, level, m, m0)
	return node, nil
}
