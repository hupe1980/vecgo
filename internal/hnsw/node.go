package hnsw

import (
	"context"
	"errors"
	"math"
	"sync/atomic"
	"unsafe"

	"github.com/hupe1980/vecgo/internal/arena"
	"github.com/hupe1980/vecgo/internal/conv"
	"github.com/hupe1980/vecgo/model"
)

// NodeOffsetSegment is a fixed-size array of node offsets.
type NodeOffsetSegment [nodeSegmentSize]atomic.Uint64

// nodeRef represents a packed reference to a node (Generation + Offset).
// Optimizes memory by avoiding pointer chasing and heap allocations for node headers.
//
// Bit Layout:
// [0:40] Offset (40 bits) -> Max 1TB arena size
// [40:64] Gen (24 bits)   -> Max 16M generations
type nodeRef uint64

const (
	offsetBits = 40
	offsetMask = (1 << offsetBits) - 1
	genBits    = 64 - offsetBits
)

// ErrOffsetOverflow is returned when a node offset exceeds 40 bits (1TB arena limit).
var ErrOffsetOverflow = errors.New("hnsw: node offset exceeds 40 bits (1TB limit)")

func packNodeRef(offset uint64, gen uint32) (nodeRef, error) {
	if offset > offsetMask {
		return 0, ErrOffsetOverflow
	}
	return nodeRef(offset | (uint64(gen) << offsetBits)), nil
}

func (n nodeRef) unpack() (offset uint64, gen uint32) {
	offset = uint64(n) & offsetMask
	gen = uint32(uint64(n) >> offsetBits)
	return
}

func (n nodeRef) isValid() bool {
	return n != 0
}

// Node is a view over the arena data.
// It uses a value receiver over the packed reference to avoid heap allocations.
type Node struct {
	ref nodeRef
}

// Neighbor represents a connection to another node with its distance.
type Neighbor struct {
	ID   model.RowID
	Dist float32
}

// AsUint64 converts Neighbor to uint64 for atomic storage.
func (n Neighbor) AsUint64() uint64 {
	return uint64(n.ID)<<32 | uint64(math.Float32bits(n.Dist))
}

// NeighborFromUint64 converts uint64 back to Neighbor.
// This is a hot-path function - uses direct casts for maximum performance
// since the data comes from AsUint64 and is guaranteed valid.
func NeighborFromUint64(v uint64) Neighbor {
	return Neighbor{
		ID:   model.RowID(uint32(v >> 32)),
		Dist: math.Float32frombits(uint32(v)),
	}
}

// IsZero returns true if the node is empty/invalid.
func (n Node) IsZero() bool {
	return !n.ref.isValid()
}

// Level returns the level of the node.
func (n Node) Level(a *arena.Arena) int {
	offset, gen := n.ref.unpack()
	ptr := a.GetSafe(arena.Ref{Gen: gen, Offset: offset})
	if ptr == nil {
		// Stale reference or invalid offset
		return -1
	}
	lvl, err := conv.Uint32ToInt(*(*uint32)(ptr))
	if err != nil {
		return -1 // Should not happen for valid levels
	}
	return lvl
}

func (n Node) setLevel(a *arena.Arena, level int) {
	offset, gen := n.ref.unpack()
	ptr := a.GetSafe(arena.Ref{Gen: gen, Offset: offset})
	if ptr == nil {
		return
	}
	lvlU32, err := conv.IntToUint32(level)
	if err != nil {
		return
	}
	*(*uint32)(ptr) = lvlU32
}

// GetConnectionListPtr returns the offset to the connection list for the given layer.
func (n Node) GetConnectionListPtr(a *arena.Arena, layer int) uint64 {
	// Node at n.Offset
	// Level at +0
	// Padding at +4
	// Ptr L0 at +8
	// Ptr L1 at +16
	// ...
	// Ptr Li at +8 + i*8

	offset, gen := n.ref.unpack()
	ptrOffset := offset + 8 + uint64(layer)*8
	ptr := a.GetSafe(arena.Ref{Gen: gen, Offset: ptrOffset})
	if ptr == nil {
		return 0
	}
	return atomic.LoadUint64((*uint64)(ptr))
}

// SetConnectionListPtr sets the offset to the connection list for the given layer.
func (n Node) SetConnectionListPtr(a *arena.Arena, layer int, listOffset uint64) {
	offset, gen := n.ref.unpack()
	ptrOffset := offset + 8 + uint64(layer)*8
	ptr := a.GetSafe(arena.Ref{Gen: gen, Offset: ptrOffset})
	if ptr == nil {
		return
	}
	atomic.StoreUint64((*uint64)(ptr), listOffset)
}

// GetConnectionsRaw returns the neighbors for the given layer as raw uint64s.
// It returns a slice backed by the arena memory.
// WARNING: The slice is valid only until the arena is reset/freed.
// Concurrent access: Safe for concurrent reads if updates are atomic (COW).
func (n Node) GetConnectionsRaw(a *arena.Arena, layer int, m, m0 int) []uint64 {
	listOffset := n.GetConnectionListPtr(a, layer)
	if listOffset == 0 {
		return nil
	}

	_, gen := n.ref.unpack()
	ptr := a.GetSafe(arena.Ref{Gen: gen, Offset: listOffset})
	if ptr == nil {
		return nil
	}

	count := atomic.LoadUint32((*uint32)(ptr))
	neighborsPtr := unsafe.Pointer(uintptr(ptr) + 8)

	countInt, err := conv.Uint32ToInt(count)
	if err != nil {
		return nil
	}
	return unsafe.Slice((*uint64)(neighborsPtr), countInt)
}

// GetConnection returns the neighbor at the given index for the given layer.
func (n Node) GetConnection(a *arena.Arena, layer int, index int, m, m0 int) Neighbor {
	listOffset := n.GetConnectionListPtr(a, layer)
	if listOffset == 0 {
		return Neighbor{}
	}

	_, gen := n.ref.unpack()
	// List layout: [Count u32][Padding u32][Neighbor0 u64][Neighbor1 u64]...
	neighborOffset := listOffset + 8 + uint64(index)*8

	ptr := a.GetSafe(arena.Ref{Gen: gen, Offset: neighborOffset})
	if ptr == nil {
		return Neighbor{}
	}
	return NeighborFromUint64(atomic.LoadUint64((*uint64)(ptr)))
}

// SetConnection sets the neighbor at the given index for the given layer.
func (n Node) SetConnection(a *arena.Arena, layer int, index int, neighbor Neighbor, m, m0 int) {
	listOffset := n.GetConnectionListPtr(a, layer)
	if listOffset == 0 {
		// Should not happen if initialized correctly
		return
	}

	_, gen := n.ref.unpack()
	neighborOffset := listOffset + 8 + uint64(index)*8

	ptr := a.GetSafe(arena.Ref{Gen: gen, Offset: neighborOffset})
	if ptr == nil {
		return
	}
	atomic.StoreUint64((*uint64)(ptr), neighbor.AsUint64())
}

// SetCount sets the number of connections for the given layer.
func (n Node) SetCount(a *arena.Arena, layer int, count int, m, m0 int) {
	listOffset := n.GetConnectionListPtr(a, layer)
	if listOffset == 0 {
		return
	}

	_, gen := n.ref.unpack()
	ptr := a.GetSafe(arena.Ref{Gen: gen, Offset: listOffset})
	if ptr == nil {
		return
	}
	countU32, err := conv.IntToUint32(count)
	if err != nil {
		return
	}
	atomic.StoreUint32((*uint32)(ptr), countU32)
}

// GetCount returns the number of connections for the given layer.
func (n Node) GetCount(a *arena.Arena, layer int, m, m0 int) int {
	listOffset := n.GetConnectionListPtr(a, layer)
	if listOffset == 0 {
		return 0
	}

	_, gen := n.ref.unpack()
	ptr := a.GetSafe(arena.Ref{Gen: gen, Offset: listOffset})
	if ptr == nil {
		return 0
	}
	count, err := conv.Uint32ToInt(atomic.LoadUint32((*uint32)(ptr)))
	if err != nil {
		return 0
	}
	return count
}

// ReplaceConnections replaces the connection list for the given layer with a new list (COW).
// This ensures atomic visibility of the new list to concurrent readers.
func (n Node) ReplaceConnections(ctx context.Context, a *arena.Arena, layer int, neighbors []Neighbor, m, m0 int) error {
	capacity := m
	if layer == 0 {
		capacity = m0
	}
	if len(neighbors) > capacity {
		neighbors = neighbors[:capacity]
	}

	// Alloc new list
	size := 8 + capacity*8
	offset, data, err := a.Alloc(size)
	if err != nil {
		return err
	}

	ptr := unsafe.Pointer(&data[0])

	// 1. Write Count
	*(*uint32)(ptr) = uint32(len(neighbors))

	// 2. Write Neighbors
	neighborsPtr := unsafe.Pointer(uintptr(ptr) + 8)

	target := unsafe.Slice((*uint64)(neighborsPtr), len(neighbors))
	for i, neighbor := range neighbors {
		target[i] = neighbor.AsUint64()
	}

	// 3. Atomic Swap
	n.SetConnectionListPtr(a, layer, offset)
	return nil
}

// AppendConnection appends a neighbor to the existing list if capacity allows.
// This is safe for concurrent readers because we write the element first, then increment the count.
func (n Node) AppendConnection(a *arena.Arena, layer int, neighbor Neighbor, m, m0 int) bool {
	listOffset := n.GetConnectionListPtr(a, layer)
	if listOffset == 0 {
		return false
	}

	_, gen := n.ref.unpack()
	ptr := a.GetSafe(arena.Ref{Gen: gen, Offset: listOffset})
	if ptr == nil {
		return false
	}

	// Read current count
	countPtr := (*uint32)(ptr)
	count := atomic.LoadUint32(countPtr)

	capacity := m
	if layer == 0 {
		capacity = m0
	}

	if int(count) >= capacity {
		return false // List full
	}

	// Write neighbor at index 'count'
	neighborOffset := 8 + uint64(count)*8
	elemPtr := unsafe.Pointer(uintptr(ptr) + uintptr(neighborOffset))

	atomic.StoreUint64((*uint64)(elemPtr), neighbor.AsUint64())
	atomic.StoreUint32(countPtr, count+1)
	return true
}

// Init initializes the node in the arena.
func (n Node) Init(ctx context.Context, a *arena.Arena, level int, m, m0 int) error {
	n.setLevel(a, level)

	// Create initial empty lists
	for i := 0; i <= level; i++ {
		capacity := m
		if i == 0 {
			capacity = m0
		}

		// Alloc list: 8 + capacity*8 (to ensure 8-byte alignment for neighbors)
		size := 8 + capacity*8
		offset, data, err := a.Alloc(size)
		if err != nil {
			return err
		}

		// Initialize count to 0
		*(*uint32)(unsafe.Pointer(&data[0])) = 0

		n.SetConnectionListPtr(a, i, offset)
	}
	return nil
}

// AllocNode allocates a new node in the arena with pointer-based connection lists.
func AllocNode(ctx context.Context, a *arena.Arena, level int, m, m0 int) (Node, error) {
	// Calculate size for Node struct:
	// Header(8) + (Level+1)*8 (Pointers)
	nodeSize := 8 + (level+1)*8

	offset, _, err := a.Alloc(nodeSize)
	if err != nil {
		return Node{}, err
	}

	// Pack offset and generation into reference
	ref, err := packNodeRef(offset, a.Generation())
	if err != nil {
		return Node{}, err
	}
	node := Node{ref: ref}

	if err := node.Init(ctx, a, level, m, m0); err != nil {
		return Node{}, err
	}
	return node, nil
}
