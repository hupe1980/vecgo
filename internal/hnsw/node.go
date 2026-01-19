package hnsw

// Node and connection list concurrency model:
//
// MEMORY MODEL (Monotonic Arena - DiskANN style):
// - Arena is append-only. Memory is never reused or reset.
// - Once an offset is allocated, it remains valid until arena.Free().
// - Connection lists use copy-on-write (COW) for atomic updates.
// - All pointer/offset stores use atomic.Store (release semantics).
// - All pointer/offset loads use atomic.Load (acquire semantics).
//
// SAFE OPERATIONS:
// - Concurrent reads of connection lists (GetConnectionsRaw, GetConnection)
// - Single-writer updates via SetConnections (COW swap)
// - Concurrent node lookups via getNode/setNode
//
// UNSAFE OPERATIONS (require external synchronization):
// - Arena Free() while nodes are being accessed
//
// LIFETIME:
// - Slices from GetConnectionsRaw are valid until SetConnections swaps
// - After SetConnections, old slices point to logically dead (but valid) memory
// - This is safe because arena is monotonic - memory is never reused
//
// BUILD vs QUERY PHASE:
// - Build phase: allocations and writes allowed
// - Freeze: call arena.Freeze() to transition to read-only
// - Query phase: read-only, no allocations, lock-free

import (
	"context"
	"math"
	"sync/atomic"
	"unsafe"

	"github.com/hupe1980/vecgo/internal/arena"
	"github.com/hupe1980/vecgo/model"
)

// NodeOffsetSegment is a fixed-size array of node offsets.
type NodeOffsetSegment [nodeSegmentSize]atomic.Uint64

// nodeRef represents a reference to a node in the arena.
// With monotonic arena semantics, this is simply the offset.
// The offset remains valid from allocation until arena.Free().
type nodeRef uint64

func (n nodeRef) offset() uint64 {
	return uint64(n)
}

func (n nodeRef) isValid() bool {
	return n != 0
}

// Node is a view over the arena data.
// It uses a value receiver to avoid heap allocations.
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
// Returns -1 if the node reference is invalid.
func (n Node) Level(a *arena.Arena) int {
	offset := n.ref.offset()
	if offset == 0 {
		return -1
	}
	ptr := a.Get(offset)
	if ptr == nil {
		return -1
	}
	return int(*(*uint32)(ptr))
}

func (n Node) setLevel(a *arena.Arena, level int) {
	offset := n.ref.offset()
	if offset == 0 {
		return
	}
	ptr := a.Get(offset)
	if ptr == nil {
		return
	}
	*(*uint32)(ptr) = uint32(level)
}

// GetConnectionListPtr returns the offset to the connection list for the given layer.
// Returns 0 if the node reference is invalid.
func (n Node) GetConnectionListPtr(a *arena.Arena, layer int) uint64 {
	// Node layout:
	// Level at +0 (4 bytes)
	// Padding at +4 (4 bytes)
	// Ptr L0 at +8 (8 bytes)
	// Ptr L1 at +16 (8 bytes)
	// ...
	// Ptr Li at +8 + i*8

	offset := n.ref.offset()
	if offset == 0 {
		return 0
	}
	ptrOffset := offset + 8 + uint64(layer)*8
	ptr := a.Get(ptrOffset)
	if ptr == nil {
		return 0
	}
	return atomic.LoadUint64((*uint64)(ptr))
}

// SetConnectionListPtr sets the offset to the connection list for the given layer.
// This is the publication point - readers will see the new list after this store.
func (n Node) SetConnectionListPtr(a *arena.Arena, layer int, listOffset uint64) {
	offset := n.ref.offset()
	if offset == 0 {
		return
	}
	ptrOffset := offset + 8 + uint64(layer)*8
	ptr := a.Get(ptrOffset)
	if ptr == nil {
		return
	}
	atomic.StoreUint64((*uint64)(ptr), listOffset)
}

// GetConnectionsRaw returns the neighbors for the given layer as raw uint64s.
// It returns a slice backed by the arena memory.
//
// SAFETY: With monotonic arena, the returned slice remains valid memory
// until arena.Free(). After SetConnections, the slice points to the OLD
// connection list which is logically dead but still valid memory.
//
// Concurrent access: Safe for concurrent reads with single-writer updates.
// Readers may see the old or new list during SetConnections, but never
// a corrupted state.
func (n Node) GetConnectionsRaw(a *arena.Arena, layer int, m, m0 int) []uint64 {
	offset := n.ref.offset()
	if offset == 0 {
		return nil
	}

	listOffset := n.GetConnectionListPtr(a, layer)
	if listOffset == 0 {
		return nil
	}

	ptr := a.Get(listOffset)
	if ptr == nil {
		return nil
	}

	count := atomic.LoadUint32((*uint32)(ptr))

	// Sanity check: count should never exceed capacity
	capacity := m
	if layer == 0 {
		capacity = m0
	}
	if count > uint32(capacity) {
		// Memory corruption or uninitialized - return empty
		return nil
	}

	neighborsPtr := unsafe.Pointer(uintptr(ptr) + 8)
	return unsafe.Slice((*uint64)(neighborsPtr), int(count))
}

// GetConnection returns the neighbor at the given index for the given layer.
func (n Node) GetConnection(a *arena.Arena, layer int, index int, m, m0 int) Neighbor {
	listOffset := n.GetConnectionListPtr(a, layer)
	if listOffset == 0 {
		return Neighbor{}
	}

	// List layout: [Count u32][Padding u32][Neighbor0 u64][Neighbor1 u64]...
	neighborOffset := listOffset + 8 + uint64(index)*8

	ptr := a.Get(neighborOffset)
	if ptr == nil {
		return Neighbor{}
	}
	return NeighborFromUint64(atomic.LoadUint64((*uint64)(ptr)))
}

// SetConnection sets the neighbor at the given index for the given layer.
// NOTE: This mutates in place. For concurrent updates, use SetConnections (COW).
func (n Node) SetConnection(a *arena.Arena, layer int, index int, neighbor Neighbor, m, m0 int) {
	listOffset := n.GetConnectionListPtr(a, layer)
	if listOffset == 0 {
		return
	}

	neighborOffset := listOffset + 8 + uint64(index)*8
	ptr := a.Get(neighborOffset)
	if ptr == nil {
		return
	}
	atomic.StoreUint64((*uint64)(ptr), neighbor.AsUint64())
}

// SetCount sets the number of connections for the given layer.
// NOTE: This mutates in place. For concurrent updates, use SetConnections (COW).
func (n Node) SetCount(a *arena.Arena, layer int, count int, m, m0 int) {
	listOffset := n.GetConnectionListPtr(a, layer)
	if listOffset == 0 {
		return
	}

	ptr := a.Get(listOffset)
	if ptr == nil {
		return
	}
	atomic.StoreUint32((*uint32)(ptr), uint32(count))
}

// GetCount returns the number of connections for the given layer.
func (n Node) GetCount(a *arena.Arena, layer int, m, m0 int) int {
	listOffset := n.GetConnectionListPtr(a, layer)
	if listOffset == 0 {
		return 0
	}

	ptr := a.Get(listOffset)
	if ptr == nil {
		return 0
	}
	return int(atomic.LoadUint32((*uint32)(ptr)))
}

// SetConnections atomically replaces the connection list for the given layer (COW).
//
// This is the ONLY safe way to update connections with concurrent readers:
// 1. Allocate new list in arena
// 2. Write all neighbors
// 3. Write count
// 4. Atomically swap pointer
//
// Old connection list memory is not reused (monotonic arena), so readers
// holding old slices still have valid memory.
func (n Node) SetConnections(ctx context.Context, a *arena.Arena, layer int, neighbors []Neighbor, m, m0 int) error {
	capacity := m
	if layer == 0 {
		capacity = m0
	}
	if len(neighbors) > capacity {
		neighbors = neighbors[:capacity]
	}

	// Allocate new list
	size := 8 + capacity*8
	listOffset, data, err := a.Alloc(size)
	if err != nil {
		return err
	}

	ptr := unsafe.Pointer(&data[0])

	// 1. Write all neighbors FIRST (before count)
	neighborsPtr := unsafe.Pointer(uintptr(ptr) + 8)
	target := unsafe.Slice((*uint64)(neighborsPtr), len(neighbors))
	for i, neighbor := range neighbors {
		target[i] = neighbor.AsUint64()
	}

	// 2. Write count - this acts as release for all neighbor writes
	atomic.StoreUint32((*uint32)(ptr), uint32(len(neighbors)))

	// 3. Publish pointer - readers will now see the new list
	n.SetConnectionListPtr(a, layer, listOffset)
	return nil
}

// ReplaceConnections is an alias for SetConnections for backward compatibility.
func (n Node) ReplaceConnections(ctx context.Context, a *arena.Arena, layer int, neighbors []Neighbor, m, m0 int) error {
	return n.SetConnections(ctx, a, layer, neighbors, m, m0)
}

// Init initializes the node in the arena with empty connection lists.
func (n Node) Init(ctx context.Context, a *arena.Arena, level int, m, m0 int) error {
	n.setLevel(a, level)

	// Create initial empty lists for each layer
	for i := 0; i <= level; i++ {
		capacity := m
		if i == 0 {
			capacity = m0
		}

		// Allocate list: 8 bytes header + capacity*8 bytes for neighbors
		size := 8 + capacity*8
		listOffset, data, err := a.Alloc(size)
		if err != nil {
			return err
		}

		// Initialize count to 0
		// Memory is already zeroed by mmap, but explicit store ensures visibility
		atomic.StoreUint32((*uint32)(unsafe.Pointer(&data[0])), 0)

		// Publish the connection list pointer
		n.SetConnectionListPtr(a, i, listOffset)
	}
	return nil
}

// AllocNode allocates a new node in the arena with empty connection lists.
func AllocNode(ctx context.Context, a *arena.Arena, level int, m, m0 int) (Node, error) {
	// Calculate size for Node struct:
	// Header(8) + (Level+1)*8 (Pointers)
	nodeSize := 8 + (level+1)*8

	offset, _, err := a.Alloc(nodeSize)
	if err != nil {
		return Node{}, err
	}

	// With monotonic arena, offset is the only reference needed
	node := Node{ref: nodeRef(offset)}

	if err := node.Init(ctx, a, level, m, m0); err != nil {
		return Node{}, err
	}
	return node, nil
}
