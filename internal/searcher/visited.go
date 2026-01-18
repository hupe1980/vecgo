package searcher

import "github.com/hupe1980/vecgo/model"

// VisitedSet tracks visited nodes using an epoch-based approach.
// Optimizations:
// - O(1) reset via epoch increment
// - uint8 epochs = 4x smaller memory footprint than uint32
// - Better L1 cache utilization for dense traversals
// - Clear only on overflow (every 256 searches)
// - No separate capacity field (use len() directly)
type VisitedSet struct {
	visited []uint8
	epoch   uint8
}

// NewVisitedSet creates a new visited set.
func NewVisitedSet(capacity int) *VisitedSet {
	return &VisitedSet{
		visited: make([]uint8, capacity),
		epoch:   1, // Start at 1, 0 is "never visited"
	}
}

// Visit marks a node as visited.
func (v *VisitedSet) Visit(id model.RowID) {
	idx := int(id)
	if idx >= len(v.visited) {
		v.grow(idx + 1)
	}
	v.visited[idx] = v.epoch
}

// Visited returns true if the node has been visited.
func (v *VisitedSet) Visited(id model.RowID) bool {
	idx := int(id)
	if idx >= len(v.visited) {
		return false
	}
	return v.visited[idx] == v.epoch
}

// CheckAndVisit checks if a node was visited and marks it visited in one operation.
// Returns true if the node was ALREADY visited (skip this node).
// This is the hot-path method - combines check+mark into single memory access.
// Pattern: if visited.CheckAndVisit(id) { continue } // already seen
func (v *VisitedSet) CheckAndVisit(id model.RowID) bool {
	idx := int(id)
	// Fast path: within pre-allocated capacity (common case after EnsureCapacity)
	if idx < len(v.visited) {
		ep := v.epoch
		if v.visited[idx] == ep {
			return true // Already visited
		}
		v.visited[idx] = ep
		return false // First visit
	}
	// Slow path: need to grow
	v.grow(idx + 1)
	v.visited[idx] = v.epoch
	return false // First visit (new allocation, definitely not visited)
}

// CheckAndVisitUnsafe is the hot-path version that skips bounds checking.
// ONLY use this after calling EnsureCapacity with a value >= max possible ID.
// This saves ~2-3ns per call by eliminating the bounds check.
func (v *VisitedSet) CheckAndVisitUnsafe(id model.RowID) bool {
	idx := int(id)
	ep := v.epoch
	if v.visited[idx] == ep {
		return true // Already visited
	}
	v.visited[idx] = ep
	return false // First visit
}

// CheckAndVisitWithEpoch is the fully optimized hot-path version.
// ONLY use this after:
// 1. Calling EnsureCapacity with a value >= max possible ID
// 2. Hoisting epoch via Epoch() at the start of search
// This eliminates bounds check AND epoch field load per call.
func (v *VisitedSet) CheckAndVisitWithEpoch(id model.RowID, epoch uint8) bool {
	idx := int(id)
	if v.visited[idx] == epoch {
		return true // Already visited
	}
	v.visited[idx] = epoch
	return false // First visit
}

// Epoch returns the current epoch for hoisting in hot loops.
// Usage: ep := visited.Epoch(); then use CheckAndVisitWithEpoch(id, ep)
func (v *VisitedSet) Epoch() uint8 {
	return v.epoch
}

// Reset clears the visited status for all nodes visited in the current session.
// It increments the epoch counter, providing O(1) reset cost.
// Full clear only happens on overflow (every 256 searches).
func (v *VisitedSet) Reset() {
	v.epoch++
	if v.epoch == 0 {
		// Overflow, reset all (every 256 searches - still cheap amortized)
		v.epoch = 1
		clear(v.visited)
	}
}

// EnsureCapacity ensures the visited set can hold at least the given number of nodes.
// After this call, CheckAndVisitUnsafe/CheckAndVisitWithEpoch can be used for IDs < capacity.
func (v *VisitedSet) EnsureCapacity(capacity int) {
	if capacity > len(v.visited) {
		v.grow(capacity)
	}
}

// Capacity returns the current capacity of the visited set.
func (v *VisitedSet) Capacity() int {
	return len(v.visited)
}

func (v *VisitedSet) grow(newLen int) {
	oldLen := len(v.visited)
	newCap := max(oldLen*2, newLen)

	newVisited := make([]uint8, newCap)
	copy(newVisited, v.visited)
	v.visited = newVisited
}
