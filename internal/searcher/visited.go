package searcher

import "github.com/hupe1980/vecgo/model"

// VisitedSet tracks visited nodes using a generation-based approach.
// optimizing for O(1) reset and efficient cache usage.
type VisitedSet struct {
	visited    []uint16
	generation uint16
}

// NewVisitedSet creates a new visited set.
func NewVisitedSet(capacity int) *VisitedSet {
	return &VisitedSet{
		visited:    make([]uint16, capacity),
		generation: 1, // Start at 1, 0 is "never visited"
	}
}

// Visit marks a node as visited.
func (v *VisitedSet) Visit(id model.RowID) {
	if int(id) >= len(v.visited) {
		v.grow(int(id) + 1)
	}
	v.visited[int(id)] = v.generation
}

// Visited returns true if the node has been visited.
func (v *VisitedSet) Visited(id model.RowID) bool {
	if int(id) >= len(v.visited) {
		return false
	}
	return v.visited[int(id)] == v.generation
}

// Reset clears the visited status for all nodes visited in the current session.
// It increments the generation counter, providing O(1) reset cost.
func (v *VisitedSet) Reset() {
	v.generation++
	if v.generation == 0 {
		// Overflow, reset all (rare: once per 65536 searches)
		v.generation = 1
		clear(v.visited)
	}
}

// EnsureCapacity ensures the visited set can hold at least the given number of nodes.
func (v *VisitedSet) EnsureCapacity(capacity int) {
	if capacity > len(v.visited) {
		v.grow(capacity)
	}
}

func (v *VisitedSet) grow(newLen int) {
	currentLen := len(v.visited)
	newCap := max(currentLen*2, newLen)

	newVisited := make([]uint16, newCap)
	copy(newVisited, v.visited)
	v.visited = newVisited
}
