package hnsw

// VisitedSet tracks visited nodes using generation tokens for O(1) reset.
type VisitedSet struct {
	visited []uint32
	token   uint32
}

// NewVisitedSet creates a new visited set.
func NewVisitedSet(capacity int) *VisitedSet {
	return &VisitedSet{
		visited: make([]uint32, capacity),
		token:   1,
	}
}

// Visit marks a node as visited.
func (v *VisitedSet) Visit(id uint32) {
	v.ensureCapacity(int(id))
	v.visited[id] = v.token
}

// Visited returns true if the node has been visited.
func (v *VisitedSet) Visited(id uint32) bool {
	if int(id) >= len(v.visited) {
		return false
	}
	return v.visited[id] == v.token
}

// Reset prepares the set for a new search by incrementing the generation token.
// This is O(1) unless the token overflows (very rare).
func (v *VisitedSet) Reset() {
	v.token++
	if v.token == 0 {
		// Overflow, clear all (O(N))
		// This happens once every 4 billion searches per thread.
		for i := range v.visited {
			v.visited[i] = 0
		}
		v.token = 1
	}
}

func (v *VisitedSet) ensureCapacity(idx int) {
	if idx < len(v.visited) {
		return
	}
	// Grow strategy: double capacity
	newCap := len(v.visited) * 2
	if newCap <= idx {
		newCap = idx + 1
	}
	newVisited := make([]uint32, newCap)
	copy(newVisited, v.visited)
	v.visited = newVisited
}
