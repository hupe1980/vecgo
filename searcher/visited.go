package searcher

import "github.com/hupe1980/vecgo/model"

// VisitedSet tracks visited nodes using a bitset and a dirty list for fast reset.
type VisitedSet struct {
	bits  []uint64
	dirty []model.RowID
}

// NewVisitedSet creates a new visited set.
func NewVisitedSet(capacity int) *VisitedSet {
	// capacity is number of nodes.
	// bits needed = (capacity + 63) / 64
	return &VisitedSet{
		bits:  make([]uint64, (capacity+63)/64),
		dirty: make([]model.RowID, 0, 128), // Initial capacity for dirty list
	}
}

// Visit marks a node as visited.
func (v *VisitedSet) Visit(id model.RowID) {
	wordIdx := int(id >> 6)
	bitMask := uint64(1) << (id & 63)

	if wordIdx >= len(v.bits) {
		v.grow(wordIdx + 1)
	}

	if v.bits[wordIdx]&bitMask == 0 {
		v.bits[wordIdx] |= bitMask
		v.dirty = append(v.dirty, id)
	}
}

// Visited returns true if the node has been visited.
func (v *VisitedSet) Visited(id model.RowID) bool {
	wordIdx := int(id >> 6)
	if wordIdx >= len(v.bits) {
		return false
	}
	return v.bits[wordIdx]&(uint64(1)<<(id&63)) != 0
}

// Reset clears the visited status for all nodes visited in the current session.
func (v *VisitedSet) Reset() {
	for _, id := range v.dirty {
		wordIdx := int(id >> 6)
		bitMask := uint64(1) << (id & 63)
		v.bits[wordIdx] &^= bitMask
	}
	v.dirty = v.dirty[:0]
}

// EnsureCapacity ensures the visited set can hold at least the given number of nodes.
func (v *VisitedSet) EnsureCapacity(capacity int) {
	wordIdx := (capacity + 63) / 64
	if wordIdx > len(v.bits) {
		v.grow(wordIdx)
	}
}

func (v *VisitedSet) grow(newLen int) {
	currentLen := len(v.bits)
	newCap := max(currentLen*2, newLen)

	newBits := make([]uint64, newCap)
	copy(newBits, v.bits)
	v.bits = newBits
}
