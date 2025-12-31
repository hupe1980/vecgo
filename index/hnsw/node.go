package hnsw

import (
	"sync/atomic"
)

// NodeSegment is a fixed-size array of pointers to Node.
type NodeSegment [nodeSegmentSize]atomic.Pointer[Node]

// Neighbor represents a connection to another node with its distance.
type Neighbor struct {
	ID   uint64
	Dist float32
}

// Node holds the data for a single element in the graph.
type Node struct {
	Level int
	// connections holds the neighbor lists for each level.
	// Index 0 is the bottom layer.
	connections []atomic.Pointer[[]Neighbor]
}

// newNode creates a new Node with the given level.
func newNode(level int) *Node {
	return &Node{
		Level:       level,
		connections: make([]atomic.Pointer[[]Neighbor], level+1),
	}
}

// getConnections returns the neighbors for the given layer.
func (n *Node) getConnections(layer int) []Neighbor {
	if layer < 0 || layer >= len(n.connections) {
		return nil
	}
	ptr := n.connections[layer].Load()
	if ptr == nil {
		return nil
	}
	return *ptr
}

// setConnections updates the neighbors for the given layer.
func (n *Node) setConnections(layer int, neighbors []Neighbor) {
	if layer < 0 || layer >= len(n.connections) {
		return
	}
	n.connections[layer].Store(&neighbors)
}
