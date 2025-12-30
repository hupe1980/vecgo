package hnsw

import (
	"sync/atomic"
)

// NodeSegment is a fixed-size array of pointers to Node.
type NodeSegment [nodeSegmentSize]atomic.Pointer[Node]

// Node holds the data for a single element in the graph.
type Node struct {
	Level int
	// connections holds the neighbor lists for each level.
	// Index 0 is the bottom layer.
	connections []atomic.Pointer[[]uint64]
}

// newNode creates a new Node with the given level.
func newNode(level int) *Node {
	return &Node{
		Level:       level,
		connections: make([]atomic.Pointer[[]uint64], level+1),
	}
}

// getConnections returns the neighbors for the given layer.
func (n *Node) getConnections(layer int) []uint64 {
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
func (n *Node) setConnections(layer int, neighbors []uint64) {
	if layer < 0 || layer >= len(n.connections) {
		return
	}
	n.connections[layer].Store(&neighbors)
}
