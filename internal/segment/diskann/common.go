package diskann

import "slices"

type distNode struct {
	id   uint32
	dist float32
}

// cmpDistNodeByDist compares distNodes by distance ascending (closest first).
// Package-level function to avoid closure allocation in hot path.
func cmpDistNodeByDist(a, b distNode) int {
	if a.dist < b.dist {
		return -1
	}
	if a.dist > b.dist {
		return 1
	}
	return 0
}

func sortDistNodes(nodes []distNode) {
	slices.SortFunc(nodes, cmpDistNodeByDist)
}
