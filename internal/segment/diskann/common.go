package diskann

import "slices"

type distNode struct {
	id   uint32
	dist float32
}

func sortDistNodes(nodes []distNode) {
	slices.SortFunc(nodes, func(a, b distNode) int {
		if a.dist < b.dist {
			return -1
		}
		if a.dist > b.dist {
			return 1
		}
		return 0
	})
}
