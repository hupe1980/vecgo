package diskann

import "sort"

type distNode struct {
	id   uint32
	dist float32
}

func sortDistNodes(nodes []distNode) {
	sort.Slice(nodes, func(i, j int) bool {
		return nodes[i].dist < nodes[j].dist
	})
}
