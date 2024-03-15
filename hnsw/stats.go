package hnsw

import "fmt"

// Stats prints statistics about the HNSW graph
func (h *HNSW) Stats() {
	fmt.Println("Options:")
	fmt.Printf("\tM = %d\n", h.opts.M)
	fmt.Printf("\tEF = %d\n", h.opts.EF)
	fmt.Printf("\tHeuristic = %v\n\n", h.opts.Heuristic)

	fmt.Println("Parameters:")
	fmt.Printf("\tmmax = %d\n", h.mmax)
	fmt.Printf("\tmmax0 = %d\n", h.mmax0)
	fmt.Printf("\tep = %d\n", h.ep)
	fmt.Printf("\tmaxLevel = %d\n", h.maxLevel)
	fmt.Printf("\tml = %f\n\n", h.ml)

	fmt.Printf("Number of nodes = %d\n\n", len(h.nodes))

	levelStats := make([]int, h.maxLevel+1)
	connectionStats := make([]int, h.maxLevel+1)
	connectionNodeStats := make([]int, h.maxLevel+1)

	for i := 0; i < len(h.nodes)-1; i++ {
		levelStats[h.nodes[i].Layer]++

		// Loop through each connection
		for i2 := int(h.nodes[i].Layer); i2 >= 0; i2-- {
			if len(h.nodes[i].Connections[i2]) > i2 {
				total := len(h.nodes[i].Connections[i2])
				connectionStats[i2] += total
				connectionNodeStats[i2]++
			}
		}
	}

	fmt.Println("Node Levels:")
	for k, v := range levelStats {
		avg := connectionStats[k] / max(1, connectionNodeStats[k])
		fmt.Printf("\tLevel %d:\n", k)
		fmt.Printf("\t\tNumber of nodes: %d\n", v)
		fmt.Printf("\t\tNumber of connections: %d\n", connectionStats[k])
		fmt.Printf("\t\tAverage connections per node: %d\n", avg)
	}

	fmt.Printf("\nTotal number of node levels = %d\n", len(levelStats))
}
