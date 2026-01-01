package hnsw

import (
	"fmt"

	"github.com/hupe1980/vecgo/core"
	"github.com/hupe1980/vecgo/index"
)

// Stats returns statistics about the HNSW graph.
func (h *HNSW) Stats() index.Stats {
	g := h.currentGraph.Load()
	// Count active vs deleted nodes
	activeNodes := 0
	deletedNodes := 0

	nodes := g.nodes.Load()
	numSegments := 0
	if nodes != nil {
		numSegments = len(*nodes)
		// Iterate all segments
		for _, seg := range *nodes {
			if seg == nil {
				continue
			}
			for j := range seg {
				nodePtr := seg[j].Load()
				if nodePtr == nil {
					deletedNodes++
				} else {
					activeNodes++
				}
			}
		}
	}

	maxLevel := int(g.maxLevelAtomic.Load())
	levelStats := make([]int, maxLevel+1)
	connectionStats := make([]int, maxLevel+1)
	connectionNodeStats := make([]int, maxLevel+1)

	// Iterate all segments again for detailed stats
	if nodes != nil {
		for i, seg := range *nodes {
			if seg == nil {
				continue
			}
			for j := range seg {
				nodePtr := seg[j].Load()
				if nodePtr == nil {
					continue
				}

				node := *nodePtr
				level := node.Level(g.arena)
				if level < len(levelStats) {
					levelStats[level]++
				}

				id := uint64(i)*nodeSegmentSize + uint64(j)

				// Loop through each connection
				for i2 := level; i2 >= 0; i2-- {
					count := 0
					conns := h.getConnections(g, core.LocalID(id), i2)
					count = len(conns)

					if count > 0 {
						connectionStats[i2] += count
						connectionNodeStats[i2]++
					}
				}
			}
		}
	}

	levelStatsStructs := make([]index.LevelStats, maxLevel+1)
	for i := 0; i <= maxLevel; i++ {
		avg := 0
		if connectionNodeStats[i] > 0 {
			avg = connectionStats[i] / connectionNodeStats[i]
		}
		levelStatsStructs[i] = index.LevelStats{
			Level:          i,
			Nodes:          levelStats[i],
			Connections:    connectionStats[i],
			AvgConnections: avg,
		}
	}

	return index.Stats{
		Options: map[string]string{
			"Type":         "HNSW",
			"DistanceType": h.opts.DistanceType.String(),
			"Heuristic":    fmt.Sprintf("%v", h.opts.Heuristic),
		},
		Parameters: map[string]string{
			"M":  fmt.Sprintf("%d", h.maxConnectionsPerLayer),
			"M0": fmt.Sprintf("%d", h.maxConnectionsLayer0),
			"EF": fmt.Sprintf("%d", h.opts.EF),
		},
		Storage: map[string]string{
			"ActiveNodes":  fmt.Sprintf("%d", activeNodes),
			"DeletedNodes": fmt.Sprintf("%d", deletedNodes),
			"Segments":     fmt.Sprintf("%d", numSegments),
			"MaxLevel":     fmt.Sprintf("%d", maxLevel),
		},
		Levels: levelStatsStructs,
	}
}
