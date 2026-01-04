package hnsw

import (
	"fmt"

	"github.com/hupe1980/vecgo/internal/conv"
	"github.com/hupe1980/vecgo/model"
)

// Stats returns statistics about the HNSW graph.
func (h *HNSW) Stats() Stats {
	g := h.currentGraph.Load()
	nodes := g.nodes.Load()

	activeNodes, deletedNodes := h.countNodes(nodes)

	maxLevel := int(g.maxLevelAtomic.Load())
	levelStats, connectionStats, connectionNodeStats := h.collectLevelStats(g, nodes, maxLevel)

	levelStatsStructs := make([]LevelStats, maxLevel+1)
	for i := 0; i <= maxLevel; i++ {
		avg := 0
		if connectionNodeStats[i] > 0 {
			avg = connectionStats[i] / connectionNodeStats[i]
		}
		levelStatsStructs[i] = LevelStats{
			Level:          i,
			Nodes:          levelStats[i],
			Connections:    connectionStats[i],
			AvgConnections: avg,
		}
	}

	numSegments := 0
	if nodes != nil {
		numSegments = len(*nodes)
	}

	return Stats{
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

func (h *HNSW) countNodes(nodes *[]*NodeSegment) (active, deleted int) {
	if nodes == nil {
		return 0, 0
	}
	for _, seg := range *nodes {
		if seg == nil {
			continue
		}
		for j := range seg {
			if seg[j].Load() == nil {
				deleted++
			} else {
				active++
			}
		}
	}
	return
}

func (h *HNSW) collectLevelStats(g *graph, nodes *[]*NodeSegment, maxLevel int) ([]int, []int, []int) {
	levelStats := make([]int, maxLevel+1)
	connectionStats := make([]int, maxLevel+1)
	connectionNodeStats := make([]int, maxLevel+1)

	if nodes == nil {
		return levelStats, connectionStats, connectionNodeStats
	}

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

			iU64, _ := conv.IntToUint64(i)
			jU64, _ := conv.IntToUint64(j)
			id := iU64*nodeSegmentSize + jU64

			// Loop through each connection
			for i2 := level; i2 >= 0; i2-- {
				idU32, _ := conv.Uint64ToUint32(id)
				conns := h.getConnections(g, model.RowID(idU32), i2)
				count := len(conns)

				if count > 0 {
					connectionStats[i2] += count
					connectionNodeStats[i2]++
				}
			}
		}
	}
	return levelStats, connectionStats, connectionNodeStats
}
