package hnsw

import (
	"fmt"

	"github.com/hupe1980/vecgo/index"
)

// Stats returns statistics about the HNSW graph.
func (h *HNSW) Stats() index.Stats {
	// Count active vs deleted nodes
	activeNodes := 0
	deletedNodes := 0

	nodes := h.nodes.Load()
	numSegments := 0
	if nodes != nil {
		numSegments = len(*nodes)
		// Iterate all segments
		for _, seg := range *nodes {
			if seg == nil {
				continue
			}
			for j := range seg {
				node := seg[j].Load()
				if node == nil {
					deletedNodes++
				} else {
					activeNodes++
				}
			}
		}
	}

	h.freeListMu.Lock()
	freeListSize := len(h.freeList)
	h.freeListMu.Unlock()

	maxLevel := int(h.maxLevelAtomic.Load())
	levelStats := make([]int, maxLevel+1)
	connectionStats := make([]int, maxLevel+1)
	connectionNodeStats := make([]int, maxLevel+1)

	// Iterate all segments again for detailed stats
	if nodes != nil {
		for _, seg := range *nodes {
			if seg == nil {
				continue
			}
			for j := range seg {
				node := seg[j].Load()
				if node == nil {
					continue
				}

				level := node.Level
				if level < len(levelStats) {
					levelStats[level]++
				}

				// Loop through each connection
				for i2 := level; i2 >= 0; i2-- {
					connections := node.getConnections(i2)
					if len(connections) > 0 {
						total := len(connections)
						connectionStats[i2] += total
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
			"VectorCount":  fmt.Sprintf("%d", activeNodes),
			"Deleted":      fmt.Sprintf("%d", deletedNodes),
			"FreeListSize": fmt.Sprintf("%d", freeListSize),
			"Segments":     fmt.Sprintf("%d", numSegments),
		},
		Levels: levelStatsStructs,
	}
}

// String returns a string representation of the HNSW index.
func (h *HNSW) String() string {
	stats := h.Stats()
	return fmt.Sprintf("HNSW(M=%s, EF=%s, Count=%s, Deleted=%s, MaxLevel=%d)",
		stats.Parameters["M"], stats.Parameters["EF"], stats.Storage["VectorCount"], stats.Storage["Deleted"], h.maxLevelAtomic.Load())
}
