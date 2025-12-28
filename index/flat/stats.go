package flat

import (
	"fmt"

	"github.com/hupe1980/vecgo/index"
)

// Stats returns statistics about flat.
func (f *Flat) Stats() index.Stats {
	// Lock-free read of current state
	currentState := f.getState()
	currentDim := int(f.dimension.Load())

	// Count active nodes (non-nil)
	activeNodes := 0
	for _, node := range currentState.nodes {
		if node != nil {
			activeNodes++
		}
	}

	deletedNodes := len(currentState.nodes) - activeNodes

	return index.Stats{
		Options: map[string]string{
			"DistanceType": f.opts.DistanceType.String(),
		},
		Parameters: map[string]string{
			"dimension": fmt.Sprintf("%d", currentDim),
		},
		Storage: map[string]string{
			"total slots":       fmt.Sprintf("%d", len(currentState.nodes)),
			"active nodes":      fmt.Sprintf("%d", activeNodes),
			"deleted nodes":     fmt.Sprintf("%d (tombstones)", deletedNodes),
			"free list size":    fmt.Sprintf("%d (reusable IDs)", len(currentState.freeList)),
			"memory efficiency": fmt.Sprintf("%.1f%%", float64(activeNodes)/float64(max(len(currentState.nodes), 1))*100),
		},
		Concurrency: map[string]string{
			"concurrency model": "Copy-on-Write (lock-free reads)",
		},
	}
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
