package flat

import (
	"fmt"

	"github.com/hupe1980/vecgo/index"
)

// Stats returns statistics about flat.
func (f *Flat) Stats() index.Stats {
	maxID := f.maxID.Load()
	currentDim := int(f.dimension.Load())

	// Count active nodes
	deletedNodes := 0
	for i := uint64(0); i < maxID; i++ {
		if f.deleted.Test(i) {
			deletedNodes++
		}
	}

	activeNodes := int(maxID) - deletedNodes

	return index.Stats{
		Options: map[string]string{
			"DistanceType": f.opts.DistanceType.String(),
		},
		Parameters: map[string]string{
			"dimension": fmt.Sprintf("%d", currentDim),
		},
		Storage: map[string]string{
			"total slots":       fmt.Sprintf("%d", maxID),
			"active nodes":      fmt.Sprintf("%d", activeNodes),
			"deleted nodes":     fmt.Sprintf("%d (tombstones)", deletedNodes),
			"memory efficiency": fmt.Sprintf("%.1f%%", float64(activeNodes)/float64(max(int(maxID), 1))*100),
		},
		Concurrency: map[string]string{
			"concurrency model": "Lock-free reads (atomic maxID + bitset)",
		},
	}
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
