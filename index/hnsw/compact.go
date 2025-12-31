package hnsw

import (
	"context"
	"runtime"
	"sync"
)

// Compact removes deleted nodes from the graph connections and attempts to repair connectivity.
// It iterates through all active nodes, removes neighbors that have been deleted,
// and if the number of connections drops too low, it searches for replacement neighbors.
func (h *HNSW) Compact(ctx context.Context) error {
	g := h.currentGraph.Load()
	maxID := g.nextIDAtomic.Load()
	tombstones := g.tombstones

	numWorkers := runtime.GOMAXPROCS(0)

	// Phase 1: Repair Active Nodes (Preserving Tombstones)
	// We identify active nodes that would lose too many connections if we pruned tombstones.
	// We find new active neighbors for them and ADD them to the connection list,
	// keeping the tombstones as bridges for now.
	{
		var wg sync.WaitGroup
		jobs := make(chan uint64, 1024)

		worker := func() {
			defer wg.Done()
			for id := range jobs {
				if ctx.Err() != nil {
					return
				}
				// Skip deleted nodes in this phase
				if tombstones.Test(id) {
					continue
				}
				// Repair active nodes, keeping tombstones
				h.reconcileNode(ctx, g, id)
			}
		}

		for i := 0; i < numWorkers; i++ {
			wg.Add(1)
			go worker()
		}

		for id := uint64(0); id < maxID; id++ {
			if ctx.Err() != nil {
				break
			}
			if h.getNode(g, id) != nil {
				jobs <- id
			}
		}
		close(jobs)
		wg.Wait()
	}

	if ctx.Err() != nil {
		return ctx.Err()
	}

	// Phase 2: Prune Active Nodes
	// Now that active nodes have found new active neighbors (in Phase 1),
	// we can safely remove the tombstones from their connection lists.
	{
		var wg sync.WaitGroup
		jobs := make(chan uint64, 1024)

		worker := func() {
			defer wg.Done()
			for id := range jobs {
				if ctx.Err() != nil {
					return
				}
				// Skip deleted nodes in this phase
				if tombstones.Test(id) {
					continue
				}
				// Remove tombstones from connections
				h.pruneNodeConnections(ctx, g, id)
			}
		}

		for i := 0; i < numWorkers; i++ {
			wg.Add(1)
			go worker()
		}

		for id := uint64(0); id < maxID; id++ {
			if ctx.Err() != nil {
				break
			}
			if h.getNode(g, id) != nil {
				jobs <- id
			}
		}
		close(jobs)
		wg.Wait()
	}

	if ctx.Err() != nil {
		return ctx.Err()
	}

	// Phase 3: Clear Deleted Nodes
	// Now that active nodes no longer point to deleted nodes,
	// we can safely clear the outgoing connections of deleted nodes to reclaim memory.
	{
		var wg sync.WaitGroup
		jobs := make(chan uint64, 1024)

		worker := func() {
			defer wg.Done()
			for id := range jobs {
				if ctx.Err() != nil {
					return
				}
				// Only process deleted nodes
				if tombstones.Test(id) {
					// CRITICAL: Do not clear connections of the entry point, even if deleted.
					// It serves as the gateway to the graph until a new entry point is chosen.
					if id == g.entryPointAtomic.Load() {
						continue
					}
					h.clearNodeConnections(g, id)
				}
			}
		}

		for i := 0; i < numWorkers; i++ {
			wg.Add(1)
			go worker()
		}

		for id := uint64(0); id < maxID; id++ {
			if ctx.Err() != nil {
				break
			}
			if h.getNode(g, id) != nil {
				jobs <- id
			}
		}
		close(jobs)
		wg.Wait()
	}

	return ctx.Err()
}

// reconcileNode checks if an active node needs repair (due to deleted neighbors).
// If so, it finds new active neighbors and appends them to the connection list,
// preserving the deleted neighbors as bridges.
func (h *HNSW) reconcileNode(ctx context.Context, g *graph, id uint64) {
	// Step 1: Check if repair is needed (Read-only check)
	needsRepair, layersToRepair := h.checkRepairNeeded(g, id)
	if !needsRepair {
		return
	}

	// Step 2: Perform search for new neighbors (without holding the lock)
	vec, err := h.VectorByID(ctx, id)
	if err != nil {
		return
	}

	node := h.getNode(g, id)
	if node == nil {
		return
	}

	// Get scratch buffer
	scratch := h.scratchPool.Get().(*scratch)
	defer h.scratchPool.Put(scratch)

	// Greedy descent to node.Level
	epID := g.entryPointAtomic.Load()
	maxLevel := int(g.maxLevelAtomic.Load())
	currID := epID
	currDist := h.dist(vec, currID)

	for level := maxLevel; level > node.Level; level-- {
		changed := true
		for changed {
			changed = false
			conns := h.getConnections(g, currID, level)
			for _, nextID := range conns {
				d := h.dist(vec, nextID)
				if d < currDist {
					currDist = d
					currID = nextID
					changed = true
				}
			}
		}
	}

	// Repair specific layers
	for level := node.Level; level >= 0; level-- {
		// Search for candidates
		filter := func(x uint64) bool { return x != id }
		candidates, err := h.searchLayer(g, vec, currID, currDist, level, h.opts.EF, filter)
		if err != nil {
			return
		}

		// If this layer needs repair, update connections
		if layersToRepair[level] {
			// Select best M active neighbors
			newNeighbors := h.selectNeighbors(candidates, h.opts.M, scratch)

			// Now we need to merge with existing tombstones
			g.shardedLocks[id%uint64(len(g.shardedLocks))].Lock()

			// Re-read current connections
			currentConns := node.getConnections(level)

			// Identify tombstones in current connections
			var tombstonesToKeep []uint64
			for _, neighborID := range currentConns {
				if g.tombstones.Test(neighborID) {
					tombstonesToKeep = append(tombstonesToKeep, neighborID)
				}
			}

			// Combine: New Active + Old Tombstones
			finalConns := make([]uint64, 0, len(newNeighbors)+len(tombstonesToKeep))
			for _, n := range newNeighbors {
				finalConns = append(finalConns, n.Node)
			}
			finalConns = append(finalConns, tombstonesToKeep...)

			node.setConnections(level, finalConns)

			g.shardedLocks[id%uint64(len(g.shardedLocks))].Unlock()
		}

		// Update currID/currDist for next layer
		if candidates.Len() > 0 {
			top, _ := candidates.TopItem()
			currID = top.Node
			currDist = top.Distance
		}
	}
}

// checkRepairNeeded checks if a node needs repair.
// Returns true and a map of layers that need repair.
func (h *HNSW) checkRepairNeeded(g *graph, id uint64) (bool, map[int]bool) {
	g.shardedLocks[id%uint64(len(g.shardedLocks))].RLock()
	defer g.shardedLocks[id%uint64(len(g.shardedLocks))].RUnlock()

	node := h.getNode(g, id)
	if node == nil {
		return false, nil
	}

	tombstones := g.tombstones
	needsRepair := false
	layersToRepair := make(map[int]bool)

	for l := 0; l <= node.Level; l++ {
		conns := node.getConnections(l)
		if len(conns) == 0 {
			continue
		}

		activeCount := 0
		for _, neighborID := range conns {
			if !tombstones.Test(neighborID) {
				activeCount++
			}
		}

		// Check if we need repair
		threshold := h.opts.M / 2
		if l == 0 {
			threshold = h.opts.M // Mmax0 is 2*M, so M is half
		}

		if activeCount < threshold {
			needsRepair = true
			layersToRepair[l] = true
		}
	}

	return needsRepair, layersToRepair
}

// pruneNodeConnections removes deleted neighbors from an active node.
func (h *HNSW) pruneNodeConnections(ctx context.Context, g *graph, id uint64) {
	g.shardedLocks[id%uint64(len(g.shardedLocks))].Lock()
	defer g.shardedLocks[id%uint64(len(g.shardedLocks))].Unlock()

	node := h.getNode(g, id)
	if node == nil {
		return
	}

	tombstones := g.tombstones

	for l := 0; l <= node.Level; l++ {
		conns := node.getConnections(l)
		if len(conns) == 0 {
			continue
		}

		// Check if we have any tombstones
		hasTombstones := false
		for _, neighborID := range conns {
			if tombstones.Test(neighborID) {
				hasTombstones = true
				break
			}
		}

		if hasTombstones {
			// Create a new slice with only active neighbors
			activeConns := make([]uint64, 0, len(conns))
			for _, neighborID := range conns {
				if !tombstones.Test(neighborID) {
					activeConns = append(activeConns, neighborID)
				}
			}

			// Create a new slice to fit perfectly
			newConns := make([]uint64, len(activeConns))
			copy(newConns, activeConns)
			node.setConnections(l, newConns)
		}
	}
}

// clearNodeConnections releases memory for a deleted node's connections.
func (h *HNSW) clearNodeConnections(g *graph, id uint64) {
	g.shardedLocks[id%uint64(len(g.shardedLocks))].Lock()
	defer g.shardedLocks[id%uint64(len(g.shardedLocks))].Unlock()

	node := h.getNode(g, id)
	if node == nil {
		return
	}

	// We can't easily free the node struct itself without re-layout,
	// but we can release the connection slices.
	for l := 0; l <= node.Level; l++ {
		node.setConnections(l, nil)
	}
}
