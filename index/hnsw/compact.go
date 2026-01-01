package hnsw

import (
	"context"
	"runtime"
	"sync"

	"github.com/hupe1980/vecgo/core"
	"github.com/hupe1980/vecgo/searcher"
)

// Compact removes deleted nodes from the graph connections and attempts to repair connectivity.
// It iterates through all active nodes, removes neighbors that have been deleted,
// and if the number of connections drops too low, it searches for replacement neighbors.
func (h *HNSW) Compact(ctx context.Context) error {
	g := h.currentGraph.Load()
	g.arena.IncRef()
	defer g.arena.DecRef()

	maxID := g.nextIDAtomic.Load()
	tombstones := g.tombstones

	numWorkers := runtime.GOMAXPROCS(0)

	// Phase 1: Repair Active Nodes (Preserving Tombstones)
	// We identify active nodes that would lose too many connections if we pruned tombstones.
	// We find new active neighbors for them and ADD them to the connection list,
	// keeping the tombstones as bridges for now.
	{
		var wg sync.WaitGroup
		jobs := make(chan core.LocalID, 1024)

		worker := func() {
			defer wg.Done()
			s := searcher.Get()
			defer searcher.Put(s)
			scratch := h.scratchPool.Get().(*scratch)
			defer h.scratchPool.Put(scratch)

			for id := range jobs {
				if ctx.Err() != nil {
					return
				}
				// Skip deleted nodes in this phase
				if tombstones.Test(uint32(id)) {
					continue
				}
				// Repair active nodes, keeping tombstones
				h.reconcileNode(s, scratch, ctx, g, id)
			}
		}

		for i := 0; i < numWorkers; i++ {
			wg.Add(1)
			go worker()
		}

		for id := core.LocalID(0); id < core.LocalID(maxID); id++ {
			if ctx.Err() != nil {
				break
			}
			if h.getNode(g, id).Offset != 0 {
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
		jobs := make(chan core.LocalID, 1024)

		worker := func() {
			defer wg.Done()
			for id := range jobs {
				if ctx.Err() != nil {
					return
				}
				// Skip deleted nodes in this phase
				if tombstones.Test(uint32(id)) {
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

		for id := core.LocalID(0); id < core.LocalID(maxID); id++ {
			if ctx.Err() != nil {
				break
			}
			if h.getNode(g, id).Offset != 0 {
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
		jobs := make(chan core.LocalID, 1024)

		worker := func() {
			defer wg.Done()
			for id := range jobs {
				if ctx.Err() != nil {
					return
				}
				// Only process deleted nodes
				if tombstones.Test(uint32(id)) {
					// CRITICAL: Do not clear connections of the entry point, even if deleted.
					// It serves as the gateway to the graph until a new entry point is chosen.
					if uint32(id) == g.entryPointAtomic.Load() {
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

		for id := core.LocalID(0); id < core.LocalID(maxID); id++ {
			if ctx.Err() != nil {
				break
			}
			if h.getNode(g, id).Offset != 0 {
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
func (h *HNSW) reconcileNode(s *searcher.Searcher, scratch *scratch, ctx context.Context, g *graph, id core.LocalID) {
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
	if node.Offset == 0 {
		return
	}

	// Greedy descent to node.Level
	epID := g.entryPointAtomic.Load()
	maxLevel := int(g.maxLevelAtomic.Load())
	currID := core.LocalID(epID)
	currDist := h.dist(vec, currID)

	for level := maxLevel; level > node.Level(g.arena); level-- {
		changed := true
		for changed {
			changed = false
			h.visitConnections(g, currID, level, func(neighbor Neighbor) bool {
				nextID := neighbor.ID
				d := h.dist(vec, nextID)
				if d < currDist {
					currDist = d
					currID = nextID
					changed = true
				}
				return true
			})
		}
	}

	// Repair specific layers
	for level := node.Level(g.arena); level >= 0; level-- {
		// Search for candidates
		filter := func(x core.LocalID) bool { return x != id }
		// searchLayer populates s.Candidates
		h.searchLayer(s, g, vec, currID, currDist, level, h.opts.EF, filter)

		// Use s.Candidates
		candidates := s.Candidates

		// Deduplicate candidates and merge with existing active neighbors
		// We use a map to ensure uniqueness.
		uniqueCandidates := make(map[core.LocalID]float32, candidates.Len()+h.opts.M)

		// Drain candidates from searchLayer
		for candidates.Len() > 0 {
			item, _ := candidates.PopItem()
			uniqueCandidates[item.Node] = item.Distance
		}

		// Add existing active neighbors
		conns := h.getConnections(g, id, level)
		for _, neighbor := range conns {
			if !g.tombstones.Test(uint32(neighbor.ID)) {
				nid := neighbor.ID
				// If already present, keep the one with smaller distance (though they should be same)
				if d, ok := uniqueCandidates[nid]; !ok || neighbor.Dist < d {
					uniqueCandidates[nid] = neighbor.Dist
				}
			}
		}

		// Push unique candidates back to heap
		for id, dist := range uniqueCandidates {
			candidates.PushItemBounded(searcher.PriorityQueueItem{Node: id, Distance: dist}, h.opts.EF)
		}

		var bestNode core.LocalID = currID
		var bestDist float32 = currDist
		foundBest := false

		// If this layer needs repair, update connections
		if layersToRepair[level] {
			// Re-read current connections to identify tombstones
			// We must re-read because we need the exact objects to preserve them
			g.shardedLocks[uint64(id)%uint64(len(g.shardedLocks))].Lock()

			var tombstonesToKeep []Neighbor
			conns := h.getConnections(g, id, level)
			for _, neighbor := range conns {
				if g.tombstones.Test(uint32(neighbor.ID)) {
					tombstonesToKeep = append(tombstonesToKeep, neighbor)
				}
			}

			// Calculate how many new neighbors we can add
			limit := h.maxConnectionsPerLayer
			if level == 0 {
				limit = h.maxConnectionsLayer0
			}

			numToSelect := max(limit-len(tombstonesToKeep), 0)

			// Select best active neighbors
			// selectNeighbors expects MaxHeap (candidates) and drains it
			newNeighbors := h.selectNeighbors(candidates, numToSelect, scratch)

			if len(newNeighbors) > 0 {
				bestNode = newNeighbors[0].Node
				bestDist = newNeighbors[0].Distance
				foundBest = true
			}

			// Combine: New Active + Old Tombstones
			finalConns := make([]Neighbor, 0, len(newNeighbors)+len(tombstonesToKeep))
			for _, n := range newNeighbors {
				finalConns = append(finalConns, Neighbor{ID: core.LocalID(n.Node), Dist: n.Distance})
			}
			finalConns = append(finalConns, tombstonesToKeep...)

			h.setConnections(g, id, level, finalConns)

			g.shardedLocks[uint64(id)%uint64(len(g.shardedLocks))].Unlock()
		} else {
			// Just find the best candidate for next layer
			// selectNeighbors drains candidates
			best := h.selectNeighbors(candidates, 1, scratch)
			if len(best) > 0 {
				bestNode = best[0].Node
				bestDist = best[0].Distance
				foundBest = true
			}
		}

		// Return candidates to pool
		candidates.Reset()

		// Update currID/currDist for next layer
		if foundBest {
			currID = bestNode
			currDist = bestDist
		}
	}
}

// checkRepairNeeded checks if a node needs repair.
// Returns true and a map of layers that need repair.
func (h *HNSW) checkRepairNeeded(g *graph, id core.LocalID) (bool, map[int]bool) {
	g.shardedLocks[uint64(id)%uint64(len(g.shardedLocks))].RLock()
	defer g.shardedLocks[uint64(id)%uint64(len(g.shardedLocks))].RUnlock()

	node := h.getNode(g, id)
	if node.Offset == 0 {
		return false, nil
	}

	tombstones := g.tombstones
	needsRepair := false
	layersToRepair := make(map[int]bool)

	for l := 0; l <= node.Level(g.arena); l++ {
		activeCount := 0
		conns := h.getConnections(g, id, l)
		for _, neighbor := range conns {
			if !tombstones.Test(uint32(neighbor.ID)) {
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
func (h *HNSW) pruneNodeConnections(ctx context.Context, g *graph, id core.LocalID) {
	g.shardedLocks[uint64(id)%uint64(len(g.shardedLocks))].Lock()
	defer g.shardedLocks[uint64(id)%uint64(len(g.shardedLocks))].Unlock()

	node := h.getNode(g, id)
	if node.Offset == 0 {
		return
	}

	tombstones := g.tombstones

	for l := 0; l <= node.Level(g.arena); l++ {
		// Check if we have any tombstones
		hasTombstones := false
		var activeConns []Neighbor
		conns := h.getConnections(g, id, l)
		for _, neighbor := range conns {
			if tombstones.Test(uint32(neighbor.ID)) {
				hasTombstones = true
			} else {
				activeConns = append(activeConns, neighbor)
			}
		}

		if hasTombstones {
			h.setConnections(g, id, l, activeConns)
		}
	}
}

// clearNodeConnections releases memory for a deleted node's connections.
func (h *HNSW) clearNodeConnections(g *graph, id core.LocalID) {
	g.shardedLocks[uint64(id)%uint64(len(g.shardedLocks))].Lock()
	defer g.shardedLocks[uint64(id)%uint64(len(g.shardedLocks))].Unlock()

	node := h.getNode(g, id)
	if node.Offset == 0 {
		return
	}

	// We can't easily free the node struct itself without re-layout,
	// but we can release the connection slices.
	for l := 0; l <= node.Level(g.arena); l++ {
		h.setConnections(g, id, l, nil)
	}
}
