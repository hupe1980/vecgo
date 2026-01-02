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

	if err := h.repairActiveNodes(ctx, g); err != nil {
		return err
	}

	if err := h.pruneActiveNodes(ctx, g); err != nil {
		return err
	}

	if err := h.clearDeletedNodes(ctx, g); err != nil {
		return err
	}

	return nil
}

func (h *HNSW) repairActiveNodes(ctx context.Context, g *graph) error {
	maxID := g.nextIDAtomic.Load()
	tombstones := g.tombstones
	numWorkers := runtime.GOMAXPROCS(0)

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
			h.reconcileNode(ctx, s, scratch, g, id)
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

	return ctx.Err()
}

func (h *HNSW) pruneActiveNodes(ctx context.Context, g *graph) error {
	maxID := g.nextIDAtomic.Load()
	tombstones := g.tombstones
	numWorkers := runtime.GOMAXPROCS(0)

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
			h.pruneNodeConnections(g, id)
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

	return ctx.Err()
}

func (h *HNSW) clearDeletedNodes(ctx context.Context, g *graph) error {
	maxID := g.nextIDAtomic.Load()
	tombstones := g.tombstones
	numWorkers := runtime.GOMAXPROCS(0)

	var wg sync.WaitGroup
	jobs := make(chan core.LocalID, 1024)

	worker := func() {
		defer wg.Done()
		for id := range jobs {
			if ctx.Err() != nil {
				return
			}
			// Only process deleted nodes
			if !tombstones.Test(uint32(id)) {
				continue
			}
			// Clear connections of deleted nodes
			h.clearNodeConnections(g, id)
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

	return ctx.Err()
}

// reconcileNode checks if an active node needs repair (due to deleted neighbors).
// If so, it finds new active neighbors and appends them to the connection list,
// preserving the deleted neighbors as bridges.
func (h *HNSW) reconcileNode(ctx context.Context, s *searcher.Searcher, scratch *scratch, g *graph, id core.LocalID) {
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
	if node == nil || node.Offset == 0 {
		return
	}

	// Greedy descent to node.Level
	currID, currDist := h.greedyDescent(g, vec, node.Level(g.arena))

	// Repair specific layers
	for level := node.Level(g.arena); level >= 0; level-- {
		// Search for candidates
		filter := func(x core.LocalID) bool { return x != id }
		// searchLayer populates s.Candidates
		h.searchLayer(s, g, vec, currID, currDist, level, h.opts.EF, filter)

		// Use s.Candidates
		candidates := s.Candidates

		// Deduplicate candidates and merge with existing active neighbors
		h.mergeCandidatesWithActiveNeighbors(g, id, level, candidates)

		// If this layer needs repair, update connections
		if layersToRepair[level] {
			h.updateConnectionsForRepair(g, id, level, candidates, scratch)
		}

		// Update entry point for next layer
		// For simplicity, we keep the current best node if we can find a better one in candidates.
		// Since candidates is a MaxHeap (farthest at top), finding the closest requires iteration.
		// Given we are in repair mode, strict greedy descent optimization might be less critical than correctness.
		// We'll stick with the current currID unless we find something obviously better in a cheap way.
		// For now, we just proceed.
	}
}

func (h *HNSW) greedyDescent(g *graph, vec []float32, targetLevel int) (core.LocalID, float32) {
	epID := g.entryPointAtomic.Load()
	maxLevel := int(g.maxLevelAtomic.Load())
	currID := core.LocalID(epID)
	currDist := h.dist(vec, currID)

	for level := maxLevel; level > targetLevel; level-- {
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
	return currID, currDist
}

func (h *HNSW) mergeCandidatesWithActiveNeighbors(g *graph, id core.LocalID, level int, candidates *searcher.PriorityQueue) {
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
}

func (h *HNSW) updateConnectionsForRepair(g *graph, id core.LocalID, level int, candidates *searcher.PriorityQueue, scratch *scratch) {
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
		// Merge tombstones + new active neighbors
		finalNeighbors := make([]Neighbor, 0, len(tombstonesToKeep)+len(newNeighbors))
		finalNeighbors = append(finalNeighbors, tombstonesToKeep...)
		for _, n := range newNeighbors {
			finalNeighbors = append(finalNeighbors, Neighbor{ID: n.Node, Dist: n.Distance})
		}

		h.setConnections(g, id, level, finalNeighbors)
	}
	g.shardedLocks[uint64(id)%uint64(len(g.shardedLocks))].Unlock()
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
func (h *HNSW) pruneNodeConnections(g *graph, id core.LocalID) {
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
