package hnsw

import (
	"context"
	"runtime"
	"sync"

	"github.com/hupe1980/vecgo/internal/searcher"
	"github.com/hupe1980/vecgo/internal/segment"
	"github.com/hupe1980/vecgo/model"
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
	jobs := make(chan model.RowID, 1024)

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

	for id := model.RowID(0); id < model.RowID(maxID); id++ {
		if ctx.Err() != nil {
			break
		}
		if !h.getNode(g, id).IsZero() {
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
	jobs := make(chan model.RowID, 1024)

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
			if err := h.pruneNodeConnections(ctx, g, id); err != nil {
				return
			}
		}
	}

	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go worker()
	}

	for id := model.RowID(0); id < model.RowID(maxID); id++ {
		if ctx.Err() != nil {
			break
		}
		if !h.getNode(g, id).IsZero() {
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
	jobs := make(chan model.RowID, 1024)

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
			if err := h.clearNodeConnections(ctx, g, id); err != nil {
				return
			}
		}
	}

	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go worker()
	}

	for id := model.RowID(0); id < model.RowID(maxID); id++ {
		if ctx.Err() != nil {
			break
		}
		if !h.getNode(g, id).IsZero() {
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
func (h *HNSW) reconcileNode(ctx context.Context, s *searcher.Searcher, scratch *scratch, g *graph, id model.RowID) {
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
	if node.IsZero() {
		return
	}

	// Create distFunc
	// We use standard distance calculation here because we are not in a tight search loop
	// and we have the raw vector available.
	distFunc := func(nid model.RowID) float32 {
		d, ok := h.vectors.ComputeDistance(nid, vec, h.opts.DistanceType)
		if !ok {
			return 3.402823466e+38 // math.MaxFloat32
		}
		return d
	}

	// Greedy descent to node.Level
	currID, currDist := h.greedyDescent(g, vec, node.Level(g.arena), distFunc)

	// Repair specific layers
	for level := node.Level(g.arena); level >= 0; level-- {
		// Search for candidates
		filter := &excludeFilter{target: uint32(id)}
		// searchLayer populates s.Candidates
		h.searchLayer(s, g, vec, currID, currDist, level, h.opts.EF, filter, 0, distFunc)

		// Use s.Candidates
		candidates := s.Candidates

		// Deduplicate candidates and merge with existing active neighbors
		h.mergeCandidatesWithActiveNeighbors(g, id, level, candidates)

		// If this layer needs repair, update connections
		if layersToRepair[level] {
			if err := h.updateConnectionsForRepair(ctx, g, id, level, candidates, scratch); err != nil {
				return
			}
		}

		// Update entry point for next layer
		// For simplicity, we keep the current best node if we can find a better one in candidates.
		// Since candidates is a MaxHeap (farthest at top), finding the closest requires iteration.
		// Given we are in repair mode, strict greedy descent optimization might be less critical than correctness.
		// We'll stick with the current currID unless we find something obviously better in a cheap way.
		// For now, we just proceed.
	}
}

func (h *HNSW) greedyDescent(g *graph, vec []float32, targetLevel int, distFunc DistFunc) (model.RowID, float32) {
	epID := g.entryPointAtomic.Load()
	maxLevel := int(g.maxLevelAtomic.Load())
	currID := model.RowID(epID)
	currDist := distFunc(currID)

	for level := maxLevel; level > targetLevel; level-- {
		changed := true
		for changed {
			changed = false
			h.visitConnections(g, currID, level, func(neighbor Neighbor) bool {
				nextID := neighbor.ID
				d := distFunc(nextID)
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

func (h *HNSW) mergeCandidatesWithActiveNeighbors(g *graph, id model.RowID, level int, candidates *searcher.PriorityQueue) {
	// Deduplicate candidates and merge with existing active neighbors
	// We use a map to ensure uniqueness.
	uniqueCandidates := make(map[model.RowID]float32, candidates.Len()+h.opts.M)

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
	for nid, dist := range uniqueCandidates {
		candidates.PushItemBounded(searcher.PriorityQueueItem{Node: nid, Distance: dist}, h.opts.EF)
	}
}

func (h *HNSW) updateConnectionsForRepair(ctx context.Context, g *graph, id model.RowID, level int, candidates *searcher.PriorityQueue, scratch *scratch) error {
	// Re-read current connections to identify tombstones
	// We must re-read because we need the exact objects to preserve them
	g.shardedLocks[uint64(id)%uint64(len(g.shardedLocks))].Lock()
	defer g.shardedLocks[uint64(id)%uint64(len(g.shardedLocks))].Unlock()

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

		if err := h.setConnections(ctx, g, id, level, finalNeighbors); err != nil {
			return err
		}
	}
	return nil
}

// checkRepairNeeded checks if a node needs repair.
// Returns true and a map of layers that need repair.
func (h *HNSW) checkRepairNeeded(g *graph, id model.RowID) (bool, map[int]bool) {
	g.shardedLocks[uint64(id)%uint64(len(g.shardedLocks))].RLock()
	defer g.shardedLocks[uint64(id)%uint64(len(g.shardedLocks))].RUnlock()

	node := h.getNode(g, id)
	if node.IsZero() {
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
func (h *HNSW) pruneNodeConnections(ctx context.Context, g *graph, id model.RowID) error {
	g.shardedLocks[uint64(id)%uint64(len(g.shardedLocks))].Lock()
	defer g.shardedLocks[uint64(id)%uint64(len(g.shardedLocks))].Unlock()

	node := h.getNode(g, id)
	if node.IsZero() {
		return nil
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
			if err := h.setConnections(ctx, g, id, l, activeConns); err != nil {
				return err
			}
		}
	}
	return nil
}

// clearNodeConnections releases memory for a deleted node's connections.
func (h *HNSW) clearNodeConnections(ctx context.Context, g *graph, id model.RowID) error {
	g.shardedLocks[uint64(id)%uint64(len(g.shardedLocks))].Lock()
	defer g.shardedLocks[uint64(id)%uint64(len(g.shardedLocks))].Unlock()

	node := h.getNode(g, id)
	if node.IsZero() {
		return nil
	}

	// We can't easily free the node struct itself without re-layout,
	// but we can release the connection slices.
	for l := 0; l <= node.Level(g.arena); l++ {
		if err := h.setConnections(ctx, g, id, l, nil); err != nil {
			return err
		}
	}
	return nil
}

// excludeFilter prevents returning the target node itself.
type excludeFilter struct {
	target uint32
}

func (f *excludeFilter) Matches(id uint32) bool {
	return id != f.target
}

func (f *excludeFilter) MatchesBatch(ids []uint32, out []bool) {
	for i, id := range ids {
		out[i] = (id != f.target)
	}
}

func (f *excludeFilter) AsBitmap() segment.Bitmap {
	return nil
}

func (f *excludeFilter) MatchesBlock(stats map[string]segment.FieldStats) bool {
	return true
}
