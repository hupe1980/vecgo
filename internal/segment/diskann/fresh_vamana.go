package diskann

import (
	"context"
	"errors"
	"fmt"
	"math/rand"
	"runtime"
	"sort"
	"sync"
	"sync/atomic"

	"github.com/hupe1980/vecgo/distance"
	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/model"
)

const (
	// FreshDefaultR is the default max out-degree for FreshVamana.
	FreshDefaultR = 64
	// FreshDefaultL is the default search list size.
	FreshDefaultL = 100
	// FreshDefaultAlpha is the default pruning factor.
	FreshDefaultAlpha = 1.2
	// consolidationThreshold triggers cleanup when deletion ratio exceeds this.
	consolidationThreshold = 0.1
)

// FreshVamana is a mutable Vamana graph supporting streaming updates.
// It implements the FreshDiskANN algorithm for incremental graph maintenance.
//
// Thread-safety:
//   - Concurrent reads are lock-free via atomic pointers
//   - Writes are serialized via growMu
//   - Insert and Delete can run concurrently with Search
type FreshVamana struct {
	// Configuration (immutable after construction)
	dim    int
	metric distance.Metric
	r      int     // Max out-degree
	l      int     // Search list size
	alpha  float32 // Pruning factor

	// Data storage - wrapped in atomic pointer for lock-free reads
	data atomic.Pointer[freshVamanaData]

	// Entry point (updated atomically)
	entryPoint atomic.Uint32

	// ID→nodeID index for O(1) delete lookups (protected by idIndexMu)
	idIndex   map[model.ID]uint32
	idIndexMu sync.RWMutex

	// Deletion bitmap
	deleted      []uint64 // Bitmap words
	deletedMu    sync.RWMutex
	deletedCount atomic.Int64

	// Size tracking
	nodeCount atomic.Int64

	// Distance function
	distFunc distance.Func

	// Synchronization
	growMu   sync.Mutex // Protects data structure growth and all writes
	closedMu sync.RWMutex
	closed   bool

	// Consolidation state
	consolidating atomic.Bool
}

// freshVamanaData holds the growable data structures.
// This is swapped atomically during growth for lock-free reads.
type freshVamanaData struct {
	vectors   [][]float32
	ids       []model.ID
	metas     []metadata.Document
	neighbors []atomic.Pointer[[]uint32]
}

// FreshVamanaOptions configures FreshVamana.
type FreshVamanaOptions struct {
	R           int     // Max out-degree (default: 64)
	L           int     // Search list size (default: 100)
	Alpha       float32 // Pruning factor (default: 1.2)
	InitialSize int     // Initial capacity (default: 1024)
}

// DefaultFreshVamanaOptions returns sensible defaults.
func DefaultFreshVamanaOptions() FreshVamanaOptions {
	return FreshVamanaOptions{
		R:           FreshDefaultR,
		L:           FreshDefaultL,
		Alpha:       FreshDefaultAlpha,
		InitialSize: 1024,
	}
}

// NewFreshVamana creates a new FreshVamana graph.
func NewFreshVamana(dim int, metric distance.Metric, opts FreshVamanaOptions) (*FreshVamana, error) {
	if dim <= 0 {
		return nil, errors.New("dimension must be positive")
	}

	if opts.R == 0 {
		opts.R = FreshDefaultR
	}
	if opts.L == 0 {
		opts.L = FreshDefaultL
	}
	if opts.Alpha == 0 {
		opts.Alpha = FreshDefaultAlpha
	}
	if opts.InitialSize == 0 {
		opts.InitialSize = 1024
	}

	distFunc, err := distance.Provider(metric)
	if err != nil {
		return nil, fmt.Errorf("invalid metric: %w", err)
	}

	// Initialize data container
	data := &freshVamanaData{
		vectors:   make([][]float32, 0, opts.InitialSize),
		ids:       make([]model.ID, 0, opts.InitialSize),
		metas:     make([]metadata.Document, 0, opts.InitialSize),
		neighbors: make([]atomic.Pointer[[]uint32], 0, opts.InitialSize),
	}

	fv := &FreshVamana{
		dim:      dim,
		metric:   metric,
		r:        opts.R,
		l:        opts.L,
		alpha:    opts.Alpha,
		distFunc: distFunc,
		deleted:  make([]uint64, (opts.InitialSize+63)/64),
		idIndex:  make(map[model.ID]uint32, opts.InitialSize),
	}
	fv.data.Store(data)

	return fv, nil
}

// Len returns the number of active (non-deleted) nodes.
func (fv *FreshVamana) Len() int {
	return int(fv.nodeCount.Load() - fv.deletedCount.Load())
}

// TotalLen returns total nodes including deleted.
func (fv *FreshVamana) TotalLen() int {
	return int(fv.nodeCount.Load())
}

// DeletedRatio returns the fraction of deleted nodes.
func (fv *FreshVamana) DeletedRatio() float64 {
	total := fv.nodeCount.Load()
	if total == 0 {
		return 0
	}
	return float64(fv.deletedCount.Load()) / float64(total)
}

// Insert adds a new vector to the graph using FreshDiskANN algorithm.
//
// The algorithm:
// 1. Append vector to data store (assign node ID)
// 2. Search graph from entry point to find candidates
// 3. Select neighbors using RobustPrune
// 4. Add bidirectional edges
// 5. Optionally update entry point
//
// Note: Insert operations are serialized via growMu to ensure thread-safety
// during graph modifications. Concurrent reads (Search) remain lock-free.
func (fv *FreshVamana) Insert(ctx context.Context, id model.ID, vec []float32, md metadata.Document) (uint32, error) {
	fv.closedMu.RLock()
	if fv.closed {
		fv.closedMu.RUnlock()
		return 0, errors.New("FreshVamana is closed")
	}
	fv.closedMu.RUnlock()

	if len(vec) != fv.dim {
		return 0, fmt.Errorf("dimension mismatch: expected %d, got %d", fv.dim, len(vec))
	}

	// Lock for entire insert operation (serialized writes, concurrent reads)
	fv.growMu.Lock()
	defer fv.growMu.Unlock()

	// Step 1: Allocate node and append data
	nodeID := fv.allocateNodeLocked(id, vec, md)

	// Step 2: First node becomes entry point with no neighbors
	if nodeID == 0 {
		fv.entryPoint.Store(0)
		return 0, nil
	}

	// Step 3: Search for candidate neighbors (read-only, safe with concurrent reads)
	ep := fv.entryPoint.Load()
	candidates := fv.searchCandidatesLocked(ctx, vec, ep, fv.l)

	// Step 4: Select neighbors using RobustPrune
	selectedNeighbors := fv.robustPruneLocked(nodeID, candidates, fv.r, fv.alpha)

	// Step 5: Set outgoing edges
	fv.setNeighborsLocked(nodeID, selectedNeighbors)

	// Step 6: Add reverse edges (bidirectional graph)
	for _, neighbor := range selectedNeighbors {
		fv.addReverseEdgeLocked(ctx, neighbor, nodeID)
	}

	// Step 7: Maybe update entry point (simple heuristic)
	fv.maybeUpdateEntryPoint(nodeID)

	return nodeID, nil
}

// Delete marks a node as deleted (soft delete).
// The node remains in graph for navigation but is excluded from results.
func (fv *FreshVamana) Delete(id model.ID) error {
	fv.closedMu.RLock()
	if fv.closed {
		fv.closedMu.RUnlock()
		return errors.New("FreshVamana is closed")
	}
	fv.closedMu.RUnlock()

	// O(1) lookup via ID index
	fv.idIndexMu.RLock()
	nodeID, exists := fv.idIndex[id]
	fv.idIndexMu.RUnlock()

	if !exists {
		return fmt.Errorf("ID not found: %d", id)
	}

	// Set deletion bit
	wordIdx := int(nodeID) / 64
	bitIdx := uint(nodeID % 64)

	fv.deletedMu.Lock()
	if wordIdx >= len(fv.deleted) {
		fv.deletedMu.Unlock()
		return errors.New("invalid node ID")
	}
	if (fv.deleted[wordIdx] & (1 << bitIdx)) != 0 {
		fv.deletedMu.Unlock()
		return nil // Already deleted
	}
	fv.deleted[wordIdx] |= (1 << bitIdx)
	fv.deletedMu.Unlock()

	fv.deletedCount.Add(1)

	// Trigger consolidation if threshold exceeded
	if fv.DeletedRatio() > consolidationThreshold {
		if fv.consolidating.CompareAndSwap(false, true) {
			go fv.consolidate(context.Background())
		}
	}

	return nil
}

// Search performs k-NN search using greedy best-first traversal.
func (fv *FreshVamana) Search(ctx context.Context, query []float32, k int) ([]SearchResult, error) {
	fv.closedMu.RLock()
	if fv.closed {
		fv.closedMu.RUnlock()
		return nil, errors.New("FreshVamana is closed")
	}
	fv.closedMu.RUnlock()

	if fv.nodeCount.Load() == 0 {
		return nil, nil
	}

	ef := k * 2
	if ef < fv.l {
		ef = fv.l
	}

	ep := fv.entryPoint.Load()
	candidates := fv.greedySearch(ctx, query, ep, ef)

	// Get data snapshot for consistent reads
	data := fv.data.Load()

	// Filter deleted and take top k
	results := make([]SearchResult, 0, k)
	for _, c := range candidates {
		if fv.isDeleted(c.nodeID) {
			continue
		}
		if data == nil || int(c.nodeID) >= len(data.ids) {
			continue
		}
		results = append(results, SearchResult{
			NodeID:   c.nodeID,
			ID:       data.ids[c.nodeID],
			Distance: c.dist,
		})
		if len(results) >= k {
			break
		}
	}

	return results, nil
}

// SearchWithFilter performs filtered k-NN search.
func (fv *FreshVamana) SearchWithFilter(ctx context.Context, query []float32, k int, filter func(model.ID) bool) ([]SearchResult, error) {
	fv.closedMu.RLock()
	if fv.closed {
		fv.closedMu.RUnlock()
		return nil, errors.New("FreshVamana is closed")
	}
	fv.closedMu.RUnlock()

	if fv.nodeCount.Load() == 0 {
		return nil, nil
	}

	// Increase search breadth for filtered search
	ef := k * 10
	if ef < fv.l*2 {
		ef = fv.l * 2
	}

	ep := fv.entryPoint.Load()
	candidates := fv.greedySearch(ctx, query, ep, ef)

	// Get data snapshot for consistent reads
	data := fv.data.Load()

	results := make([]SearchResult, 0, k)
	for _, c := range candidates {
		if fv.isDeleted(c.nodeID) {
			continue
		}
		if data == nil || int(c.nodeID) >= len(data.ids) {
			continue
		}
		if filter != nil && !filter(data.ids[c.nodeID]) {
			continue
		}
		results = append(results, SearchResult{
			NodeID:   c.nodeID,
			ID:       data.ids[c.nodeID],
			Distance: c.dist,
		})
		if len(results) >= k {
			break
		}
	}

	return results, nil
}

// GetVector returns the vector for a node.
func (fv *FreshVamana) GetVector(nodeID uint32) ([]float32, bool) {
	if int64(nodeID) >= fv.nodeCount.Load() {
		return nil, false
	}
	data := fv.data.Load()
	if data == nil || int(nodeID) >= len(data.vectors) {
		return nil, false
	}
	return data.vectors[nodeID], true
}

// GetID returns the external ID for a node.
func (fv *FreshVamana) GetID(nodeID uint32) (model.ID, bool) {
	if int64(nodeID) >= fv.nodeCount.Load() {
		return 0, false
	}
	data := fv.data.Load()
	if data == nil || int(nodeID) >= len(data.ids) {
		return 0, false
	}
	return data.ids[nodeID], true
}

// GetMetadata returns metadata for a node.
func (fv *FreshVamana) GetMetadata(nodeID uint32) (metadata.Document, bool) {
	if int64(nodeID) >= fv.nodeCount.Load() {
		return nil, false
	}
	data := fv.data.Load()
	if data == nil || int(nodeID) >= len(data.metas) {
		return nil, false
	}
	return data.metas[nodeID], true
}

// Close releases resources.
func (fv *FreshVamana) Close() error {
	fv.closedMu.Lock()
	if fv.closed {
		fv.closedMu.Unlock()
		return nil
	}
	fv.closed = true
	fv.closedMu.Unlock()

	// Wait for consolidation to finish with backoff
	for fv.consolidating.Load() {
		// Yield the processor to avoid busy-spinning
		runtime.Gosched()
	}

	// Clear data under lock (allow GC)
	fv.growMu.Lock()
	fv.data.Store(nil)
	fv.growMu.Unlock()

	fv.deletedMu.Lock()
	fv.deleted = nil
	fv.deletedMu.Unlock()

	fv.idIndexMu.Lock()
	fv.idIndex = nil
	fv.idIndexMu.Unlock()

	return nil
}

// SearchResult represents a search result.
type SearchResult struct {
	NodeID   uint32
	ID       model.ID
	Distance float32
}

// --- Internal methods ---

// allocateNodeLocked appends a new node. Caller must hold growMu.
func (fv *FreshVamana) allocateNodeLocked(id model.ID, vec []float32, md metadata.Document) uint32 {
	nodeID := uint32(fv.nodeCount.Add(1) - 1)

	// Get current data and ensure capacity
	data := fv.data.Load()
	data = fv.ensureCapacityLocked(data, int(nodeID)+1)

	// Copy vector (take ownership)
	v := make([]float32, len(vec))
	copy(v, vec)

	data.vectors[nodeID] = v
	data.ids[nodeID] = id
	data.metas[nodeID] = md

	// Update ID→nodeID index for O(1) lookups
	fv.idIndexMu.Lock()
	fv.idIndex[id] = nodeID
	fv.idIndexMu.Unlock()

	// Initialize empty neighbor list
	emptyNeighbors := make([]uint32, 0, fv.r)
	data.neighbors[nodeID].Store(&emptyNeighbors)

	return nodeID
}

// ensureCapacityLocked grows data structures and atomically publishes new snapshot.
// Caller must hold growMu. Returns the (potentially new) data pointer.
func (fv *FreshVamana) ensureCapacityLocked(data *freshVamanaData, size int) *freshVamanaData {
	if len(data.vectors) >= size {
		return data
	}

	newCap := max(size, len(data.vectors)*2)
	if newCap < 16 {
		newCap = 16
	}

	// Create new data snapshot with grown slices
	newData := &freshVamanaData{
		vectors:   make([][]float32, newCap),
		ids:       make([]model.ID, newCap),
		metas:     make([]metadata.Document, newCap),
		neighbors: make([]atomic.Pointer[[]uint32], newCap),
	}

	// Copy existing data
	copy(newData.vectors, data.vectors)
	copy(newData.ids, data.ids)
	copy(newData.metas, data.metas)

	// Copy atomic pointers (their stored values remain valid)
	for i := 0; i < len(data.neighbors); i++ {
		ptr := data.neighbors[i].Load()
		if ptr != nil {
			newData.neighbors[i].Store(ptr)
		}
	}

	// Atomically publish new data - readers will see consistent snapshot
	fv.data.Store(newData)

	// Grow deletion bitmap under its own lock
	requiredWords := (newCap + 63) / 64
	fv.deletedMu.Lock()
	if len(fv.deleted) < requiredWords {
		newDeleted := make([]uint64, requiredWords)
		copy(newDeleted, fv.deleted)
		fv.deleted = newDeleted
	}
	fv.deletedMu.Unlock()

	return newData
}

// isDeleted checks if a node is marked deleted.
func (fv *FreshVamana) isDeleted(nodeID uint32) bool {
	wordIdx := int(nodeID / 64)
	bitIdx := uint(nodeID % 64)

	fv.deletedMu.RLock()
	defer fv.deletedMu.RUnlock()

	if wordIdx >= len(fv.deleted) {
		return false
	}
	return (fv.deleted[wordIdx] & (1 << bitIdx)) != 0
}

// greedySearch performs greedy best-first search using lock-free snapshots.
func (fv *FreshVamana) greedySearch(ctx context.Context, query []float32, ep uint32, ef int) []searchCandidate {
	// Check closed state once at the start
	fv.closedMu.RLock()
	if fv.closed {
		fv.closedMu.RUnlock()
		return nil
	}
	fv.closedMu.RUnlock()

	visited := make(map[uint32]bool, ef*2)
	candidates := make([]searchCandidate, 0, ef)
	results := make([]searchCandidate, 0, ef)

	// Start from entry point
	epVec, ok := fv.getVectorSafe(ep)
	if !ok {
		return nil
	}
	startDist := fv.distFunc(query, epVec)
	candidates = append(candidates, searchCandidate{nodeID: ep, dist: startDist})
	results = append(results, searchCandidate{nodeID: ep, dist: startDist})
	visited[ep] = true

	// Counter for periodic closed state check
	iterations := 0

	for len(candidates) > 0 {
		// Periodic checks every 64 iterations
		iterations++
		if iterations&63 == 0 {
			select {
			case <-ctx.Done():
				return results
			default:
			}

			// Periodic closed state check
			fv.closedMu.RLock()
			closed := fv.closed
			fv.closedMu.RUnlock()
			if closed {
				return results
			}
		}

		// Pop closest (min-heap simulation with sorted slice)
		closest := candidates[0]
		candidates = candidates[1:]

		// Stop if closest is worse than worst result
		if len(results) >= ef && closest.dist > results[len(results)-1].dist {
			break
		}

		// Expand neighbors using lock-free snapshot access
		neighbors := fv.getNeighborsSafe(closest.nodeID)
		for _, n := range neighbors {
			if visited[n] {
				continue
			}
			visited[n] = true

			nVec, ok := fv.getVectorSafe(n)
			if !ok {
				continue
			}
			dist := fv.distFunc(query, nVec)
			c := searchCandidate{nodeID: n, dist: dist}

			// Insert into candidates (sorted)
			candidates = insertCandidate(candidates, c, ef*2)

			// Insert into results
			results = insertCandidate(results, c, ef)
		}
	}

	return results
}

// searchCandidatesLocked searches for insert candidates. Caller must hold growMu.
func (fv *FreshVamana) searchCandidatesLocked(ctx context.Context, vec []float32, ep uint32, ef int) []searchCandidate {
	data := fv.data.Load()
	visited := make(map[uint32]bool, ef*2)
	candidates := make([]searchCandidate, 0, ef)
	results := make([]searchCandidate, 0, ef)

	if int(ep) >= len(data.vectors) {
		return nil
	}
	startDist := fv.distFunc(vec, data.vectors[ep])
	candidates = append(candidates, searchCandidate{nodeID: ep, dist: startDist})
	if !fv.isDeleted(ep) {
		results = append(results, searchCandidate{nodeID: ep, dist: startDist})
	}
	visited[ep] = true

	for len(candidates) > 0 {
		select {
		case <-ctx.Done():
			return results
		default:
		}

		closest := candidates[0]
		candidates = candidates[1:]

		if len(results) >= ef && closest.dist > results[len(results)-1].dist {
			break
		}

		neighborsPtr := data.neighbors[closest.nodeID].Load()
		if neighborsPtr == nil {
			continue
		}
		for _, n := range *neighborsPtr {
			if visited[n] {
				continue
			}
			visited[n] = true

			if int(n) >= len(data.vectors) {
				continue
			}
			dist := fv.distFunc(vec, data.vectors[n])
			c := searchCandidate{nodeID: n, dist: dist}

			candidates = insertCandidate(candidates, c, ef*2)

			if !fv.isDeleted(n) {
				results = insertCandidate(results, c, ef)
			}
		}
	}

	return results
}

// getNeighbors returns a node's neighbors (lock-free read).
func (fv *FreshVamana) getNeighbors(nodeID uint32) []uint32 {
	data := fv.data.Load()
	if data == nil || int(nodeID) >= len(data.neighbors) {
		return nil
	}
	ptr := data.neighbors[nodeID].Load()
	if ptr == nil {
		return nil
	}
	return *ptr
}

// setNeighborsLocked atomically sets a node's neighbors. Caller must hold growMu.
func (fv *FreshVamana) setNeighborsLocked(nodeID uint32, neighbors []uint32) {
	data := fv.data.Load()
	if data == nil || int(nodeID) >= len(data.neighbors) {
		return
	}
	newNeighbors := make([]uint32, len(neighbors))
	copy(newNeighbors, neighbors)
	data.neighbors[nodeID].Store(&newNeighbors)
}

// addReverseEdgeLocked adds nodeID to target's neighbor list. Caller must hold growMu.
func (fv *FreshVamana) addReverseEdgeLocked(ctx context.Context, target, nodeID uint32) {
	data := fv.data.Load()
	if data == nil || int(target) >= len(data.neighbors) {
		return
	}

	current := fv.getNeighbors(target)

	// Already connected?
	for _, n := range current {
		if n == nodeID {
			return
		}
	}

	// Under capacity - just add
	if len(current) < fv.r {
		newNeighbors := make([]uint32, len(current)+1)
		copy(newNeighbors, current)
		newNeighbors[len(current)] = nodeID
		data.neighbors[target].Store(&newNeighbors)
		return
	}

	// Need to prune - create candidate list
	candidates := make([]searchCandidate, 0, len(current)+1)
	targetVec := data.vectors[target]
	for _, n := range current {
		if int(n) >= len(data.vectors) {
			continue
		}
		dist := fv.distFunc(targetVec, data.vectors[n])
		candidates = append(candidates, searchCandidate{nodeID: n, dist: dist})
	}
	dist := fv.distFunc(targetVec, data.vectors[nodeID])
	candidates = append(candidates, searchCandidate{nodeID: nodeID, dist: dist})

	// Sort by distance
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].dist < candidates[j].dist
	})

	// Prune to R best
	pruned := fv.robustPruneLocked(target, candidates, fv.r, fv.alpha)
	newNeighbors := make([]uint32, len(pruned))
	copy(newNeighbors, pruned)
	data.neighbors[target].Store(&newNeighbors)
}

// robustPruneLocked implements Vamana's RobustPrune. Caller must hold growMu.
func (fv *FreshVamana) robustPruneLocked(nodeID uint32, candidates []searchCandidate, r int, alpha float32) []uint32 {
	if len(candidates) == 0 {
		return nil
	}

	data := fv.data.Load()

	// Sort by distance
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].dist < candidates[j].dist
	})

	result := make([]uint32, 0, r)

	for _, c := range candidates {
		if len(result) >= r {
			break
		}
		if c.nodeID == nodeID {
			continue
		}
		if fv.isDeleted(c.nodeID) {
			continue
		}

		// Check α-domination
		dominated := false
		for _, s := range result {
			if int(c.nodeID) >= len(data.vectors) || int(s) >= len(data.vectors) {
				continue
			}
			distCS := fv.distFunc(data.vectors[c.nodeID], data.vectors[s])
			if distCS < alpha*c.dist {
				dominated = true
				break
			}
		}

		if !dominated {
			result = append(result, c.nodeID)
		}
	}

	return result
}

// maybeUpdateEntryPoint updates entry point heuristically.
func (fv *FreshVamana) maybeUpdateEntryPoint(nodeID uint32) {
	// Simple heuristic: random updates for small graphs
	count := fv.nodeCount.Load()
	if count < 100 || (count%500 == 0 && rand.Float32() < 0.1) {
		fv.entryPoint.Store(nodeID)
	}
}

// consolidate cleans up deleted nodes and repairs edges.
func (fv *FreshVamana) consolidate(ctx context.Context) {
	defer fv.consolidating.Store(false)

	// Check closed state
	fv.closedMu.RLock()
	if fv.closed {
		fv.closedMu.RUnlock()
		return
	}
	fv.closedMu.RUnlock()

	count := int(fv.nodeCount.Load())
	if count == 0 {
		return
	}

	// Find nodes with deleted neighbors and repair
	for i := 0; i < count; i++ {
		select {
		case <-ctx.Done():
			return
		default:
		}

		// Check closed state again
		fv.closedMu.RLock()
		if fv.closed {
			fv.closedMu.RUnlock()
			return
		}
		fv.closedMu.RUnlock()

		if fv.isDeleted(uint32(i)) {
			continue
		}

		neighbors := fv.getNeighbors(uint32(i))
		needsRepair := false
		for _, n := range neighbors {
			if fv.isDeleted(n) {
				needsRepair = true
				break
			}
		}

		if needsRepair {
			// Acquire growMu for safe modification
			fv.growMu.Lock()

			data := fv.data.Load()
			if data == nil || i >= len(data.vectors) {
				fv.growMu.Unlock()
				continue
			}

			vec := data.vectors[i]
			candidates := fv.searchCandidatesLocked(ctx, vec, fv.entryPoint.Load(), fv.l)
			newNeighbors := fv.robustPruneLocked(uint32(i), candidates, fv.r, fv.alpha)
			fv.setNeighborsLocked(uint32(i), newNeighbors)

			fv.growMu.Unlock()
		}
	}
}

// getVectorSafe gets a vector using lock-free atomic snapshot.
func (fv *FreshVamana) getVectorSafe(nodeID uint32) ([]float32, bool) {
	data := fv.data.Load()
	if data == nil || int(nodeID) >= len(data.vectors) {
		return nil, false
	}
	return data.vectors[nodeID], true
}

// getNeighborsSafe gets neighbors using lock-free atomic snapshot.
func (fv *FreshVamana) getNeighborsSafe(nodeID uint32) []uint32 {
	data := fv.data.Load()
	if data == nil || int(nodeID) >= len(data.neighbors) {
		return nil
	}
	ptr := data.neighbors[nodeID].Load()
	if ptr == nil {
		return nil
	}
	return *ptr
}

// searchCandidate is an internal search candidate.
type searchCandidate struct {
	nodeID uint32
	dist   float32
}

// insertCandidate inserts into a sorted slice, maintaining max size.
func insertCandidate(slice []searchCandidate, c searchCandidate, maxSize int) []searchCandidate {
	// Find insertion point (binary search for larger slices)
	i := sort.Search(len(slice), func(j int) bool {
		return slice[j].dist > c.dist
	})

	if i >= maxSize {
		return slice
	}

	// Insert
	if len(slice) < maxSize {
		slice = append(slice, searchCandidate{})
	}
	if i < len(slice)-1 {
		copy(slice[i+1:], slice[i:])
	}
	slice[i] = c

	// Trim
	if len(slice) > maxSize {
		slice = slice[:maxSize]
	}

	return slice
}
