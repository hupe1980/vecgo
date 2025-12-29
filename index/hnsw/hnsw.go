// Package hnsw implements the Hierarchical Navigable Small World (HNSW) graph for approximate nearest neighbor search.
package hnsw

import (
	"context"
	"fmt"
	"iter"
	"math"
	"math/rand"
	"sync"
	"sync/atomic"

	"github.com/bits-and-blooms/bitset"
	"github.com/hupe1980/vecgo/distance"
	"github.com/hupe1980/vecgo/index"
	"github.com/hupe1980/vecgo/internal/arena"
	"github.com/hupe1980/vecgo/internal/queue"
	"github.com/hupe1980/vecgo/vectorstore"
	"github.com/hupe1980/vecgo/vectorstore/columnar"
)

const (
	// layerNormalizationBase is the base constant for exponential layer probability distribution.
	layerNormalizationBase = 1.0

	// mmax0Multiplier is the multiplier for calculating maximum connections at layer 0.
	mmax0Multiplier = 2

	// minimumM is the minimum valid value for M.
	minimumM = 2

	// DefaultM is the default number of bidirectional links.
	DefaultM = 8

	// DefaultEF is the default size of the dynamic candidate list.
	DefaultEF = 200

	// nodeSegmentSize is the size of each node segment (65536).
	// Using segments avoids copying the entire node array during growth.
	nodeSegmentBits = 16
	nodeSegmentSize = 1 << nodeSegmentBits
	nodeSegmentMask = nodeSegmentSize - 1
)

// Compile-time checks
var _ index.Index = (*HNSW)(nil)
var _ index.TransactionalIndex = (*HNSW)(nil)

// Node represents a node in the HNSW graph with lock-free reads via RCU.
// Cache-line aligned to 128 bytes.
type Node struct {
	// === CACHE LINE 1: Hot read path (64 bytes) ===

	// Connections stores links to other nodes at each layer.
	// Lock-free reads via atomic.Pointer, copy-on-write updates.
	Connections []atomic.Pointer[[]uint32]

	// ID is the unique identifier for this node.
	ID uint32

	// Layer is the highest layer this node appears in.
	Layer int

	// Padding to align to 64 bytes
	_ [4]byte

	// === CACHE LINE 2: Cold write path (64 bytes) ===

	// mu protects Connections during WRITE operations only.
	mu sync.Mutex

	// Padding to complete second cache line
	_ [56]byte
}

// Options represents the options for configuring HNSW.
type Options struct {
	Dimension        int
	M                int
	EF               int
	Heuristic        bool
	DistanceType     index.DistanceType
	NormalizeVectors bool
	Vectors          vectorstore.Store
}

var DefaultOptions = Options{
	Dimension:    0,
	M:            DefaultM,
	EF:           DefaultEF,
	Heuristic:    true,
	DistanceType: index.DistanceTypeSquaredL2,
}

// HNSW represents the Hierarchical Navigable Small World graph.
type HNSW struct {
	// Hot path: Atomic fields
	entryPointAtomic atomic.Uint32
	maxLevelAtomic   atomic.Int32
	dimensionAtomic  atomic.Int32
	nextIDAtomic     atomic.Uint32
	countAtomic      atomic.Int64 // Track live count

	// Node storage: Segmented array to avoid global locks/copying on grow
	// segments[i] contains nodes [i*segmentSize, (i+1)*segmentSize)
	segments   []atomic.Pointer[[]*Node]
	segmentsMu sync.RWMutex

	// Components
	distanceFunc index.DistanceFunc
	vectors      vectorstore.Store

	// Configuration
	maxConnectionsPerLayer int
	maxConnectionsLayer0   int
	layerMultiplier        float64
	opts                   Options

	// Sharding
	shardID   int
	numShards int

	// Locks
	epMu       sync.RWMutex // Protects entryPoint and maxLevel updates
	freeListMu sync.Mutex   // Protects freeList

	// State
	freeList []uint32

	// Resources
	arena        *arena.Arena
	minQueuePool *sync.Pool
	maxQueuePool *sync.Pool
	bitsetPool   *sync.Pool
}

func (*HNSW) Name() string { return "HNSW" }

// New creates a new HNSW instance.
func New(optFns ...func(o *Options)) (*HNSW, error) {
	opts := DefaultOptions
	for _, fn := range optFns {
		fn(&opts)
	}

	if err := index.ValidateBasicOptions(opts.Dimension, opts.DistanceType); err != nil {
		return nil, err
	}

	if opts.DistanceType == index.DistanceTypeCosine {
		opts.NormalizeVectors = true
	}

	if opts.M < minimumM {
		opts.M = minimumM
	}

	h := &HNSW{
		maxConnectionsPerLayer: opts.M,
		maxConnectionsLayer0:   mmax0Multiplier * opts.M,
		layerMultiplier:        layerNormalizationBase / math.Log(float64(opts.M)),
		distanceFunc:           index.NewDistanceFunc(opts.DistanceType),
		opts:                   opts,
		segments:               make([]atomic.Pointer[[]*Node], 0, 16), // Start with capacity for 1M nodes
		freeList:               make([]uint32, 0),
		vectors:                opts.Vectors,
		arena:                  arena.New(arena.DefaultChunkSize),
		shardID:                0,
		numShards:              1,
		minQueuePool: &sync.Pool{
			New: func() any { return queue.NewMin(opts.EF) },
		},
		maxQueuePool: &sync.Pool{
			New: func() any { return queue.NewMax(opts.EF) },
		},
		bitsetPool: &sync.Pool{
			New: func() any { return &bitset.BitSet{} },
		},
	}

	h.dimensionAtomic.Store(int32(opts.Dimension))
	if h.vectors == nil {
		h.vectors = columnar.New(opts.Dimension)
	}

	// Initialize first segment
	h.growSegments(0)

	return h, nil
}

// getNode returns the node for the given ID, or nil if it doesn't exist/deleted.
// Lock-free read access.
func (h *HNSW) getNode(id uint32) *Node {
	segmentIdx := int(id >> nodeSegmentBits)

	// Fast path: check if segment exists without lock (atomic load implicit in slice access if pre-sized,
	// but here we use RLock on segments slice if we expect growth, or just atomic load of the pointer)
	// For simplicity and safety with growth, we use RLock on the segments slice wrapper.
	// Optimization: We can just read the slice if we assume it only grows.
	// But to be strictly safe in Go:
	h.segmentsMu.RLock()
	if segmentIdx >= len(h.segments) {
		h.segmentsMu.RUnlock()
		return nil
	}
	segmentPtr := h.segments[segmentIdx].Load()
	h.segmentsMu.RUnlock()

	if segmentPtr == nil {
		return nil
	}

	// No bounds check needed on segment due to bitmask, but good for safety
	idx := id & nodeSegmentMask
	return (*segmentPtr)[idx]
}

// setNode sets the node at the given ID. Thread-safe.
func (h *HNSW) setNode(id uint32, node *Node) {
	h.growSegments(id)

	segmentIdx := int(id >> nodeSegmentBits)
	h.segmentsMu.RLock()
	segmentPtr := h.segments[segmentIdx].Load()
	h.segmentsMu.RUnlock()

	(*segmentPtr)[id&nodeSegmentMask] = node
}

// growSegments ensures capacity for the given ID.
func (h *HNSW) growSegments(id uint32) {
	segmentIdx := int(id >> nodeSegmentBits)

	h.segmentsMu.RLock()
	if segmentIdx < len(h.segments) && h.segments[segmentIdx].Load() != nil {
		h.segmentsMu.RUnlock()
		return
	}
	h.segmentsMu.RUnlock()

	h.segmentsMu.Lock()
	defer h.segmentsMu.Unlock()

	// Double check
	for len(h.segments) <= segmentIdx {
		h.segments = append(h.segments, atomic.Pointer[[]*Node]{})
	}

	if h.segments[segmentIdx].Load() == nil {
		newSegment := make([]*Node, nodeSegmentSize)
		h.segments[segmentIdx].Store(&newSegment)
	}
}

// AllocateID returns a new ID.
func (h *HNSW) AllocateID() uint32 {
	h.freeListMu.Lock()
	if n := len(h.freeList); n > 0 {
		id := h.freeList[n-1]
		h.freeList = h.freeList[:n-1]
		h.freeListMu.Unlock()
		return id
	}
	h.freeListMu.Unlock()
	return h.nextIDAtomic.Add(1) - 1
}

// ReleaseID releases an ID.
func (h *HNSW) ReleaseID(id uint32) {
	if h.getNode(id) != nil {
		return // Don't release live IDs
	}
	h.freeListMu.Lock()
	h.freeList = append(h.freeList, id)
	h.freeListMu.Unlock()
}

// Insert inserts a vector.
func (h *HNSW) Insert(ctx context.Context, v []float32) (uint32, error) {
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	return h.insert(v, 0, -1, false)
}

// ApplyInsert inserts a vector with a specific ID (for WAL replay).
func (h *HNSW) ApplyInsert(ctx context.Context, id uint32, v []float32) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	_, err := h.insert(v, id, -1, true)
	return err
}

// ApplyDelete deletes a node with a specific ID (for WAL replay).
func (h *HNSW) ApplyDelete(ctx context.Context, id uint32) error {
	return h.Delete(ctx, id)
}

// ApplyUpdate updates a node with a specific ID (for WAL replay).
func (h *HNSW) ApplyUpdate(ctx context.Context, id uint32, v []float32) error {
	return h.Update(ctx, id, v)
}

// BatchInsert inserts multiple vectors.
func (h *HNSW) BatchInsert(ctx context.Context, vectors [][]float32) index.BatchInsertResult {
	result := index.BatchInsertResult{
		IDs:    make([]uint32, len(vectors)),
		Errors: make([]error, len(vectors)),
	}
	if err := ctx.Err(); err != nil {
		for i := range result.Errors {
			result.Errors[i] = err
		}
		return result
	}
	for i, v := range vectors {
		id, err := h.Insert(ctx, v)
		result.IDs[i] = id
		result.Errors[i] = err
	}
	return result
}

// insert is the unified insertion logic.
func (h *HNSW) insert(v []float32, id uint32, layer int, useProvidedID bool) (uint32, error) {
	if len(v) == 0 {
		return 0, index.ErrEmptyVector
	}
	dim := int(h.dimensionAtomic.Load())
	if len(v) != dim {
		return 0, &index.ErrDimensionMismatch{Expected: dim, Actual: len(v)}
	}

	// Normalize if needed
	var vec []float32
	if h.opts.NormalizeVectors {
		vec = make([]float32, len(v))
		copy(vec, v)
		if !distance.NormalizeL2InPlace(vec) {
			return 0, fmt.Errorf("hnsw: cannot normalize zero vector")
		}
	} else {
		vec = v // Zero-copy if not normalizing (vectors store will copy if needed)
	}

	// ID Allocation
	if useProvidedID {
		// Ensure ID is not in free list
		h.freeListMu.Lock()
		for i := len(h.freeList) - 1; i >= 0; i-- {
			if h.freeList[i] == id {
				h.freeList = append(h.freeList[:i], h.freeList[i+1:]...)
				break
			}
		}
		h.freeListMu.Unlock()

		// Ensure nextID > id
		for {
			cur := h.nextIDAtomic.Load()
			if cur > id {
				break
			}
			if h.nextIDAtomic.CompareAndSwap(cur, id+1) {
				break
			}
		}
	} else {
		id = h.AllocateID()
	}

	// Layer Selection
	if layer < 0 {
		if useProvidedID {
			layer = h.layerForApplyInsert(id)
		} else {
			layer = int(math.Floor(-math.Log(rand.Float64()) * h.layerMultiplier))
		}
	}

	// Create Node
	node := &Node{
		ID:          id,
		Layer:       layer,
		Connections: make([]atomic.Pointer[[]uint32], layer+1),
	}
	for l := 0; l <= layer; l++ {
		cap := h.maxConnectionsPerLayer
		if l == 0 {
			cap = h.maxConnectionsLayer0
		}
		empty := make([]uint32, 0, cap)
		node.Connections[l].Store(&empty)
	}

	// Store Vector
	if err := h.vectors.SetVector(id, vec); err != nil {
		if !useProvidedID {
			h.ReleaseID(id)
		}
		return 0, err
	}

	// Handle First Node
	if h.countAtomic.Load() == 0 {
		h.epMu.Lock()
		if h.countAtomic.Load() == 0 {
			h.setNode(id, node)
			h.entryPointAtomic.Store(id)
			h.maxLevelAtomic.Store(int32(layer))
			h.countAtomic.Add(1)
			h.epMu.Unlock()
			return id, nil
		}
		h.epMu.Unlock()
	}

	// Insert into Graph
	if err := h.insertNode(node, vec); err != nil {
		return 0, err
	}

	h.setNode(id, node)
	h.countAtomic.Add(1)

	// Update Entry Point
	maxLevel := int(h.maxLevelAtomic.Load())
	if layer > maxLevel {
		h.epMu.Lock()
		if layer > int(h.maxLevelAtomic.Load()) {
			h.maxLevelAtomic.Store(int32(layer))
			h.entryPointAtomic.Store(id)
		}
		h.epMu.Unlock()
	}

	return id, nil
}

// insertNode performs the graph traversal and linking.
func (h *HNSW) insertNode(node *Node, vec []float32) error {
	epID := h.entryPointAtomic.Load()
	epNode := h.getNode(epID)

	// Handle case where entry point was deleted concurrently
	if epNode == nil {
		// Fallback: try to find any live node as entry point
		// This is rare but possible during concurrent deletes
		// For now, just return error or handle gracefully.
		// Since we have count > 0 check before, this implies race.
		// We'll proceed with best effort or fail.
		return index.ErrEntryPointDeleted
	}

	currObj := epNode
	currDist := h.dist(vec, currObj.ID)

	// 1. Greedy search from top to node.Layer + 1
	maxLevel := int(h.maxLevelAtomic.Load())
	for level := maxLevel; level > node.Layer; level-- {
		changed := true
		for changed {
			changed = false
			if level >= len(currObj.Connections) {
				continue
			}
			conns := currObj.Connections[level].Load()
			for _, nextID := range *conns {
				nextDist := h.dist(vec, nextID)
				if nextDist < currDist {
					if nextNode := h.getNode(nextID); nextNode != nil {
						currObj = nextNode
						currDist = nextDist
						changed = true
					}
				}
			}
		}
	}

	// 2. Search and link from node.Layer down to 0
	for level := min(node.Layer, maxLevel); level >= 0; level-- {
		// Search layer (no filtering during insertion)
		candidates, err := h.searchLayer(vec, currObj, currDist, level, h.opts.EF, nil)
		if err != nil {
			return err
		}

		// Update entry point for next level
		if best, ok := candidates.MinItem(); ok {
			if bestNode := h.getNode(best.Node); bestNode != nil {
				currObj = bestNode
				currDist = best.Distance
			}
		}

		// Select neighbors
		maxConns := h.maxConnectionsPerLayer
		if level == 0 {
			maxConns = h.maxConnectionsLayer0
		}

		neighbors := h.selectNeighbors(candidates, maxConns)

		// Store connections
		node.Connections[level].Store(&neighbors)

		// Link back
		for _, neighborID := range neighbors {
			h.link(neighborID, node.ID, level, vec)
		}

		// Cleanup
		candidates.Reset()
		h.maxQueuePool.Put(candidates)
	}

	return nil
}

// link adds a bidirectional connection.
func (h *HNSW) link(srcID, dstID uint32, level int, dstVec []float32) {
	srcNode := h.getNode(srcID)
	if srcNode == nil {
		return
	}

	if level >= len(srcNode.Connections) {
		return
	}

	srcNode.mu.Lock()
	defer srcNode.mu.Unlock()

	oldConns := srcNode.Connections[level].Load()

	// Check if already connected (rare but possible)
	for _, id := range *oldConns {
		if id == dstID {
			return
		}
	}

	// Create new list
	newConns := make([]uint32, 0, len(*oldConns)+1)
	newConns = append(newConns, *oldConns...)
	newConns = append(newConns, dstID)

	maxConns := h.maxConnectionsPerLayer
	if level == 0 {
		maxConns = h.maxConnectionsLayer0
	}

	if len(newConns) > maxConns {
		// Prune
		candidates := h.maxQueuePool.Get().(*queue.PriorityQueue)
		candidates.Reset()

		srcVec, _ := h.vectors.GetVector(srcID) // Should exist

		for _, id := range newConns {
			var d float32
			if id == dstID {
				d = h.distanceFunc(srcVec, dstVec)
			} else {
				d = h.dist(srcVec, id)
			}
			candidates.PushItem(queue.PriorityQueueItem{Node: id, Distance: d})
		}

		pruned := h.selectNeighbors(candidates, maxConns)
		srcNode.Connections[level].Store(&pruned)

		candidates.Reset()
		h.maxQueuePool.Put(candidates)
	} else {
		srcNode.Connections[level].Store(&newConns)
	}
}

// selectNeighbors selects the best neighbors from candidates.
func (h *HNSW) selectNeighbors(candidates *queue.PriorityQueue, m int) []uint32 {
	if h.opts.Heuristic {
		return h.selectNeighborsHeuristic(candidates, m)
	}
	return h.selectNeighborsSimple(candidates, m)
}

func (h *HNSW) selectNeighborsSimple(candidates *queue.PriorityQueue, m int) []uint32 {
	// Simple selection (keep top M)
	for candidates.Len() > m {
		_, _ = candidates.PopItem()
	}

	res := make([]uint32, 0, candidates.Len())
	for candidates.Len() > 0 {
		item, _ := candidates.PopItem()
		res = append(res, item.Node)
	}
	// Reverse to have best first
	for i, j := 0, len(res)-1; i < j; i, j = i+1, j-1 {
		res[i], res[j] = res[j], res[i]
	}
	return res
}

func (h *HNSW) selectNeighborsHeuristic(candidates *queue.PriorityQueue, m int) []uint32 {
	if candidates.Len() <= m {
		return h.selectNeighborsSimple(candidates, m) // Fallback to simple if few candidates
	}

	// Extract all candidates to a slice sorted by distance (nearest first)
	// candidates is a MaxHeap (stores worst at top), so popping gives worst-to-best.
	// We want best-to-worst for the heuristic.

	temp := make([]queue.PriorityQueueItem, candidates.Len())
	for i := len(temp) - 1; i >= 0; i-- {
		temp[i], _ = candidates.PopItem()
	}

	result := make([]uint32, 0, m)
	for _, cand := range temp {
		if len(result) >= m {
			break
		}

		// Check if this candidate is closer to any already selected neighbor
		// than to the source node. (Relative Neighborhood Graph property)
		good := true
		candVec, _ := h.vectors.GetVector(cand.Node)

		for _, resID := range result {
			resVec, _ := h.vectors.GetVector(resID)
			dist := h.distanceFunc(candVec, resVec)
			if dist < cand.Distance {
				good = false
				break
			}
		}

		if good {
			result = append(result, cand.Node)
		}
	}

	// Fill up if needed
	if len(result) < m {
		for _, cand := range temp {
			if len(result) >= m {
				break
			}
			// Check if already added
			found := false
			for _, r := range result {
				if r == cand.Node {
					found = true
					break
				}
			}
			if !found {
				result = append(result, cand.Node)
			}
		}
	}

	return result
}

// searchLayer searches a specific layer with optional pre-filtering.
// filter: if not nil, nodes are filtered DURING traversal (not after).
// This ensures:
// 1. Correct recall (returns k results if available, not fewer)
// 2. Less wasted computation (skips filtered regions)
// 3. Matches exact search behavior
func (h *HNSW) searchLayer(query []float32, ep *Node, epDist float32, level int, ef int, filter func(uint32) bool) (*queue.PriorityQueue, error) {
	visited := h.bitsetPool.Get().(*bitset.BitSet)
	defer func() { visited.ClearAll(); h.bitsetPool.Put(visited) }()

	candidates := h.minQueuePool.Get().(*queue.PriorityQueue) // MinHeap: stores best candidates to explore
	candidates.Reset()

	results := h.maxQueuePool.Get().(*queue.PriorityQueue) // MaxHeap: stores current top EF results
	results.Reset()                                        // Caller must put back

	visited.Set(uint(ep.ID))

	// CRITICAL: Always add entry point to candidates for navigation (even if filtered)
	// This ensures we have a starting point for graph traversal
	candidates.PushItem(queue.PriorityQueueItem{Node: ep.ID, Distance: epDist})

	// Only add to results if it passes the filter
	if filter == nil || filter(ep.ID) {
		results.PushItem(queue.PriorityQueueItem{Node: ep.ID, Distance: epDist})
	}

	for candidates.Len() > 0 {
		curr, _ := candidates.PopItem()

		// Termination condition: only check if we have valid results
		if results.Len() > 0 {
			worst, _ := results.TopItem()
			if curr.Distance > worst.Distance && results.Len() >= ef {
				break
			}
		}

		node := h.getNode(curr.Node)
		if node == nil || level >= len(node.Connections) {
			continue
		}

		conns := node.Connections[level].Load()
		for _, nextID := range *conns {
			if !visited.Test(uint(nextID)) {
				visited.Set(uint(nextID))

				nextDist := h.dist(query, nextID)

				// Always add to candidates for navigation
				candidates.PushItem(queue.PriorityQueueItem{Node: nextID, Distance: nextDist})

				// PRE-FILTER: Only add to results if passes filter
				if filter == nil || filter(nextID) {
					// Check if we should add to results
					if results.Len() == 0 {
						// No results yet, add it
						results.PushItem(queue.PriorityQueueItem{Node: nextID, Distance: nextDist})
					} else {
						worst, _ := results.TopItem()
						if results.Len() < ef || nextDist < worst.Distance {
							results.PushItem(queue.PriorityQueueItem{Node: nextID, Distance: nextDist})

							if results.Len() > ef {
								_, _ = results.PopItem()
							}
						}
					}
				}
			}
		}
	}

	return results, nil
}

// dist computes distance between vector and node ID.
func (h *HNSW) dist(v []float32, id uint32) float32 {
	vec, ok := h.vectors.GetVector(id)
	if !ok {
		return math.MaxFloat32
	}
	return h.distanceFunc(v, vec)
}

// Delete deletes a node.
func (h *HNSW) Delete(ctx context.Context, id uint32) error {
	if err := ctx.Err(); err != nil {
		return err
	}

	node := h.getNode(id)
	if node == nil {
		return &index.ErrNodeDeleted{ID: id}
	}

	// Remove from graph (soft delete from array, hard delete from neighbors)
	h.setNode(id, nil)
	h.countAtomic.Add(-1)
	h.ReleaseID(id)

	// Remove connections to this node
	// Note: This is expensive (scan all nodes? No, scan neighbors).
	// But we don't know who points to us.
	// HNSW usually requires full scan or reverse index for clean delete.
	// Optimization: We only remove from neighbors we point to (undirected graph assumption).
	// Since links are bidirectional, our neighbors point to us.

	for level := 0; level < len(node.Connections); level++ {
		conns := node.Connections[level].Load()
		for _, neighborID := range *conns {
			h.removeLink(neighborID, id, level)
		}
	}

	// If entry point, find new one
	if h.entryPointAtomic.Load() == id {
		// Scan for new entry point (expensive but rare)
		// For production, we might just pick a random node or scan top layer
		// Simple scan:
		var newEP uint32
		var maxL int32 = -1

		// This is slow. In production, maybe maintain a set of top nodes?
		// Or just lazy update.
		// For now, linear scan of segments.
		h.segmentsMu.RLock()
		for i := range h.segments {
			seg := h.segments[i].Load()
			if seg == nil {
				continue
			}
			for _, n := range *seg {
				if n != nil && n.ID != id {
					if int32(n.Layer) > maxL {
						maxL = int32(n.Layer)
						newEP = n.ID
					}
				}
			}
		}
		h.segmentsMu.RUnlock()

		h.epMu.Lock()
		if h.entryPointAtomic.Load() == id {
			if maxL == -1 {
				h.entryPointAtomic.Store(0)
				h.maxLevelAtomic.Store(0)
			} else {
				h.entryPointAtomic.Store(newEP)
				h.maxLevelAtomic.Store(maxL)
			}
		}
		h.epMu.Unlock()
	}

	return h.vectors.DeleteVector(id)
}

func (h *HNSW) removeLink(srcID, targetID uint32, level int) {
	node := h.getNode(srcID)
	if node == nil {
		return
	}

	node.mu.Lock()
	defer node.mu.Unlock()

	oldConns := node.Connections[level].Load()
	newConns := make([]uint32, 0, len(*oldConns))
	for _, id := range *oldConns {
		if id != targetID {
			newConns = append(newConns, id)
		}
	}

	if len(newConns) != len(*oldConns) {
		node.Connections[level].Store(&newConns)
	}
}

// Update updates a vector.
func (h *HNSW) Update(ctx context.Context, id uint32, v []float32) error {
	node := h.getNode(id)
	if node == nil {
		return &index.ErrNodeNotFound{ID: id}
	}
	layer := node.Layer

	if err := h.Delete(ctx, id); err != nil {
		return err
	}

	_, err := h.insert(v, id, layer, true)
	return err
}

// KNNSearch performs search.
func (h *HNSW) KNNSearch(ctx context.Context, q []float32, k int, opts *index.SearchOptions) ([]index.SearchResult, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	// Normalize
	if h.opts.NormalizeVectors {
		// Copy to avoid modifying input
		qc := make([]float32, len(q))
		copy(qc, q)
		if !distance.NormalizeL2InPlace(qc) {
			return nil, fmt.Errorf("hnsw: zero query vector")
		}
		q = qc
	}

	epID := h.entryPointAtomic.Load()
	epNode := h.getNode(epID)
	if epNode == nil {
		return nil, nil
	}

	ef := h.opts.EF
	if opts != nil && opts.EFSearch > 0 {
		ef = opts.EFSearch
	}
	if ef < k {
		ef = k
	}

	// 1. Greedy to layer 0
	currObj := epNode
	currDist := h.dist(q, currObj.ID)
	maxLevel := int(h.maxLevelAtomic.Load())

	for level := maxLevel; level > 0; level-- {
		changed := true
		for changed {
			changed = false
			if currObj == nil {
				break
			}
			if level >= len(currObj.Connections) {
				continue
			}
			conns := currObj.Connections[level].Load()
			for _, nextID := range *conns {
				nextDist := h.dist(q, nextID)
				if nextDist < currDist {
					if nextNode := h.getNode(nextID); nextNode != nil {
						currObj = nextNode
						currDist = nextDist
						changed = true
					}
				}
			}
		}
	}

	// 2. Search layer 0 with pre-filtering
	var filter func(uint32) bool
	if opts != nil && opts.Filter != nil {
		filter = opts.Filter
	}

	results, err := h.searchLayer(q, currObj, currDist, 0, ef, filter)
	if err != nil {
		return nil, err
	}
	defer func() { results.Reset(); h.maxQueuePool.Put(results) }()

	// Extract K (filter is already applied during traversal)
	count := results.Len()
	if count > k {
		count = k
	}

	// MaxHeap pops worst first, so we need to pop (Len-k) items first
	for results.Len() > k {
		_, _ = results.PopItem()
	}

	res := make([]index.SearchResult, results.Len())
	for i := results.Len() - 1; i >= 0; i-- {
		item, _ := results.PopItem()
		res[i] = index.SearchResult{ID: item.Node, Distance: item.Distance}
	}

	// No post-filtering needed - filtering happened during traversal
	return res, nil
}

// KNNSearchStream implements streaming search.
func (h *HNSW) KNNSearchStream(ctx context.Context, q []float32, k int, opts *index.SearchOptions) iter.Seq2[index.SearchResult, error] {
	return func(yield func(index.SearchResult, error) bool) {
		res, err := h.KNNSearch(ctx, q, k, opts)
		if err != nil {
			yield(index.SearchResult{}, err)
			return
		}
		for _, r := range res {
			if !yield(r, nil) {
				return
			}
		}
	}
}

// BruteSearch implements brute force search.
func (h *HNSW) BruteSearch(ctx context.Context, query []float32, k int, filter func(id uint32) bool) ([]index.SearchResult, error) {
	// Simple scan
	pq := queue.NewMax(k)

	h.segmentsMu.RLock()
	defer h.segmentsMu.RUnlock()

	for i := range h.segments {
		seg := h.segments[i].Load()
		if seg == nil {
			continue
		}
		for _, node := range *seg {
			if node == nil {
				continue
			}
			if filter != nil && !filter(node.ID) {
				continue
			}

			d := h.dist(query, node.ID)
			if pq.Len() < k {
				pq.PushItem(queue.PriorityQueueItem{Node: node.ID, Distance: d})
			} else {
				top, _ := pq.TopItem()
				if d < top.Distance {
					_, _ = pq.PopItem()
					pq.PushItem(queue.PriorityQueueItem{Node: node.ID, Distance: d})
				}
			}
		}
	}

	res := make([]index.SearchResult, pq.Len())
	for i := pq.Len() - 1; i >= 0; i-- {
		item, _ := pq.PopItem()
		res[i] = index.SearchResult{ID: item.Node, Distance: item.Distance}
	}
	return res, nil
}

// Helper for layer generation
func (h *HNSW) layerForApplyInsert(id uint32) int {
	x := uint64(id) + 0x9e3779b97f4a7c15
	x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9
	x = (x ^ (x >> 27)) * 0x94d049bb133111eb
	x ^= x >> 31
	const inv = 1.0 / (1 << 53)
	u := x >> 11
	r := float64(u) * inv
	if r == 0 {
		r = inv
	}
	return int(math.Floor(-math.Log(r) * h.layerMultiplier))
}

// Sharding stubs
func (h *HNSW) VectorCount() int          { return int(h.countAtomic.Load()) }
func (h *HNSW) ContainsID(id uint32) bool { return true }
func (h *HNSW) ShardID() int              { return 0 }
func (h *HNSW) NumShards() int            { return 1 }
func NewSharded(shardID, numShards int, optFns ...func(o *Options)) (*HNSW, error) {
	return New(optFns...)
}

// VectorByID returns vector.
func (h *HNSW) VectorByID(ctx context.Context, id uint32) ([]float32, error) {
	vec, ok := h.vectors.GetVector(id)
	if !ok {
		return nil, &index.ErrNodeNotFound{ID: id}
	}
	return vec, nil
}

// Close closes the index.
func (h *HNSW) Close() error {
	if h.arena != nil {
		h.arena.Free()
	}
	return nil
}
