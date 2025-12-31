// Package hnsw implements the Hierarchical Navigable Small World (HNSW) graph for approximate nearest neighbor search.
package hnsw

import (
	"context"
	"fmt"
	"iter"
	"math"
	"math/rand"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"github.com/hupe1980/vecgo/distance"
	"github.com/hupe1980/vecgo/index"
	"github.com/hupe1980/vecgo/internal/bitset"
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

// OffsetSegment is a fixed-size array of node offsets.
type OffsetSegment [nodeSegmentSize]atomic.Uint64

// Options represents the options for configuring HNSW.
type Options struct {
	Dimension        int
	M                int
	EF               int
	Heuristic        bool
	DistanceType     index.DistanceType
	NormalizeVectors bool
	Vectors          vectorstore.Store
	InitialArenaSize int
	RandomSeed       *int64
}

var DefaultOptions = Options{
	Dimension:        0,
	M:                DefaultM,
	EF:               DefaultEF,
	Heuristic:        true,
	DistanceType:     index.DistanceTypeSquaredL2,
	InitialArenaSize: 1024 * 1024 * 1024, // 1GB
}

// graph holds the mutable state of the HNSW index.
type graph struct {
	// Hot path: Atomic fields
	entryPointAtomic atomic.Uint64
	maxLevelAtomic   atomic.Int32
	nextIDAtomic     atomic.Uint64
	countAtomic      atomic.Int64 // Track live count

	// Node storage: Segmented array of nodes
	nodes   atomic.Pointer[[]*NodeSegment]
	nodesMu sync.Mutex // Protects nodes growth

	// Sharded locks for node updates
	shardedLocks []sync.RWMutex

	// State
	tombstones *bitset.BitSet
}

// HNSW represents the Hierarchical Navigable Small World graph.
type HNSW struct {
	// Graph state (RCU-style)
	currentGraph atomic.Pointer[graph]

	// Components
	dimensionAtomic atomic.Int32
	distanceFunc    index.DistanceFunc
	vectors         vectorstore.Store
	rng             *rand.Rand
	rngMu           sync.Mutex

	// Configuration
	maxConnectionsPerLayer int
	maxConnectionsLayer0   int
	layerMultiplier        float64
	opts                   Options

	// Sharding
	shardID   int
	numShards int

	// Resources
	minQueuePool *sync.Pool
	maxQueuePool *sync.Pool
	visitedPool  *sync.Pool
	scratchPool  *sync.Pool
}

type scratch struct {
	floats  []float32
	results []index.SearchResult

	// Heuristic scratch buffers
	heuristicCandidates []queue.PriorityQueueItem
	heuristicResult     []queue.PriorityQueueItem
	heuristicResultVecs [][]float32
}

func (*HNSW) Name() string { return "HNSW" }

// Reset clears the HNSW index for reuse.
func (h *HNSW) Reset() {
	h.currentGraph.Store(newGraph())

	// Reset vectors if supported
	if reseter, ok := h.vectors.(interface{ Reset() }); ok {
		reseter.Reset()
	}
}

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

	if opts.InitialArenaSize == 0 {
		opts.InitialArenaSize = 1024 * 1024 * 1024 // 1GB
	}

	var rng *rand.Rand
	if opts.RandomSeed != nil {
		rng = rand.New(rand.NewSource(*opts.RandomSeed))
	} else {
		rng = rand.New(rand.NewSource(time.Now().UnixNano()))
	}

	h := &HNSW{
		maxConnectionsPerLayer: opts.M,
		maxConnectionsLayer0:   mmax0Multiplier * opts.M,
		layerMultiplier:        layerNormalizationBase / math.Log(float64(opts.M)),
		distanceFunc:           index.NewDistanceFunc(opts.DistanceType),
		opts:                   opts,
		rng:                    rng,
		vectors:                opts.Vectors,
		shardID:                0,
		numShards:              1,
	}
	h.currentGraph.Store(newGraph())

	h.initPools()

	h.dimensionAtomic.Store(int32(opts.Dimension))
	if h.vectors == nil {
		h.vectors = columnar.New(opts.Dimension)
	}

	return h, nil
}

func newGraph() *graph {
	g := &graph{
		shardedLocks: make([]sync.RWMutex, 1024),
		tombstones:   bitset.New(1024),
	}
	g.maxLevelAtomic.Store(-1)
	// Initialize first segment
	nodes := make([]*NodeSegment, 1)
	nodes[0] = new(NodeSegment)
	g.nodes.Store(&nodes)
	return g
}

func (h *HNSW) initPools() {
	h.minQueuePool = &sync.Pool{
		New: func() any { return queue.NewMin(h.opts.EF) },
	}
	h.maxQueuePool = &sync.Pool{
		New: func() any { return queue.NewMax(h.opts.EF) },
	}
	h.visitedPool = &sync.Pool{
		New: func() any { return bitset.NewFast(1024) },
	}
	h.scratchPool = &sync.Pool{
		New: func() any {
			return &scratch{
				floats:              make([]float32, h.opts.Dimension),
				results:             make([]index.SearchResult, 0, h.opts.EF),
				heuristicCandidates: make([]queue.PriorityQueueItem, 0, h.opts.EF),
				heuristicResult:     make([]queue.PriorityQueueItem, 0, h.opts.M),
				heuristicResultVecs: make([][]float32, 0, h.opts.M),
			}
		},
	}
}

// getNodeOffset returns the offset for the given ID, or 0 if not found.
// getNode returns the node for the given ID, or nil if not found.
func (h *HNSW) getNode(g *graph, id uint64) *Node {
	nodes := g.nodes.Load()
	if nodes == nil {
		return nil
	}

	segmentIdx := int(id >> nodeSegmentBits)
	if segmentIdx >= len(*nodes) {
		return nil
	}

	segment := (*nodes)[segmentIdx]
	if segment == nil {
		return nil
	}

	return segment[id&nodeSegmentMask].Load()
}

// setNode sets the node for the given ID.
func (h *HNSW) setNode(g *graph, id uint64, node *Node) {
	h.growNodes(g, id)

	nodes := g.nodes.Load()
	segmentIdx := int(id >> nodeSegmentBits)
	segment := (*nodes)[segmentIdx]

	segment[id&nodeSegmentMask].Store(node)
}

// growNodes ensures capacity for the given ID.
func (h *HNSW) growNodes(g *graph, id uint64) {
	segmentIdx := int(id >> nodeSegmentBits)

	// Fast path: check if segment exists
	nodes := g.nodes.Load()
	if nodes != nil && segmentIdx < len(*nodes) && (*nodes)[segmentIdx] != nil {
		return
	}

	// Slow path: grow using lock
	g.nodesMu.Lock()
	defer g.nodesMu.Unlock()

	// Reload under lock
	nodes = g.nodes.Load()
	var currentNodes []*NodeSegment
	if nodes != nil {
		currentNodes = *nodes
	}

	// Check again
	if segmentIdx < len(currentNodes) && currentNodes[segmentIdx] != nil {
		return
	}

	// Grow slice if needed
	newNodes := currentNodes
	for len(newNodes) <= segmentIdx {
		newNodes = append(newNodes, new(NodeSegment))
	}

	// Store updated slice
	// Note: We are replacing the slice pointer, but the underlying array might be reallocated by append.
	// Readers loading the old pointer are safe because we only append.
	// However, to be fully safe with concurrent readers who might be iterating,
	// we should ensure we don't invalidate the old backing array if they are using it.
	// But here we are just storing a slice of pointers to segments.
	// The segments themselves are stable.
	g.nodes.Store(&newNodes)
}

// Helper methods for node access

func (h *HNSW) getConnections(g *graph, id uint64, layer int) []Neighbor {
	node := h.getNode(g, id)
	if node == nil {
		return nil
	}
	return node.getConnections(layer)
}

func (h *HNSW) setConnections(g *graph, id uint64, layer int, conns []Neighbor) {
	node := h.getNode(g, id)
	if node == nil {
		return
	}

	// Copy to new slice to ensure safety (COW)
	newConns := make([]Neighbor, len(conns))
	copy(newConns, conns)

	node.setConnections(layer, newConns)
}

func (h *HNSW) addConnection(g *graph, sourceID, targetID uint64, level int, dist float32) {
	g.shardedLocks[sourceID%uint64(len(g.shardedLocks))].Lock()
	defer g.shardedLocks[sourceID%uint64(len(g.shardedLocks))].Unlock()

	node := h.getNode(g, sourceID)
	if node == nil {
		return
	}

	if level > node.Level {
		return // Should not happen
	}

	conns := node.getConnections(level)

	// Check if already connected using binary search (conns is sorted)
	// We assume conns is sorted. If not, we should sort it or use linear scan.
	// For now, let's use linear scan but optimized.
	// Actually, let's enforce sorting on insert.

	// Linear scan is fine for small M (8-32).
	// slices.Contains is optimized in Go 1.21+.
	// Optimization: Use manual loop to avoid overhead of generic slices.Contains
	found := false
	for _, c := range conns {
		if c.ID == targetID {
			found = true
			break
		}
	}
	if found {
		return
	}

	maxM := h.maxConnectionsPerLayer
	if level == 0 {
		maxM = h.maxConnectionsLayer0
	}

	if len(conns) < maxM {
		// Just append
		var newConns []Neighbor
		if cap(conns) >= len(conns)+1 {
			// We have capacity, append in place (safe because we hold lock and readers only read up to len)
			newConns = conns[:len(conns)+1]
			newConns[len(conns)] = Neighbor{ID: targetID, Dist: dist}
		} else {
			// Allocate with capacity
			newConns = make([]Neighbor, len(conns)+1, maxM)
			copy(newConns, conns)
			newConns[len(conns)] = Neighbor{ID: targetID, Dist: dist}
		}
		node.setConnections(level, newConns)
	} else {
		// Prune
		candidates := h.maxQueuePool.Get().(*queue.PriorityQueue)
		candidates.Reset()
		defer h.maxQueuePool.Put(candidates)

		// Add existing - use cached distances!
		for _, c := range conns {
			candidates.PushItem(queue.PriorityQueueItem{Node: c.ID, Distance: c.Dist})
		}
		// Add new
		candidates.PushItem(queue.PriorityQueueItem{Node: targetID, Distance: dist})

		// Use scratch for selectNeighbors
		scratch := h.scratchPool.Get().(*scratch)
		neighbors := h.selectNeighbors(candidates, maxM, scratch)

		// Allocate final slice with capacity maxM
		finalConns := make([]Neighbor, len(neighbors), maxM)
		for i, n := range neighbors {
			finalConns[i] = Neighbor{ID: n.Node, Dist: n.Distance}
		}

		h.scratchPool.Put(scratch)

		node.setConnections(level, finalConns)
	}
}

// AllocateID returns a new ID.
func (h *HNSW) AllocateID() uint64 {
	return h.allocateID(h.currentGraph.Load())
}

func (h *HNSW) allocateID(g *graph) uint64 {
	return g.nextIDAtomic.Add(1) - 1
}

// ReleaseID releases an ID.
func (h *HNSW) ReleaseID(id uint64) {
	// No-op: IDs are never reused to ensure stability.
}

func (h *HNSW) releaseID(g *graph, id uint64) {
	// No-op: IDs are never reused to ensure stability.
}

// Insert inserts a vector.
func (h *HNSW) Insert(ctx context.Context, v []float32) (uint64, error) {
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	return h.insert(h.currentGraph.Load(), v, 0, -1, false)
}

// ApplyInsert inserts a vector with a specific ID (for WAL replay).
func (h *HNSW) ApplyInsert(ctx context.Context, id uint64, v []float32) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	_, err := h.insert(h.currentGraph.Load(), v, id, -1, true)
	return err
}

// ApplyBatchInsert inserts multiple vectors with specific IDs concurrently.
func (h *HNSW) ApplyBatchInsert(ctx context.Context, ids []uint64, vectors [][]float32) error {
	if len(ids) != len(vectors) {
		return fmt.Errorf("ids and vectors length mismatch")
	}
	if err := ctx.Err(); err != nil {
		return err
	}

	g := h.currentGraph.Load()
	var wg sync.WaitGroup

	// Limit concurrency to avoid overwhelming the system
	concurrency := runtime.GOMAXPROCS(0)
	sem := make(chan struct{}, concurrency)

	var firstErr error
	var errMu sync.Mutex

	for i := range vectors {
		wg.Add(1)
		sem <- struct{}{}
		go func(idx int) {
			defer wg.Done()
			defer func() { <-sem }()

			if _, err := h.insert(g, vectors[idx], ids[idx], -1, true); err != nil {
				errMu.Lock()
				if firstErr == nil {
					firstErr = err
				}
				errMu.Unlock()
			}
		}(i)
	}
	wg.Wait()
	return firstErr
}

// ApplyDelete deletes a node with a specific ID (for WAL replay).
func (h *HNSW) ApplyDelete(ctx context.Context, id uint64) error {
	return h.Delete(ctx, id)
}

// ApplyUpdate updates a node with a specific ID (for WAL replay).
func (h *HNSW) ApplyUpdate(ctx context.Context, id uint64, v []float32) error {
	return h.Update(ctx, id, v)
}

// BatchInsert inserts multiple vectors.
func (h *HNSW) BatchInsert(ctx context.Context, vectors [][]float32) index.BatchInsertResult {
	result := index.BatchInsertResult{
		IDs:    make([]uint64, len(vectors)),
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
func (h *HNSW) insert(g *graph, v []float32, id uint64, layer int, useProvidedID bool) (uint64, error) {
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
		// Ensure nextID > id
		for {
			cur := g.nextIDAtomic.Load()
			if cur > id {
				break
			}
			if g.nextIDAtomic.CompareAndSwap(cur, id+1) {
				break
			}
		}
	} else {
		id = h.allocateID(g)
	}

	// Layer Selection
	if layer < 0 {
		if useProvidedID {
			layer = h.layerForApplyInsert(id)
		} else {
			h.rngMu.Lock()
			r := h.rng.Float64()
			h.rngMu.Unlock()
			layer = int(math.Floor(-math.Log(r) * h.layerMultiplier))
		}
	}

	// Create Node (Heap)
	node := newNode(layer)

	// Store Vector
	if err := h.vectors.SetVector(id, vec); err != nil {
		if !useProvidedID {
			h.releaseID(g, id)
		}
		return 0, err
	}

	// Publish node so it can be found
	h.setNode(g, id, node)

	retries := 0
	for {
		// Handle First Node
		if g.countAtomic.Load() == 0 {
			if g.countAtomic.CompareAndSwap(0, 1) {
				h.setNode(g, id, node)
				g.entryPointAtomic.Store(id)
				g.maxLevelAtomic.Store(int32(layer))
				return id, nil
			}
			// Lost race, continue
		}

		// Insert into Graph
		err := h.insertNode(g, id, vec, layer)
		if err == index.ErrEntryPointDeleted {
			retries++
			if retries > 10 {
				h.recoverEntryPoint(g)
				retries = 0
			}
			runtime.Gosched()
			continue
		}
		if err != nil {
			return 0, err
		}
		break
	}

	g.countAtomic.Add(1)

	// Update Entry Point
	maxLevel := int(g.maxLevelAtomic.Load())
	if layer > maxLevel {
		for {
			oldMax := g.maxLevelAtomic.Load()
			if layer <= int(oldMax) {
				break
			}
			if g.maxLevelAtomic.CompareAndSwap(oldMax, int32(layer)) {
				g.entryPointAtomic.Store(id)
				break
			}
		}
	}

	return id, nil
}

// insertNode performs the graph traversal and linking.
func (h *HNSW) insertNode(g *graph, id uint64, vec []float32, layer int) error {
	epID := g.entryPointAtomic.Load()

	// Handle case where entry point was deleted concurrently
	if h.getNode(g, epID) == nil {
		return index.ErrEntryPointDeleted
	}

	currID := epID
	currDist := h.dist(vec, currID)

	// 1. Greedy search from top to node.Layer + 1
	maxLevel := int(g.maxLevelAtomic.Load())
	for level := maxLevel; level > layer; level-- {
		changed := true
		for changed {
			changed = false
			conns := h.getConnections(g, currID, level)
			for _, next := range conns {
				nextDist := h.dist(vec, next.ID)
				if nextDist < currDist {
					currID = next.ID
					currDist = nextDist
					changed = true
				}
			}
		}
	}

	// 2. Search and link from node.Layer down to 0
	scratch := h.scratchPool.Get().(*scratch)
	defer h.scratchPool.Put(scratch)

	for level := min(layer, maxLevel); level >= 0; level-- {
		// Search layer (no filtering during insertion)
		candidates, err := h.searchLayer(g, vec, currID, currDist, level, h.opts.EF, nil)
		if err != nil {
			return err
		}

		// Update entry point for next level
		if best, ok := candidates.MinItem(); ok {
			currID = best.Node
			currDist = best.Distance
		}

		// Select neighbors
		maxConns := h.maxConnectionsPerLayer
		if level == 0 {
			maxConns = h.maxConnectionsLayer0
		}

		neighbors := h.selectNeighbors(candidates, maxConns, scratch)

		// Put back candidates (results from searchLayer)
		candidates.Reset()
		h.maxQueuePool.Put(candidates)

		// Extract IDs for setConnections
		neighborConns := make([]Neighbor, len(neighbors))
		for i, n := range neighbors {
			neighborConns[i] = Neighbor{ID: n.Node, Dist: n.Distance}
		}

		// Add bidirectional connections
		g.shardedLocks[id%uint64(len(g.shardedLocks))].Lock()
		h.setConnections(g, id, level, neighborConns)
		g.shardedLocks[id%uint64(len(g.shardedLocks))].Unlock()

		for _, neighbor := range neighbors {
			h.addConnection(g, neighbor.Node, id, level, neighbor.Distance)
		}
	}

	return nil
}

// selectNeighbors selects the best neighbors from candidates.
func (h *HNSW) selectNeighbors(candidates *queue.PriorityQueue, m int, scratch *scratch) []queue.PriorityQueueItem {
	if h.opts.Heuristic {
		return h.selectNeighborsHeuristic(candidates, m, scratch)
	}
	return h.selectNeighborsSimple(candidates, m, scratch)
}

func (h *HNSW) selectNeighborsSimple(candidates *queue.PriorityQueue, m int, scratch *scratch) []queue.PriorityQueueItem {
	// Simple selection (keep top M)
	for candidates.Len() > m {
		_, _ = candidates.PopItem()
	}

	res := scratch.heuristicResult[:0]
	for candidates.Len() > 0 {
		item, _ := candidates.PopItem()
		res = append(res, item)
	}
	// Reverse to have best first
	for i, j := 0, len(res)-1; i < j; i, j = i+1, j-1 {
		res[i], res[j] = res[j], res[i]
	}
	return res
}

func (h *HNSW) selectNeighborsHeuristic(candidates *queue.PriorityQueue, m int, scratch *scratch) []queue.PriorityQueueItem {
	if candidates.Len() <= m {
		return h.selectNeighborsSimple(candidates, m, scratch) // Fallback to simple if few candidates
	}

	// Extract all candidates to a slice sorted by distance (nearest first)
	// candidates is a MaxHeap (stores worst at top), so popping gives worst-to-best.
	// We want best-to-worst for the heuristic.

	// Use scratch buffer for candidates
	temp := scratch.heuristicCandidates[:0]
	for candidates.Len() > 0 {
		item, _ := candidates.PopItem()
		temp = append(temp, item)
	}
	// Reverse to get nearest first
	for i, j := 0, len(temp)-1; i < j; i, j = i+1, j-1 {
		temp[i], temp[j] = temp[j], temp[i]
	}

	// Use scratch buffers for results
	result := scratch.heuristicResult[:0]
	resultVecs := scratch.heuristicResultVecs[:0]

	for _, cand := range temp {
		if len(result) >= m {
			break
		}

		// Check if this candidate is closer to any already selected neighbor
		// than to the source node. (Relative Neighborhood Graph property)
		good := true
		candVec, ok := h.vectors.GetVector(cand.Node)
		if !ok {
			continue
		}

		for _, resVec := range resultVecs {
			dist := h.distanceFunc(candVec, resVec)
			if dist < cand.Distance {
				good = false
				break
			}
		}

		if good {
			result = append(result, cand)
			resultVecs = append(resultVecs, candVec)
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
				if r.Node == cand.Node {
					found = true
					break
				}
			}
			if !found {
				result = append(result, cand)
				// We don't need to update resultVecs here as we don't use it anymore
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
func (h *HNSW) searchLayer(g *graph, query []float32, epID uint64, epDist float32, level int, ef int, filter func(uint64) bool) (*queue.PriorityQueue, error) {
	visited := h.visitedPool.Get().(*bitset.FastBitSet)
	visited.Reset()
	defer h.visitedPool.Put(visited)

	candidates := h.minQueuePool.Get().(*queue.PriorityQueue) // MinHeap: stores best candidates to explore
	candidates.Reset()
	defer func() {
		candidates.Reset()
		h.minQueuePool.Put(candidates)
	}()

	results := h.maxQueuePool.Get().(*queue.PriorityQueue) // MaxHeap: stores current top EF results
	results.Reset()                                        // Caller must put back

	visited.Set(epID)

	// CRITICAL: Always add entry point to candidates for navigation (even if filtered)
	// This ensures we have a starting point for graph traversal
	candidates.PushItem(queue.PriorityQueueItem{Node: epID, Distance: epDist})

	// Only add to results if it passes the filter AND is not deleted
	if (filter == nil || filter(epID)) && !g.tombstones.Test(epID) {
		results.PushItem(queue.PriorityQueueItem{Node: epID, Distance: epDist})
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

		conns := h.getConnections(g, curr.Node, level)
		for _, next := range conns {
			if !visited.Test(next.ID) {
				visited.Set(next.ID)

				nextDist := h.dist(query, next.ID)

				// Classic HNSW pruning: avoid pushing obviously-bad candidates once we already
				// have ef results. This substantially reduces heap churn.
				//
				// IMPORTANT: we only apply this optimization when there is no filter.
				// With filtering enabled, we intentionally keep traversal more permissive to
				// avoid getting trapped in filtered-out regions.
				shouldExplore := true
				if filter == nil && results.Len() > 0 && results.Len() >= ef {
					worst, _ := results.TopItem()
					if nextDist > worst.Distance {
						shouldExplore = false
					}
				}

				if shouldExplore {
					candidates.PushItem(queue.PriorityQueueItem{Node: next.ID, Distance: nextDist})

					// Only add to results if it passes the filter AND is not deleted
					if (filter == nil || filter(next.ID)) && !g.tombstones.Test(next.ID) {
						// Use bounded push for results to avoid heap churn
						results.PushItemBounded(queue.PriorityQueueItem{Node: next.ID, Distance: nextDist}, ef)
					}
				}
			}
		}
	}

	return results, nil
}

// dist computes distance between vector and node ID.
func (h *HNSW) dist(v []float32, id uint64) float32 {
	vec, ok := h.vectors.GetVector(id)
	if !ok {
		return math.MaxFloat32
	}
	return h.distanceFunc(v, vec)
}

// Delete marks a node as deleted (logical delete).
// This is O(1) and avoids graph instability.
func (h *HNSW) Delete(ctx context.Context, id uint64) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	g := h.currentGraph.Load()

	// Use lock to protect tombstones
	g.shardedLocks[id%uint64(len(g.shardedLocks))].Lock()
	defer g.shardedLocks[id%uint64(len(g.shardedLocks))].Unlock()

	// Ensure bitset capacity
	g.tombstones.Grow(id + 1)
	if g.tombstones.Test(id) {
		return nil
	}
	g.tombstones.Set(id)

	// We do NOT remove from graph or release ID.
	// This preserves graph connectivity and avoids O(N) entry point scans.
	// The node remains in the graph but is ignored during search.

	// We can optionally delete the vector data to save memory,
	// but keeping it might be safer for concurrent readers.
	// For now, we keep it.

	g.countAtomic.Add(-1)
	return nil
}

// Update updates a vector.
func (h *HNSW) Update(ctx context.Context, id uint64, v []float32) error {
	g := h.currentGraph.Load()
	node := h.getNode(g, id)
	if node == nil {
		return &index.ErrNodeNotFound{ID: id}
	}
	layer := node.Level

	if err := h.Delete(ctx, id); err != nil {
		return err
	}

	_, err := h.insert(g, v, id, layer, true)
	return err
}

// KNNSearch performs search.
func (h *HNSW) KNNSearch(ctx context.Context, q []float32, k int, opts *index.SearchOptions) ([]index.SearchResult, error) {
	res := make([]index.SearchResult, 0, k)
	if err := h.KNNSearchWithBuffer(ctx, q, k, opts, &res); err != nil {
		return nil, err
	}
	return res, nil
}

// KNNSearchWithBuffer performs a K-nearest neighbor search and appends results to the provided buffer.
// This avoids allocating a new slice for results, which is critical for high-throughput scenarios.
func (h *HNSW) KNNSearchWithBuffer(ctx context.Context, q []float32, k int, opts *index.SearchOptions, buf *[]index.SearchResult) error {
	if err := ctx.Err(); err != nil {
		return err
	}

	// Safety check: Dimension mismatch
	// This is critical because internal distance functions (SIMD) do not check bounds.
	dim := int(h.dimensionAtomic.Load())
	if dim > 0 && len(q) != dim {
		return &index.ErrDimensionMismatch{Expected: dim, Actual: len(q)}
	}

	g := h.currentGraph.Load()

	// Get scratch buffer
	scratch := h.scratchPool.Get().(*scratch)
	defer h.scratchPool.Put(scratch)

	// Normalize
	if h.opts.NormalizeVectors {
		if len(scratch.floats) < len(q) {
			scratch.floats = make([]float32, len(q))
		}
		copy(scratch.floats, q)
		if !distance.NormalizeL2InPlace(scratch.floats) {
			return fmt.Errorf("hnsw: zero query vector")
		}
		q = scratch.floats
	}

	epID := g.entryPointAtomic.Load()
	if h.getNode(g, epID) == nil {
		return nil
	}

	ef := h.opts.EF
	if opts != nil && opts.EFSearch > 0 {
		ef = opts.EFSearch
	}
	if ef < k {
		ef = k
	}

	// 1. Greedy to layer 0
	currID := epID
	currDist := h.dist(q, currID)
	maxLevel := int(g.maxLevelAtomic.Load())

	for level := maxLevel; level > 0; level-- {
		changed := true
		for changed {
			changed = false
			conns := h.getConnections(g, currID, level)
			for _, next := range conns {
				nextDist := h.dist(q, next.ID)
				if nextDist < currDist {
					currID = next.ID
					currDist = nextDist
					changed = true
				}
			}
		}
	}

	// 2. Search layer 0 with pre-filtering
	var filter func(uint64) bool
	if opts != nil && opts.Filter != nil {
		filter = opts.Filter
	}

	results, err := h.searchLayer(g, q, currID, currDist, 0, ef, filter)
	if err != nil {
		return err
	}
	defer func() { results.Reset(); h.maxQueuePool.Put(results) }()

	// Extract K (filter is already applied during traversal)
	// MaxHeap pops worst first, so we need to pop (Len-k) items first
	for results.Len() > k {
		_, _ = results.PopItem()
	}

	// Collect results in reverse order (nearest first)
	// We use scratch.results as a temporary stack to reverse
	scratch.results = scratch.results[:0]
	for results.Len() > 0 {
		item, _ := results.PopItem()
		scratch.results = append(scratch.results, index.SearchResult{ID: item.Node, Distance: item.Distance})
	}

	// Append to output buffer in correct order
	for i := len(scratch.results) - 1; i >= 0; i-- {
		*buf = append(*buf, scratch.results[i])
	}

	return nil
}

// KNNSearchStream implements streaming search.
func (h *HNSW) KNNSearchStream(ctx context.Context, q []float32, k int, opts *index.SearchOptions) iter.Seq2[index.SearchResult, error] {
	return func(yield func(index.SearchResult, error) bool) {
		if err := ctx.Err(); err != nil {
			yield(index.SearchResult{}, err)
			return
		}
		g := h.currentGraph.Load()

		// Get scratch buffer
		scratch := h.scratchPool.Get().(*scratch)
		defer h.scratchPool.Put(scratch)

		// Normalize
		if h.opts.NormalizeVectors {
			if len(scratch.floats) < len(q) {
				scratch.floats = make([]float32, len(q))
			}
			copy(scratch.floats, q)
			if !distance.NormalizeL2InPlace(scratch.floats) {
				yield(index.SearchResult{}, fmt.Errorf("hnsw: zero query vector"))
				return
			}
			q = scratch.floats
		}

		epID := g.entryPointAtomic.Load()
		if h.getNode(g, epID) == nil {
			return
		}

		ef := h.opts.EF
		if opts != nil && opts.EFSearch > 0 {
			ef = opts.EFSearch
		}
		if ef < k {
			ef = k
		}

		// 1. Greedy to layer 0
		currID := epID
		currDist := h.dist(q, currID)
		maxLevel := int(g.maxLevelAtomic.Load())

		for level := maxLevel; level > 0; level-- {
			changed := true
			for changed {
				changed = false
				conns := h.getConnections(g, currID, level)
				for _, next := range conns {
					nextDist := h.dist(q, next.ID)
					if nextDist < currDist {
						currID = next.ID
						currDist = nextDist
						changed = true
					}
				}
			}
		}

		// 2. Search layer 0 with pre-filtering
		var filter func(uint64) bool
		if opts != nil && opts.Filter != nil {
			filter = opts.Filter
		}

		results, err := h.searchLayer(g, q, currID, currDist, 0, ef, filter)
		if err != nil {
			yield(index.SearchResult{}, err)
			return
		}
		defer func() { results.Reset(); h.maxQueuePool.Put(results) }()

		// Extract K
		count := results.Len()
		if count > k {
			count = k
		}

		// MaxHeap pops worst first, so we need to pop (Len-k) items first
		for results.Len() > k {
			_, _ = results.PopItem()
		}

		// Collect results in reverse order (nearest first)
		scratch.results = scratch.results[:0]
		for results.Len() > 0 {
			item, _ := results.PopItem()
			scratch.results = append(scratch.results, index.SearchResult{ID: item.Node, Distance: item.Distance})
		}

		// Yield in reverse (since we popped worst first, the last item in scratch.results is the best)
		for i := len(scratch.results) - 1; i >= 0; i-- {
			if !yield(scratch.results[i], nil) {
				return
			}
		}
	}
}

// BruteSearch implements brute force search.
func (h *HNSW) BruteSearch(ctx context.Context, query []float32, k int, filter func(id uint64) bool) ([]index.SearchResult, error) {
	g := h.currentGraph.Load()
	// Simple scan
	pq := queue.NewMax(k)

	nodes := g.nodes.Load()
	if nodes != nil {
		for i, seg := range *nodes {
			if seg == nil {
				continue
			}
			for j := range seg {
				node := seg[j].Load()
				if node == nil {
					continue
				}
				nodeID := uint64(i)*nodeSegmentSize + uint64(j)
				if filter != nil && !filter(nodeID) {
					continue
				}

				d := h.dist(query, nodeID)
				if pq.Len() < k {
					pq.PushItem(queue.PriorityQueueItem{Node: nodeID, Distance: d})
				} else {
					top, _ := pq.TopItem()
					if d < top.Distance {
						_, _ = pq.PopItem()
						pq.PushItem(queue.PriorityQueueItem{Node: nodeID, Distance: d})
					}
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
func (h *HNSW) layerForApplyInsert(id uint64) int {
	x := id + 0x9e3779b97f4a7c15
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
func (h *HNSW) VectorCount() int { return int(h.currentGraph.Load().countAtomic.Load()) }
func (h *HNSW) ContainsID(id uint64) bool {
	g := h.currentGraph.Load()
	if g.tombstones.Test(id) {
		return false
	}
	return h.getNode(g, id) != nil
}
func (h *HNSW) ShardID() int   { return 0 }
func (h *HNSW) NumShards() int { return 1 }
func NewSharded(shardID, numShards int, optFns ...func(o *Options)) (*HNSW, error) {
	return New(optFns...)
}

// VectorByID returns vector.
func (h *HNSW) VectorByID(ctx context.Context, id uint64) ([]float32, error) {
	vec, ok := h.vectors.GetVector(id)
	if !ok {
		return nil, &index.ErrNodeNotFound{ID: id}
	}
	return vec, nil
}

// Close closes the index.
func (h *HNSW) Close() error {
	return nil
}

func (h *HNSW) recoverEntryPoint(g *graph) {
	// Double check
	epID := g.entryPointAtomic.Load()
	if h.getNode(g, epID) != nil {
		return
	}

	// Scan for valid node
	nodes := g.nodes.Load()
	if nodes == nil {
		g.countAtomic.Store(0)
		return
	}

	for i, segment := range *nodes {
		if segment == nil {
			continue
		}
		for j := 0; j < nodeSegmentSize; j++ {
			node := segment[j].Load()
			if node != nil {
				// Found one!
				id := uint64(i)*nodeSegmentSize + uint64(j)
				if !g.tombstones.Test(id) {
					g.entryPointAtomic.Store(id)
					g.maxLevelAtomic.Store(int32(node.Level))
					return
				}
			}
		}
	}

	// If no nodes found, reset count
	g.countAtomic.Store(0)
}

// Dimension returns the dimensionality of the vectors in the index.
func (h *HNSW) Dimension() int {
	return int(h.dimensionAtomic.Load())
}
