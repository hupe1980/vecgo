// Package hnsw implements the Hierarchical Navigable Small World (HNSW) graph for approximate nearest neighbor search.
package hnsw

import (
	"context"
	"encoding/binary"
	"fmt"
	"iter"
	"math"
	"math/rand"
	"sync"
	"sync/atomic"
	"time"

	"github.com/hupe1980/vecgo/distance"
	"github.com/hupe1980/vecgo/index"
	"github.com/hupe1980/vecgo/internal/arena"
	"github.com/hupe1980/vecgo/internal/bitset"
	"github.com/hupe1980/vecgo/internal/queue"
	"github.com/hupe1980/vecgo/internal/visited"
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
	InitialArenaSize: 64 * 1024 * 1024, // 64MB
}

// HNSW represents the Hierarchical Navigable Small World graph.
type HNSW struct {
	// Hot path: Atomic fields
	entryPointAtomic atomic.Uint64
	maxLevelAtomic   atomic.Int32
	dimensionAtomic  atomic.Int32
	nextIDAtomic     atomic.Uint64
	countAtomic      atomic.Int64 // Track live count

	// Node storage: Segmented array of offsets
	segments atomic.Pointer[[]*OffsetSegment]
	// mmapOffsets is used for read-only mmap'd indexes to avoid indirection overhead.
	// If non-nil, it takes precedence over segments.
	mmapOffsets []uint64

	// Arena for node data
	arena  *arena.FlatArena
	layout *nodeLayout

	// Sharded locks for node updates
	shardedLocks []sync.RWMutex

	// Components
	distanceFunc index.DistanceFunc
	vectors      vectorstore.Store
	rng          *rand.Rand
	rngMu        sync.Mutex

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
	freeList   []uint64
	tombstones *bitset.BitSet

	// Resources
	minQueuePool *sync.Pool
	maxQueuePool *sync.Pool
	visitedPool  *sync.Pool
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

	if opts.InitialArenaSize == 0 {
		opts.InitialArenaSize = 64 * 1024 * 1024
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
		// segments initialized to nil (atomic.Pointer zero value)
		freeList:     make([]uint64, 0),
		vectors:      opts.Vectors,
		shardID:      0,
		numShards:    1,
		arena:        arena.NewFlat(opts.InitialArenaSize),
		layout:       newNodeLayout(opts.M),
		shardedLocks: make([]sync.RWMutex, 1024),
		minQueuePool: &sync.Pool{
			New: func() any { return queue.NewMin(opts.EF) },
		},
		maxQueuePool: &sync.Pool{
			New: func() any { return queue.NewMax(opts.EF) },
		},
		visitedPool: &sync.Pool{
			New: func() any { return visited.New(1024) },
		},
		tombstones: bitset.New(1024),
	}

	h.dimensionAtomic.Store(int32(opts.Dimension))
	if h.vectors == nil {
		h.vectors = columnar.New(opts.Dimension)
	}

	// Initialize first segment
	h.growSegments(0)

	return h, nil
}

// getNodeOffset returns the offset for the given ID, or 0 if not found.
func (h *HNSW) getNodeOffset(id uint64) uint64 {
	// Fast path for mmap'd index
	if len(h.mmapOffsets) > 0 {
		if id >= uint64(len(h.mmapOffsets)) {
			return 0
		}
		return h.mmapOffsets[id]
	}

	segments := h.segments.Load()
	if segments == nil {
		return 0
	}

	segmentIdx := int(id >> nodeSegmentBits)
	if segmentIdx >= len(*segments) {
		return 0
	}

	segment := (*segments)[segmentIdx]
	if segment == nil {
		return 0
	}

	return segment[id&nodeSegmentMask].Load()
}

// setNodeOffset sets the offset for the given ID.
func (h *HNSW) setNodeOffset(id uint64, offset uint64) {
	h.growSegments(id)

	segments := h.segments.Load()
	segmentIdx := int(id >> nodeSegmentBits)
	segment := (*segments)[segmentIdx]

	segment[id&nodeSegmentMask].Store(offset)
}

// growSegments ensures capacity for the given ID.
// Uses Copy-On-Write (COW) for lock-free growth.
func (h *HNSW) growSegments(id uint64) {
	segmentIdx := int(id >> nodeSegmentBits)

	// Fast path: check if segment exists
	segments := h.segments.Load()
	if segments != nil && segmentIdx < len(*segments) && (*segments)[segmentIdx] != nil {
		return
	}

	// Slow path: grow using CAS loop
	for {
		oldSegments := h.segments.Load()
		var newSegments []*OffsetSegment

		currentLen := 0
		if oldSegments != nil {
			currentLen = len(*oldSegments)
		}

		if segmentIdx < currentLen && (*oldSegments)[segmentIdx] != nil {
			return // Already grown by someone else
		}

		// Create new slice with enough capacity
		newLen := segmentIdx + 1
		if newLen < currentLen {
			newLen = currentLen
		}
		newSegments = make([]*OffsetSegment, newLen)

		// Copy existing segments
		if oldSegments != nil {
			copy(newSegments, *oldSegments)
		}

		// Allocate new segment
		if newSegments[segmentIdx] == nil {
			newSegments[segmentIdx] = new(OffsetSegment)
		}

		// CAS
		newSegmentsPtr := new([]*OffsetSegment)
		*newSegmentsPtr = newSegments

		if h.segments.CompareAndSwap(oldSegments, newSegmentsPtr) {
			return
		}
		// Retry
	}
}

// Helper methods for node access

func (h *HNSW) getLevel(id uint64) int {
	offset := h.getNodeOffset(id)
	if offset == 0 {
		return -1
	}
	buf := h.arena.Buffer()
	return h.layout.getLevel(buf[int(offset):])
}

func (h *HNSW) getConnections(id uint64, layer int) []uint64 {
	h.shardedLocks[id%uint64(len(h.shardedLocks))].RLock()
	defer h.shardedLocks[id%uint64(len(h.shardedLocks))].RUnlock()

	offset := h.getNodeOffset(id)
	if offset == 0 {
		return nil
	}
	buf := h.arena.Buffer()
	level := h.layout.getLevel(buf[int(offset):])
	return h.layout.getNeighbors(buf[int(offset):], level, layer)
}

func (h *HNSW) setConnections(id uint64, layer int, conns []uint64) {
	offset := h.getNodeOffset(id)
	if offset == 0 {
		return
	}
	buf := h.arena.Buffer()

	h.layout.setLayerCount(buf[int(offset):], layer, uint32(len(conns)))

	_, neighborsOff := h.layout.layerOffsets(0, layer)
	dest := buf[int(offset)+int(neighborsOff):]

	for i, c := range conns {
		binary.LittleEndian.PutUint64(dest[i*8:], c)
	}
}

func (h *HNSW) addConnection(sourceID, targetID uint64, level int) {
	h.shardedLocks[sourceID%uint64(len(h.shardedLocks))].Lock()
	defer h.shardedLocks[sourceID%uint64(len(h.shardedLocks))].Unlock()

	offset := h.getNodeOffset(sourceID)
	if offset == 0 {
		return
	}

	buf := h.arena.Buffer()
	sourceLevel := h.layout.getLevel(buf[int(offset):])
	if level > sourceLevel {
		return // Should not happen
	}

	conns := h.layout.getNeighbors(buf[int(offset):], sourceLevel, level)

	// Check if already connected
	for _, c := range conns {
		if c == targetID {
			return
		}
	}

	maxM := h.maxConnectionsPerLayer
	if level == 0 {
		maxM = h.maxConnectionsLayer0
	}

	if len(conns) < maxM {
		// Just append
		newConns := append(conns, targetID)
		h.setConnections(sourceID, level, newConns)
	} else {
		// Prune
		candidates := h.maxQueuePool.Get().(*queue.PriorityQueue)
		candidates.Reset()
		defer h.maxQueuePool.Put(candidates)

		// Add existing
		vSource, ok := h.vectors.GetVector(sourceID)
		if !ok {
			return
		}

		for _, c := range conns {
			d := h.dist(vSource, c)
			candidates.PushItem(queue.PriorityQueueItem{Node: c, Distance: d})
		}

		// Add new one
		d := h.dist(vSource, targetID)
		candidates.PushItem(queue.PriorityQueueItem{Node: targetID, Distance: d})

		// Select best
		newNeighbors := h.selectNeighbors(candidates, maxM)
		h.setConnections(sourceID, level, newNeighbors)
	}
}

// AllocateID returns a new ID.
func (h *HNSW) AllocateID() uint64 {
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
func (h *HNSW) ReleaseID(id uint64) {
	if h.getNodeOffset(id) != 0 {
		return // Don't release live IDs
	}
	h.freeListMu.Lock()
	h.freeList = append(h.freeList, id)
	h.freeListMu.Unlock()
}

// Insert inserts a vector.
func (h *HNSW) Insert(ctx context.Context, v []float32) (uint64, error) {
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	return h.insert(v, 0, -1, false)
}

// ApplyInsert inserts a vector with a specific ID (for WAL replay).
func (h *HNSW) ApplyInsert(ctx context.Context, id uint64, v []float32) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	_, err := h.insert(v, id, -1, true)
	return err
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
func (h *HNSW) insert(v []float32, id uint64, layer int, useProvidedID bool) (uint64, error) {
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
			h.rngMu.Lock()
			r := h.rng.Float64()
			h.rngMu.Unlock()
			layer = int(math.Floor(-math.Log(r) * h.layerMultiplier))
		}
	}

	// Create Node using Arena
	nodeSize := h.layout.Size(layer)
	offset, err := h.arena.Alloc(uint64(nodeSize))
	if err != nil {
		return 0, err
	}

	// Initialize fields
	buf := h.arena.Buffer()
	binary.LittleEndian.PutUint64(buf[int(offset)+nodeIDOffset:], id)
	binary.LittleEndian.PutUint32(buf[int(offset)+nodeLevelOffset:], uint32(layer))

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
			h.setNodeOffset(id, offset)
			h.entryPointAtomic.Store(id)
			h.maxLevelAtomic.Store(int32(layer))
			h.countAtomic.Add(1)
			h.epMu.Unlock()
			return id, nil
		}
		h.epMu.Unlock()
	}

	// Publish node so it can be found
	h.setNodeOffset(id, offset)

	// Insert into Graph
	if err := h.insertNode(id, vec, layer); err != nil {
		return 0, err
	}

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
func (h *HNSW) insertNode(id uint64, vec []float32, layer int) error {
	epID := h.entryPointAtomic.Load()

	// Handle case where entry point was deleted concurrently
	if h.getNodeOffset(epID) == 0 {
		return index.ErrEntryPointDeleted
	}

	currID := epID
	currDist := h.dist(vec, currID)

	// 1. Greedy search from top to node.Layer + 1
	maxLevel := int(h.maxLevelAtomic.Load())
	for level := maxLevel; level > layer; level-- {
		changed := true
		for changed {
			changed = false
			conns := h.getConnections(currID, level)
			for _, nextID := range conns {
				nextDist := h.dist(vec, nextID)
				if nextDist < currDist {
					currID = nextID
					currDist = nextDist
					changed = true
				}
			}
		}
	}

	// 2. Search and link from node.Layer down to 0
	for level := min(layer, maxLevel); level >= 0; level-- {
		// Search layer (no filtering during insertion)
		candidates, err := h.searchLayer(vec, currID, currDist, level, h.opts.EF, nil)
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

		neighbors := h.selectNeighbors(candidates, maxConns)

		// Put back candidates (results from searchLayer)
		candidates.Reset()
		h.maxQueuePool.Put(candidates)

		// Add bidirectional connections
		h.shardedLocks[id%uint64(len(h.shardedLocks))].Lock()
		h.setConnections(id, level, neighbors)
		h.shardedLocks[id%uint64(len(h.shardedLocks))].Unlock()

		for _, neighborID := range neighbors {
			h.addConnection(neighborID, id, level)
		}
	}

	return nil
}

// selectNeighbors selects the best neighbors from candidates.
func (h *HNSW) selectNeighbors(candidates *queue.PriorityQueue, m int) []uint64 {
	if h.opts.Heuristic {
		return h.selectNeighborsHeuristic(candidates, m)
	}
	return h.selectNeighborsSimple(candidates, m)
}

func (h *HNSW) selectNeighborsSimple(candidates *queue.PriorityQueue, m int) []uint64 {
	// Simple selection (keep top M)
	for candidates.Len() > m {
		_, _ = candidates.PopItem()
	}

	res := make([]uint64, 0, candidates.Len())
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

func (h *HNSW) selectNeighborsHeuristic(candidates *queue.PriorityQueue, m int) []uint64 {
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

	result := make([]uint64, 0, m)
	resultVecs := make([][]float32, 0, m)

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
			result = append(result, cand.Node)
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
func (h *HNSW) searchLayer(query []float32, epID uint64, epDist float32, level int, ef int, filter func(uint64) bool) (*queue.PriorityQueue, error) {
	visited := h.visitedPool.Get().(*visited.VisitedSet)
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

	visited.Visit(epID)

	// CRITICAL: Always add entry point to candidates for navigation (even if filtered)
	// This ensures we have a starting point for graph traversal
	candidates.PushItem(queue.PriorityQueueItem{Node: epID, Distance: epDist})

	// Only add to results if it passes the filter AND is not deleted
	if (filter == nil || filter(epID)) && !h.tombstones.Test(epID) {
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

		conns := h.getConnections(curr.Node, level)
		for _, nextID := range conns {
			if !visited.Visited(nextID) {
				visited.Visit(nextID)

				nextDist := h.dist(query, nextID)

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
					candidates.PushItem(queue.PriorityQueueItem{Node: nextID, Distance: nextDist})

					// Only add to results if it passes the filter AND is not deleted
					if (filter == nil || filter(nextID)) && !h.tombstones.Test(nextID) {
						results.PushItem(queue.PriorityQueueItem{Node: nextID, Distance: nextDist})
						if results.Len() > ef {
							_, _ = results.PopItem()
						}
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

	// Ensure bitset capacity
	h.tombstones.Grow(id + 1)
	if h.tombstones.Test(id) {
		return nil
	}
	h.tombstones.Set(id)

	// We do NOT remove from graph or release ID.
	// This preserves graph connectivity and avoids O(N) entry point scans.
	// The node remains in the graph but is ignored during search.

	// We can optionally delete the vector data to save memory,
	// but keeping it might be safer for concurrent readers.
	// For now, we keep it.

	h.countAtomic.Add(-1)
	return nil
}

func (h *HNSW) removeLink(id uint64, neighborID uint64, layer int) {
	h.shardedLocks[id%uint64(len(h.shardedLocks))].Lock()
	defer h.shardedLocks[id%uint64(len(h.shardedLocks))].Unlock()

	offset := h.getNodeOffset(id)
	if offset == 0 {
		return
	}
	buf := h.arena.Buffer()
	level := h.layout.getLevel(buf[int(offset):])
	conns := h.layout.getNeighbors(buf[int(offset):], level, layer)

	for i, c := range conns {
		if c == neighborID {
			// Remove
			// Swap with last
			conns[i] = conns[len(conns)-1]
			conns = conns[:len(conns)-1]
			h.setConnections(id, layer, conns)
			return
		}
	}
}

// Update updates a vector.
func (h *HNSW) Update(ctx context.Context, id uint64, v []float32) error {
	offset := h.getNodeOffset(id)
	if offset == 0 {
		return &index.ErrNodeNotFound{ID: id}
	}
	// Read layer
	buf := h.arena.Buffer()
	layer := h.layout.getLevel(buf[int(offset):])

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
	if h.getNodeOffset(epID) == 0 {
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
	currID := epID
	currDist := h.dist(q, currID)
	maxLevel := int(h.maxLevelAtomic.Load())

	for level := maxLevel; level > 0; level-- {
		changed := true
		for changed {
			changed = false
			conns := h.getConnections(currID, level)
			for _, nextID := range conns {
				nextDist := h.dist(q, nextID)
				if nextDist < currDist {
					currID = nextID
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

	results, err := h.searchLayer(q, currID, currDist, 0, ef, filter)
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
func (h *HNSW) BruteSearch(ctx context.Context, query []float32, k int, filter func(id uint64) bool) ([]index.SearchResult, error) {
	// Simple scan
	pq := queue.NewMax(k)

	segments := h.segments.Load()
	if segments != nil {
		for i, seg := range *segments {
			if seg == nil {
				continue
			}
			for j := range seg {
				offset := seg[j].Load()
				if offset == 0 {
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
func (h *HNSW) VectorCount() int { return int(h.countAtomic.Load()) }
func (h *HNSW) ContainsID(id uint64) bool {
	if h.tombstones.Test(id) {
		return false
	}
	return h.getNodeOffset(id) != 0
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

// Dimension returns the dimensionality of the vectors in the index.
func (h *HNSW) Dimension() int {
	return int(h.dimensionAtomic.Load())
}
