package hnsw

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"math"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"github.com/hupe1980/vecgo/distance"
	"github.com/hupe1980/vecgo/internal/arena"
	"github.com/hupe1980/vecgo/internal/bitset"
	"github.com/hupe1980/vecgo/internal/conv"
	"github.com/hupe1980/vecgo/internal/searcher"
	"github.com/hupe1980/vecgo/internal/segment"
	"github.com/hupe1980/vecgo/internal/vectorstore"
	"github.com/hupe1980/vecgo/model"
)

const (
	// layerNormalizationBase is the base constant for exponential layer probability distribution.
	layerNormalizationBase = 1.0

	// mmax0Multiplier is the multiplier for calculating maximum connections at layer 0.
	mmax0Multiplier = 2

	// minimumM is the minimum valid value for M.
	minimumM = 2

	// DefaultM is the default number of bidirectional links.
	DefaultM = 32

	// DefaultEF is the default size of the dynamic candidate list.
	DefaultEF = 300

	// nodeSegmentSize is the size of each node segment (65536).
	// Using segments avoids copying the entire node array during growth.
	nodeSegmentBits = 16
	nodeSegmentSize = 1 << nodeSegmentBits
	nodeSegmentMask = nodeSegmentSize - 1
)

// OffsetSegment is a fixed-size array of node offsets.
type OffsetSegment [nodeSegmentSize]atomic.Uint64

// NodeSegment is a fixed-size array of node pointers.
type NodeSegment [nodeSegmentSize]atomic.Uint64

// Options represents the options for configuring HNSW.
type Options struct {
	Dimension        int
	M                int
	EF               int
	Heuristic        bool
	DistanceType     distance.Metric
	NormalizeVectors bool
	Vectors          vectorstore.VectorStore
	InitialArenaSize int
	RandomSeed       *int64
	MemoryAcquirer   arena.MemoryAcquirer
}

// DefaultOptions contains the default options for HNSW.
var DefaultOptions = Options{
	Dimension:        0,
	M:                DefaultM,
	EF:               DefaultEF,
	Heuristic:        true,
	DistanceType:     distance.MetricL2,
	InitialArenaSize: 1024 * 1024 * 1024, // 1GB
}

// graph holds the mutable state of the HNSW index.
type graph struct {
	// Hot path: Atomic fields
	entryPointAtomic atomic.Uint32
	maxLevelAtomic   atomic.Int32
	nextIDAtomic     atomic.Uint32
	countAtomic      atomic.Int64 // Track live count

	// Node storage: Segmented array of nodes
	nodes   atomic.Pointer[[]*NodeSegment]
	nodesMu sync.Mutex // Protects nodes growth

	// Memory Arena
	arena *arena.Arena

	// Sharded locks for node updates
	shardedLocks []sync.RWMutex

	// State
	tombstones *bitset.BitSet
}

func (g *graph) Size() int64 {
	if g.arena == nil {
		return 0
	}
	return int64(g.arena.Stats().BytesReserved)
}

// HNSW represents the Hierarchical Navigable Small World graph.
type HNSW struct {
	// Graph state (RCU-style)
	currentGraph atomic.Pointer[graph]

	// Components
	dimensionAtomic atomic.Int32
	distanceFunc    distance.Func
	vectors         vectorstore.VectorStore
	distOp          func(model.RowID, []float32) float32 // Optimized distance calc
	rngSeed         atomic.Uint64                        // Lock-free RNG seed

	// Configuration
	maxConnectionsPerLayer int
	maxConnectionsLayer0   int
	layerMultiplier        float64
	opts                   Options

	// Sharding
	shardID   int
	numShards int

	// Resources
	scratchPool *sync.Pool
}

// Close releases resources associated with the HNSW index.
func (h *HNSW) Close() error {
	g := h.currentGraph.Load()
	if g != nil && g.arena != nil {
		g.arena.Free()
	}
	return nil
}

// DistFunc computes the distance from a query/source vector to a target node ID.
type DistFunc func(id model.RowID) float32

type scratch struct {
	floats  []float32
	results []SearchResult

	// Heuristic scratch buffers
	heuristicCandidates []searcher.PriorityQueueItem
	heuristicResult     []searcher.PriorityQueueItem
	heuristicResultVecs [][]float32

	// Connection scratch buffer
	connections []Neighbor
}

// Name returns the name of the index.
func (*HNSW) Name() string { return "HNSW" }

// Reset clears the HNSW index for reuse.
func (h *HNSW) Reset() error {
	g, err := newGraph(h.opts)
	if err != nil {
		return err
	}
	h.currentGraph.Store(g)

	// Reset vectors if supported
	if reseter, ok := h.vectors.(interface{ Reset() }); ok {
		reseter.Reset()
	}
	return nil
}

// New creates a new HNSW instance.
func New(optFns ...func(o *Options)) (*HNSW, error) {
	opts := DefaultOptions
	for _, fn := range optFns {
		fn(&opts)
	}

	if opts.Dimension <= 0 {
		return nil, &ErrInvalidDimension{Dimension: opts.Dimension}
	}

	if opts.DistanceType == distance.MetricCosine {
		opts.NormalizeVectors = true
	}

	if opts.M < minimumM {
		opts.M = minimumM
	}

	if opts.InitialArenaSize == 0 {
		opts.InitialArenaSize = 1024 * 1024 * 1024 // 1GB
	}

	var rngSeed uint64
	if opts.RandomSeed != nil {
		rngSeed = uint64(*opts.RandomSeed)
	} else {
		rngSeed = uint64(time.Now().UnixNano())
	}

	distFunc, err := newDistanceFunc(opts.DistanceType)
	if err != nil {
		return nil, err
	}

	h := &HNSW{
		maxConnectionsPerLayer: opts.M,
		maxConnectionsLayer0:   mmax0Multiplier * opts.M,
		layerMultiplier:        layerNormalizationBase / math.Log(float64(opts.M)),
		distanceFunc:           distFunc,
		opts:                   opts,
		vectors:                opts.Vectors,
		shardID:                0,
		numShards:              1,
	}
	h.rngSeed.Store(rngSeed)
	g, err := newGraph(opts)
	if err != nil {
		return nil, err
	}
	h.currentGraph.Store(g)

	h.initPools()

	dimI32, err := conv.IntToInt32(opts.Dimension)
	if err != nil {
		return nil, err
	}
	h.dimensionAtomic.Store(dimI32)
	if h.vectors == nil {
		var err error
		h.vectors, err = vectorstore.New(opts.Dimension, opts.MemoryAcquirer)
		if err != nil {
			return nil, err
		}
	}

	// Optimize distance calculation
	if cs, ok := h.vectors.(interface {
		OptimizedDistanceComputer(distance.Metric) (func(model.RowID, []float32) float32, bool)
	}); ok {
		if fn, ok := cs.OptimizedDistanceComputer(opts.DistanceType); ok {
			h.distOp = fn
		}
	}
	if h.distOp == nil {
		h.distOp = func(id model.RowID, q []float32) float32 {
			d, ok := h.vectors.ComputeDistance(id, q, opts.DistanceType)
			if !ok {
				return math.MaxFloat32
			}
			return d
		}
	}

	return h, nil
}

func newGraph(opts Options) (*graph, error) {
	var arenaOpts []arena.Option
	if opts.MemoryAcquirer != nil {
		arenaOpts = append(arenaOpts, arena.WithMemoryAcquirer(opts.MemoryAcquirer))
	}
	a, err := arena.New(opts.InitialArenaSize, arenaOpts...)
	if err != nil {
		return nil, err
	}

	g := &graph{
		shardedLocks: make([]sync.RWMutex, 1024),
		tombstones:   bitset.New(1024),
		arena:        a,
	}
	g.maxLevelAtomic.Store(-1)
	// Initialize first segment
	nodes := make([]*NodeSegment, 1)
	nodes[0] = new(NodeSegment)
	g.nodes.Store(&nodes)
	return g, nil
}

func (h *HNSW) newNode(ctx context.Context, g *graph, level int) (Node, error) {
	n, err := AllocNode(ctx, g.arena, level, h.maxConnectionsPerLayer, h.maxConnectionsLayer0)
	if err != nil {
		return Node{}, err
	}
	return n, nil
}

func (h *HNSW) initPools() {
	// Pre-calculate max possible connections to size scratch buffers correctly
	maxConns := h.maxConnectionsPerLayer
	if h.maxConnectionsLayer0 > maxConns {
		maxConns = h.maxConnectionsLayer0
	}

	h.scratchPool = &sync.Pool{
		New: func() any {
			return &scratch{
				floats:              make([]float32, h.opts.Dimension),
				results:             make([]SearchResult, 0, h.opts.EF),
				heuristicCandidates: make([]searcher.PriorityQueueItem, 0, h.opts.EF),
				heuristicResult:     make([]searcher.PriorityQueueItem, 0, maxConns),
				heuristicResultVecs: make([][]float32, 0, maxConns),
				connections:         make([]Neighbor, 0, maxConns),
			}
		},
	}
}

// getNode returns the node for the given ID.
// Returns a zero Node if not found (check using node.IsZero()).
// This returns a value type to avoid heap allocation.
func (h *HNSW) getNode(g *graph, id model.RowID) Node {
	nodes := g.nodes.Load()
	if nodes == nil {
		return Node{}
	}

	segmentIdx := int(id >> nodeSegmentBits)
	if segmentIdx >= len(*nodes) {
		return Node{}
	}

	segment := (*nodes)[segmentIdx]
	if segment == nil {
		return Node{}
	}

	val := segment[id&nodeSegmentMask].Load()
	return Node{ref: nodeRef(val)}
}

// setNode sets the node for the given ID.
func (h *HNSW) setNode(g *graph, id model.RowID, node Node) {
	h.growNodes(g, id)

	nodes := g.nodes.Load()
	segmentIdx := int(id >> nodeSegmentBits)
	segment := (*nodes)[segmentIdx]

	segment[id&nodeSegmentMask].Store(uint64(node.ref))
}

// growNodes ensures capacity for the given ID.
func (h *HNSW) growNodes(g *graph, id model.RowID) {
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

func (h *HNSW) getConnectionsBuf(g *graph, id model.RowID, layer int, buf []Neighbor) []Neighbor {
	node := h.getNode(g, id)
	if node.IsZero() {
		return buf[:0]
	}

	count := node.GetCount(g.arena, layer, h.maxConnectionsPerLayer, h.maxConnectionsLayer0)
	if cap(buf) < count {
		// If buffer is too small, allocate a new one.
		// We don't want to grow the scratch buffer permanently if we hit a very dense node,
		// but for HNSW max connections are bounded by M/M0.
		// So this allocation should be rare if scratch is sized correctly.
		buf = make([]Neighbor, count)
	}
	buf = buf[:count]
	for i := 0; i < count; i++ {
		buf[i] = node.GetConnection(g.arena, layer, i, h.maxConnectionsPerLayer, h.maxConnectionsLayer0)
	}
	return buf
}

func (h *HNSW) getConnections(g *graph, id model.RowID, layer int) []Neighbor {
	return h.getConnectionsBuf(g, id, layer, nil)
}

// visitConnections iterates over the connections of a node without allocating a slice.
// It returns true if the iteration completed, or false if the callback stopped it.
func (h *HNSW) visitConnections(g *graph, id model.RowID, layer int, callback func(Neighbor) bool) {
	node := h.getNode(g, id)
	if node.IsZero() {
		return
	}

	raw := node.GetConnectionsRaw(g.arena, layer, h.maxConnectionsPerLayer, h.maxConnectionsLayer0)
	for i := 0; i < len(raw); i++ {
		v := atomic.LoadUint64(&raw[i])
		n := NeighborFromUint64(v)
		if !callback(n) {
			return
		}
	}
}

func (h *HNSW) setConnections(ctx context.Context, g *graph, id model.RowID, layer int, conns []Neighbor) error {
	node := h.getNode(g, id)
	if node.IsZero() {
		return nil
	}

	// Use COW replacement to ensure atomic visibility
	return node.ReplaceConnections(ctx, g.arena, layer, conns, h.maxConnectionsPerLayer, h.maxConnectionsLayer0)
}

func (h *HNSW) addConnection(ctx context.Context, s *searcher.Searcher, scratch *scratch, g *graph, sourceID, targetID model.RowID, level int, dist float32) error {
	g.shardedLocks[uint64(sourceID)%uint64(len(g.shardedLocks))].Lock()
	defer g.shardedLocks[uint64(sourceID)%uint64(len(g.shardedLocks))].Unlock()

	node := h.getNode(g, sourceID)
	if node.IsZero() {
		return nil
	}

	if level > node.Level(g.arena) {
		return nil // Should not happen
	}

	conns := h.getConnectionsBuf(g, sourceID, level, scratch.connections)

	// Check if already connected using binary search (conns is sorted)
	// We assume conns is sorted. If not, we should sort it or use linear scan.
	// For now, let's use linear scan but optimized.
	// Actually, let's enforce sorting on insert.

	// Linear scan is fine for small M (8-32).
	// slices.Contains is optimized in Go 1.21+.
	// Optimization: Use manual loop to avoid generic overhead
	found := false
	for _, c := range conns {
		if c.ID == targetID {
			found = true
			break
		}
	}
	if found {
		return nil
	}

	maxM := h.maxConnectionsPerLayer
	if level == 0 {
		maxM = h.maxConnectionsLayer0
	}

	if len(conns) < maxM {
		return h.addConnectionSimple(ctx, g, sourceID, level, conns, targetID, dist)
	}

	return h.addConnectionPrune(ctx, s, scratch, g, sourceID, level, conns, targetID, dist, maxM)
}

func (h *HNSW) addConnectionSimple(ctx context.Context, g *graph, sourceID model.RowID, level int, conns []Neighbor, targetID model.RowID, dist float32) error {
	// Try optimized Append
	node := h.getNode(g, sourceID)
	if node.IsZero() {
		return nil
	}

	if ok := node.AppendConnection(g.arena, level, Neighbor{ID: targetID, Dist: dist}, h.maxConnectionsPerLayer, h.maxConnectionsLayer0); ok {
		return nil
	}

	// Fallback to COW if Append failed (e.g. race condition filled it up or something else)
	// This ensures robustness.

	var newConns []Neighbor
	if cap(conns) >= len(conns)+1 {
		newConns = conns[:len(conns)+1]
	} else {
		newConns = make([]Neighbor, len(conns)+1)
		copy(newConns, conns)
	}
	newConns[len(conns)] = Neighbor{ID: targetID, Dist: dist}
	return h.setConnections(ctx, g, sourceID, level, newConns)
}

func (h *HNSW) addConnectionPrune(ctx context.Context, s *searcher.Searcher, scratch *scratch, g *graph, sourceID model.RowID, level int, conns []Neighbor, targetID model.RowID, dist float32, maxM int) error {
	// Prune
	candidates := s.Candidates
	candidates.Reset()

	// Add existing - use cached distances!
	for _, c := range conns {
		candidates.PushItem(searcher.PriorityQueueItem{Node: c.ID, Distance: c.Dist})
	}
	// Add new
	candidates.PushItem(searcher.PriorityQueueItem{Node: targetID, Distance: dist})

	// Use scratch for selectNeighbors
	neighbors := h.selectNeighbors(candidates, maxM, scratch)

	// Reuse scratch.connections for finalConns if possible, or allocate if needed.
	// But wait, setConnections copies data. So we can use scratch.connections!
	// We need to be careful not to overwrite what we are reading if we were reading from scratch.connections.
	// In this block, 'conns' (the input) is backed by scratch.connections.
	// 'neighbors' is backed by scratch.heuristicResult.
	// So we can safely reuse scratch.connections for the output to setConnections,
	// effectively overwriting the input 'conns' which we don't need anymore.

	// Ensure scratch.connections has enough capacity (it should, maxM)
	if cap(scratch.connections) < len(neighbors) {
		scratch.connections = make([]Neighbor, len(neighbors), maxM)
	}
	finalConns := scratch.connections[:len(neighbors)]

	for i, n := range neighbors {
		finalConns[i] = Neighbor{ID: n.Node, Dist: n.Distance}
	}

	return h.setConnections(ctx, g, sourceID, level, finalConns)
}

// AllocateID returns a new ID.
func (h *HNSW) AllocateID() model.RowID {
	return h.allocateID(h.currentGraph.Load())
}

func (h *HNSW) allocateID(g *graph) model.RowID {
	return model.RowID(g.nextIDAtomic.Add(1) - 1)
}

// ReleaseID releases an ID.
func (h *HNSW) ReleaseID(id model.RowID) {
	// No-op: IDs are never reused to ensure stability.
}

func (h *HNSW) releaseID(g *graph, id model.RowID) {
	// No-op: IDs are never reused to ensure stability.
}

// Insert inserts a vector.
func (h *HNSW) Insert(ctx context.Context, v []float32) (model.RowID, error) {
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	g := h.currentGraph.Load()
	g.arena.IncRef()
	defer g.arena.DecRef()
	return h.insert(ctx, g, v, 0, -1, false)
}

// InsertDeferred inserts a vector without building the graph connections (Bulk Load).
// This is significantly faster (~10x) but the inserted vector will NOT be found in search results
// until the MemTable is flushed and compacted. The Flush operation (which creates a Flat segment)
// works correctly because it iterates vectors by ID, which are preserved.
func (h *HNSW) InsertDeferred(ctx context.Context, v []float32) (model.RowID, error) {
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	g := h.currentGraph.Load()
	g.arena.IncRef()
	defer g.arena.DecRef()

	vec, err := h.prepareVector(v)
	if err != nil {
		return 0, err
	}

	// Always allocate new ID
	id := h.allocateID(g)

	// Force layer 0 for minimal memory usage since we discard graph on Flush
	layer := 0

	node, err := h.newNode(ctx, g, layer)
	if err != nil {
		h.releaseID(g, id)
		return 0, err
	}

	if err := h.vectors.SetVector(ctx, id, vec); err != nil {
		h.releaseID(g, id)
		return 0, err
	}

	// Publish node so it is visible to iteration (Flush) and ContainsID
	h.setNode(g, id, node)
	g.countAtomic.Add(1)

	return id, nil
}

// ApplyInsert inserts a vector with a specific ID (for WAL replay).
func (h *HNSW) ApplyInsert(ctx context.Context, id model.RowID, v []float32) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	g := h.currentGraph.Load()
	g.arena.IncRef()
	defer g.arena.DecRef()
	_, err := h.insert(ctx, g, v, id, -1, true)
	return err
}

// ApplyBatchInsert inserts multiple vectors with specific IDs concurrently.
func (h *HNSW) ApplyBatchInsert(ctx context.Context, ids []model.RowID, vectors [][]float32) error {
	if len(ids) != len(vectors) {
		return fmt.Errorf("ids and vectors length mismatch")
	}
	if err := ctx.Err(); err != nil {
		return err
	}

	g := h.currentGraph.Load()
	g.arena.IncRef()
	defer g.arena.DecRef()

	var wg sync.WaitGroup

	// Limit concurrency to avoid overwhelming the system
	concurrency := runtime.GOMAXPROCS(0)
	if len(vectors) < concurrency {
		concurrency = len(vectors)
	}

	var firstErr error
	var errMu sync.Mutex
	var idx atomic.Int64

	wg.Add(concurrency)
	for i := 0; i < concurrency; i++ {
		go func() {
			defer wg.Done()
			for {
				i := int(idx.Add(1) - 1)
				if i >= len(vectors) {
					return
				}

				if _, err := h.insert(ctx, g, vectors[i], ids[i], -1, true); err != nil {
					errMu.Lock()
					if firstErr == nil {
						firstErr = err
					}
					errMu.Unlock()
				}
			}
		}()
	}
	wg.Wait()
	return firstErr
}

// ApplyDelete deletes a node with a specific ID (for WAL replay).
func (h *HNSW) ApplyDelete(ctx context.Context, id model.RowID) error {
	return h.Delete(ctx, id)
}

// ApplyUpdate updates a node with a specific ID (for WAL replay).
func (h *HNSW) ApplyUpdate(ctx context.Context, id model.RowID, v []float32) error {
	return h.Update(ctx, id, v)
}

// BatchInsert inserts multiple vectors.
func (h *HNSW) BatchInsert(ctx context.Context, vectors [][]float32) BatchInsertResult {
	result := BatchInsertResult{
		IDs:    make([]model.RowID, len(vectors)),
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
func (h *HNSW) insert(ctx context.Context, g *graph, v []float32, id model.RowID, layer int, useProvidedID bool) (model.RowID, error) {
	vec, err := h.prepareVector(v)
	if err != nil {
		return 0, err
	}

	id = h.allocateOrValidateID(g, id, useProvidedID)

	if useProvidedID {
		// Ensure any previous tombstone is cleared (e.g. during Update or WAL replay)
		g.tombstones.Unset(uint32(id))
	}

	layer = h.determineLayer(id, layer, useProvidedID)

	node, err := h.newNode(ctx, g, layer)
	if err != nil {
		if !useProvidedID {
			h.releaseID(g, id)
		}
		return 0, err
	}

	// If updating an existing node, preserve its connections to maintain graph connectivity
	// during the update. The new node will start with the old connections, which effectively
	// serves as finding the "entry point" for the insertion search from the node's previous location.
	if useProvidedID {
		if oldNode := h.getNode(g, id); !oldNode.IsZero() {
			oldLevel := oldNode.Level(g.arena)
			minLevel := layer
			if oldLevel < minLevel {
				minLevel = oldLevel
			}

			// Copy connections from old node to new node
			for l := 0; l <= minLevel; l++ {
				conns := h.getConnections(g, id, l)
				node.SetCount(g.arena, l, len(conns), h.maxConnectionsPerLayer, h.maxConnectionsLayer0)
				for i, c := range conns {
					node.SetConnection(g.arena, l, i, c, h.maxConnectionsPerLayer, h.maxConnectionsLayer0)
				}
			}
		}
	}

	if err := h.vectors.SetVector(ctx, id, vec); err != nil {
		if !useProvidedID {
			h.releaseID(g, id)
		}
		return 0, err
	}

	// Publish node so it can be found
	h.setNode(g, id, node)

	// Create distFunc for insertion traversal
	var distFunc DistFunc
	if snapshottable, ok := h.vectors.(interface {
		Snapshot() vectorstore.VectorSnapshot
	}); ok {
		snap := snapshottable.Snapshot()
		distFunc = func(nid model.RowID) float32 {
			d, _ := snap.ComputeDistance(nid, vec, h.opts.DistanceType)
			return d
		}
	} else {
		distFunc = func(nid model.RowID) float32 {
			return h.dist(vec, nid)
		}
	}

	wasFirst, err := h.performInsertion(ctx, g, id, vec, layer, distFunc)
	if err != nil {
		return 0, err
	}

	if !wasFirst {
		g.countAtomic.Add(1)
		h.updateEntryPoint(g, id, layer)
	}

	return id, nil
}

func (h *HNSW) prepareVector(v []float32) ([]float32, error) {
	if len(v) == 0 {
		return nil, ErrEmptyVector
	}
	dim := int(h.dimensionAtomic.Load())
	if len(v) != dim {
		return nil, &ErrDimensionMismatch{Expected: dim, Actual: len(v)}
	}

	// Normalize if needed
	if h.opts.NormalizeVectors {
		vec := make([]float32, len(v))
		copy(vec, v)
		if !distance.NormalizeL2InPlace(vec) {
			return nil, fmt.Errorf("hnsw: cannot normalize zero vector")
		}
		return vec, nil
	}
	return v, nil // Zero-copy if not normalizing
}

func (h *HNSW) allocateOrValidateID(g *graph, id model.RowID, useProvidedID bool) model.RowID {
	if useProvidedID {
		// Ensure nextID > id
		for {
			cur := g.nextIDAtomic.Load()
			if cur > uint32(id) {
				break
			}
			if g.nextIDAtomic.CompareAndSwap(cur, uint32(id)+1) {
				break
			}
		}
		return id
	}
	return h.allocateID(g)
}

func (h *HNSW) determineLayer(id model.RowID, layer int, useProvidedID bool) int {
	if layer >= 0 {
		return layer
	}
	if useProvidedID {
		return h.layerForApplyInsert(uint64(id))
	}
	// Lock-free RNG using xorshift64* algorithm
	// This is much faster than mutex-protected rand.Rand
	seed := h.rngSeed.Add(0x9E3779B97F4A7C15) // Golden ratio prime
	seed ^= seed >> 12
	seed ^= seed << 25
	seed ^= seed >> 27
	r := float64(seed*0x2545F4914F6CDD1D>>11) / float64(1<<53) // Convert to [0, 1)
	return int(math.Floor(-math.Log(r) * h.layerMultiplier))
}

func (h *HNSW) performInsertion(ctx context.Context, g *graph, id model.RowID, vec []float32, layer int, distFunc DistFunc) (bool, error) {
	retries := 0
	for {
		// Handle First Node
		if g.countAtomic.Load() == 0 {
			if g.countAtomic.CompareAndSwap(0, 1) {
				g.entryPointAtomic.Store(uint32(id))
				g.maxLevelAtomic.Store(int32(layer))
				return true, nil
			}
			// Lost race, continue
		}

		// Insert into Graph
		err := h.insertNode(ctx, g, id, vec, layer, distFunc)
		if errors.Is(err, ErrEntryPointDeleted) {
			retries++
			if retries > 10 {
				h.recoverEntryPoint(g)
				retries = 0
			}
			runtime.Gosched()
			continue
		}
		if err != nil {
			return false, err
		}
		return false, nil
	}
}

func (h *HNSW) updateEntryPoint(g *graph, id model.RowID, layer int) {
	maxLevel := int(g.maxLevelAtomic.Load())
	if layer > maxLevel {
		for {
			oldMax := g.maxLevelAtomic.Load()
			if layer <= int(oldMax) {
				break
			}
			if g.maxLevelAtomic.CompareAndSwap(oldMax, int32(layer)) {
				g.entryPointAtomic.Store(uint32(id))
				break
			}
		}
	}
}

// insertNode performs the graph traversal and linking.
func (h *HNSW) insertNode(ctx context.Context, g *graph, id model.RowID, vec []float32, layer int, distFunc DistFunc) error {
	epID := g.entryPointAtomic.Load()

	// Handle case where entry point was deleted concurrently
	if h.getNode(g, model.RowID(epID)).IsZero() {
		return ErrEntryPointDeleted
	}

	currID := model.RowID(epID)
	currDist := distFunc(currID)

	// Acquire Searcher
	s := searcher.Get()
	defer searcher.Put(s)

	// 1. Greedy search from top to node.Layer + 1
	maxLevel := int(g.maxLevelAtomic.Load())
	for level := maxLevel; level > layer; level-- {
		changed := true
		for changed {
			changed = false
			h.visitConnections(g, currID, level, func(next Neighbor) bool {
				nextDist := distFunc(next.ID)
				if nextDist < currDist {
					currID = next.ID
					currDist = nextDist
					changed = true
				}
				return true
			})
		}
	}

	// 2. Search and link from node.Layer down to 0
	scratch := h.scratchPool.Get().(*scratch)
	defer h.scratchPool.Put(scratch)

	for level := min(layer, maxLevel); level >= 0; level-- {
		// Search layer (no filtering during insertion)
		h.searchLayer(s, g, vec, currID, currDist, level, h.opts.EF, nil, distFunc)
		candidates := s.Candidates

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

		// Extract IDs for setConnections - reuse scratch.connections to avoid allocation
		if cap(scratch.connections) < len(neighbors) {
			scratch.connections = make([]Neighbor, len(neighbors), maxConns)
		}
		neighborConns := scratch.connections[:len(neighbors)]
		for i, n := range neighbors {
			neighborConns[i] = Neighbor{ID: n.Node, Dist: n.Distance}
		}

		// Add bidirectional connections
		g.shardedLocks[uint64(id)%uint64(len(g.shardedLocks))].Lock()
		err := h.setConnections(ctx, g, id, level, neighborConns)
		g.shardedLocks[uint64(id)%uint64(len(g.shardedLocks))].Unlock()
		if err != nil {
			return err
		}

		for _, neighbor := range neighbors {
			if err := h.addConnection(ctx, s, scratch, g, neighbor.Node, id, level, neighbor.Distance); err != nil {
				return err
			}
		}
	}

	return nil
}

// selectNeighbors selects the best neighbors from candidates.
func (h *HNSW) selectNeighbors(candidates *searcher.PriorityQueue, m int, scratch *scratch) []searcher.PriorityQueueItem {
	if h.opts.Heuristic {
		return h.selectNeighborsHeuristic(candidates, m, scratch)
	}
	return h.selectNeighborsSimple(candidates, m, scratch)
}

func (h *HNSW) selectNeighborsSimple(candidates *searcher.PriorityQueue, m int, scratch *scratch) []searcher.PriorityQueueItem {
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

func (h *HNSW) selectNeighborsHeuristic(candidates *searcher.PriorityQueue, m int, scratch *scratch) []searcher.PriorityQueueItem {
	if candidates.Len() <= m {
		return h.selectNeighborsSimple(candidates, m, scratch) // Fallback to simple if few candidates
	}

	temp := h.extractSortedCandidates(candidates, scratch)
	result := h.applyHeuristic(temp, m, scratch)

	if len(result) < m {
		result = h.fillUpNeighbors(result, temp, m)
	}

	return result
}

func (h *HNSW) extractSortedCandidates(candidates *searcher.PriorityQueue, scratch *scratch) []searcher.PriorityQueueItem {
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

	// Update scratch buffer
	scratch.heuristicCandidates = temp

	return temp
}

func (h *HNSW) applyHeuristic(candidates []searcher.PriorityQueueItem, m int, scratch *scratch) []searcher.PriorityQueueItem {
	// Use scratch buffers for results
	result := scratch.heuristicResult[:0]
	resultVecs := scratch.heuristicResultVecs[:0]

	for _, cand := range candidates {
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

	// Update scratch buffers to reflect the new length
	scratch.heuristicResult = result
	scratch.heuristicResultVecs = resultVecs

	return result
}

func (h *HNSW) fillUpNeighbors(result []searcher.PriorityQueueItem, candidates []searcher.PriorityQueueItem, m int) []searcher.PriorityQueueItem {
	// Fill up if needed
	for _, cand := range candidates {
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
		}
	}
	return result
}

func (h *HNSW) searchLayer(s *searcher.Searcher, g *graph, query []float32, epID model.RowID, epDist float32, level int, ef int, filter segment.Filter, distFunc DistFunc) {
	h.initializeSearch(s, epID)
	h.processEntryPoint(s, g, epID, epDist, filter)

	candidates := s.ScratchCandidates
	results := s.Candidates
	visited := s.Visited

	for candidates.Len() > 0 {
		curr, _ := candidates.PopItem()

		// Termination condition: only check if we have valid results
		if results.Len() > 0 {
			worst, _ := results.TopItem()
			if curr.Distance > worst.Distance && results.Len() >= ef {
				break
			}
		}

		// Optimization: Inline visitConnections and processNeighbor logic to avoid closure overhead
		node := h.getNode(g, curr.Node)
		if !node.IsZero() {
			raw := node.GetConnectionsRaw(g.arena, level, h.maxConnectionsPerLayer, h.maxConnectionsLayer0)
			for i := 0; i < len(raw); i++ {
				v := atomic.LoadUint64(&raw[i])
				next := NeighborFromUint64(v)
				if !visited.Visited(next.ID) {
					visited.Visit(next.ID)

					// Inline processNeighbor logic
					nextDist := distFunc(next.ID)

					// Classic HNSW pruning: avoid pushing obviously-bad candidates once we already
					// have ef results. This substantially reduces heap churn.
					shouldExplore := true
					if results.Len() >= ef {
						worst, _ := results.TopItem()
						if nextDist > worst.Distance {
							shouldExplore = false
						}
					}

					if shouldExplore {
						candidates.PushItem(searcher.PriorityQueueItem{Node: next.ID, Distance: nextDist})

						// Only add to results if it passes the filter AND is not deleted
						if (filter == nil || filter.Matches(uint32(next.ID))) && !g.tombstones.Test(uint32(next.ID)) {
							// Use bounded push for results to avoid heap churn
							results.PushItemBounded(searcher.PriorityQueueItem{Node: next.ID, Distance: nextDist}, ef)
						}
					}
				}
			}
		}
	}
}

func (h *HNSW) initializeSearch(s *searcher.Searcher, epID model.RowID) {
	s.Visited.Reset()
	s.ScratchCandidates.Reset()
	s.Candidates.Reset()
	s.Visited.Visit(epID)
}

func (h *HNSW) processEntryPoint(s *searcher.Searcher, g *graph, epID model.RowID, epDist float32, filter segment.Filter) {
	// CRITICAL: Always add entry point to candidates for navigation (even if filtered)
	// This ensures we have a starting point for graph traversal
	s.ScratchCandidates.PushItem(searcher.PriorityQueueItem{Node: epID, Distance: epDist})

	// Only add to results if it passes the filter AND is not deleted
	if (filter == nil || filter.Matches(uint32(epID))) && !g.tombstones.Test(uint32(epID)) {
		s.Candidates.PushItem(searcher.PriorityQueueItem{Node: epID, Distance: epDist})
	}
}

// dist computes distance between vector and node ID.
func (h *HNSW) dist(v []float32, id model.RowID) float32 {
	if h.distOp != nil {
		return h.distOp(id, v)
	}
	d, ok := h.vectors.ComputeDistance(id, v, h.opts.DistanceType)
	if !ok {
		return math.MaxFloat32
	}
	return d
}

// Delete marks a node as deleted (logical delete).
// This is O(1) and avoids graph instability.
func (h *HNSW) Delete(ctx context.Context, id model.RowID) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	g := h.currentGraph.Load()
	g.arena.IncRef()
	defer g.arena.DecRef()

	// Use lock to protect tombstones
	g.shardedLocks[uint64(id)%uint64(len(g.shardedLocks))].Lock()
	defer g.shardedLocks[uint64(id)%uint64(len(g.shardedLocks))].Unlock()

	// Ensure bitset capacity
	g.tombstones.Grow(uint32(id) + 1)
	if g.tombstones.Test(uint32(id)) {
		return nil
	}
	g.tombstones.Set(uint32(id))

	// We do NOT remove from graph or release ID.
	// This preserves graph connectivity and avoids O(N) entry point scans.
	// The node remains in the graph but is ignored during searcher.

	// We can optionally delete the vector data to save memory,
	// but keeping it might be safer for concurrent readers.
	// For now, we keep it.

	g.countAtomic.Add(-1)
	return nil
}

// Update updates a vector.
func (h *HNSW) Update(ctx context.Context, id model.RowID, v []float32) error {
	g := h.currentGraph.Load()
	node := h.getNode(g, id)
	if node.IsZero() {
		return &ErrNodeNotFound{ID: id}
	}
	layer := node.Level(g.arena)

	if err := h.Delete(ctx, id); err != nil {
		return err
	}

	_, err := h.insert(ctx, g, v, id, layer, true)
	return err
}

// KNNSearch performs searcher.
func (h *HNSW) KNNSearch(ctx context.Context, q []float32, k int, opts *SearchOptions) ([]SearchResult, error) {
	res := make([]SearchResult, 0, k)
	if err := h.KNNSearchWithBuffer(ctx, q, k, opts, &res); err != nil {
		return nil, err
	}
	return res, nil
}

// KNNSearchWithBuffer performs a K-nearest neighbor search and appends results to the provided buffer.
// This avoids allocating a new slice for results, which is critical for high-throughput scenarios.
func (h *HNSW) KNNSearchWithBuffer(ctx context.Context, q []float32, k int, opts *SearchOptions, buf *[]SearchResult) error {
	s := searcher.Get()
	defer searcher.Put(s)

	// Prepare query (normalize if needed)
	q, err := h.prepareQuery(s, q)
	if err != nil {
		return err
	}

	var distFunc DistFunc
	if snapshottable, ok := h.vectors.(interface {
		Snapshot() vectorstore.VectorSnapshot
	}); ok {
		snap := snapshottable.Snapshot()
		distFunc = func(id model.RowID) float32 {
			d, _ := snap.ComputeDistance(id, q, h.opts.DistanceType)
			return d
		}
	} else if h.distOp != nil {
		distFunc = func(id model.RowID) float32 {
			return h.distOp(id, q)
		}
	} else {
		distFunc = func(id model.RowID) float32 {
			return h.dist(q, id)
		}
	}

	// Optimization: Hybrid Filtered Search
	// If the filter exposes a bitmap and selectivity is low, scan the bitmap directly.
	if opts != nil && opts.Filter != nil {
		if bm, ok := opts.Filter.(segment.Bitmap); ok {
			g := h.currentGraph.Load()
			if g != nil {
				total := g.countAtomic.Load()
				card := bm.Cardinality()

				// Heuristic: If match count is small (< 2% or < 2000), brute force the bitmap.
				// Traversing HNSW for very sparse targets is wasteful.
				const (
					selectivityThreshold = 0.02
					absoluteThreshold    = 2000
				)

				if total > 0 && (float64(card) < float64(total)*selectivityThreshold || card < absoluteThreshold) {
					if err := h.searchBitmap(ctx, s, k, bm, distFunc); err != nil {
						return err
					}
					goto extractResults
				}
			}
		}
	}

	if err := h.searchExecute(ctx, s, q, k, opts, distFunc); err != nil {
		return err
	}

extractResults:
	// Extract results from s.Candidates to buf
	results := s.Candidates
	// MaxHeap pops worst first, so we need to pop (Len-k) items first
	for results.Len() > k {
		_, _ = results.PopItem()
	}

	startIdx := len(*buf)
	for results.Len() > 0 {
		item, _ := results.PopItem()
		*buf = append(*buf, SearchResult{ID: uint32(item.Node), Distance: item.Distance})
	}

	// Reverse
	endIdx := len(*buf) - 1
	for i := 0; i < (endIdx-startIdx+1)/2; i++ {
		(*buf)[startIdx+i], (*buf)[endIdx-i] = (*buf)[endIdx-i], (*buf)[startIdx+i]
	}
	return nil
}

// KNNSearchWithContext performs a K-nearest neighbor search using the provided Searcher context.
func (h *HNSW) KNNSearchWithContext(ctx context.Context, s *searcher.Searcher, q []float32, k int, opts *SearchOptions) error {
	// Prepare query (normalize if needed)
	q, err := h.prepareQuery(s, q)
	if err != nil {
		return err
	}

	var distFunc DistFunc
	// Try to get a snapshot for lock-free distance calculation
	if snapshottable, ok := h.vectors.(interface {
		Snapshot() vectorstore.VectorSnapshot
	}); ok {
		snap := snapshottable.Snapshot()
		distFunc = func(id model.RowID) float32 {
			d, _ := snap.ComputeDistance(id, q, h.opts.DistanceType)
			return d
		}
	} else if h.distOp != nil {
		distFunc = func(id model.RowID) float32 {
			return h.distOp(id, q)
		}
	} else {
		distFunc = func(id model.RowID) float32 {
			return h.dist(q, id)
		}
	}

	return h.searchExecute(ctx, s, q, k, opts, distFunc)
}

func (h *HNSW) searchExecute(ctx context.Context, s *searcher.Searcher, q []float32, k int, opts *SearchOptions, distFunc DistFunc) error {
	if err := ctx.Err(); err != nil {
		return err
	}

	g := h.currentGraph.Load()
	if g == nil {
		return fmt.Errorf("hnsw graph is nil")
	}
	if g.arena == nil {
		return fmt.Errorf("hnsw graph arena is nil")
	}
	g.arena.IncRef()
	defer g.arena.DecRef()

	epID := g.entryPointAtomic.Load()
	if h.getNode(g, model.RowID(epID)).IsZero() {
		return nil
	}

	ef := h.determineEF(k, opts, g.countAtomic.Load())

	currID, currDist := h.greedySearch(g, q, model.RowID(epID), distFunc)

	// 2. Search layer 0 with pre-filtering
	var filter segment.Filter
	if opts != nil && opts.Filter != nil {
		filter = opts.Filter
	}

	h.searchLayer(s, g, q, currID, currDist, 0, ef, filter, distFunc)
	return nil
}

func (h *HNSW) prepareQuery(s *searcher.Searcher, q []float32) ([]float32, error) {
	// Safety check: Dimension mismatch
	// This is critical because internal distance functions (SIMD) do not check bounds.
	dim := int(h.dimensionAtomic.Load())
	if dim > 0 && len(q) != dim {
		return nil, &ErrDimensionMismatch{Expected: dim, Actual: len(q)}
	}

	// Normalize
	if h.opts.NormalizeVectors {
		if len(s.ScratchVec) < len(q) {
			s.ScratchVec = make([]float32, len(q))
		}
		copy(s.ScratchVec, q)
		if !distance.NormalizeL2InPlace(s.ScratchVec) {
			return nil, fmt.Errorf("hnsw: zero query vector")
		}
		return s.ScratchVec, nil
	}
	return q, nil
}

func (h *HNSW) determineEF(k int, opts *SearchOptions, totalCount int64) int {
	ef := h.opts.EF
	if opts != nil && opts.EFSearch > 0 {
		ef = opts.EFSearch
	}

	// Dynamic EF expansion for filtered search.
	// We expand ef inversely proportional to selectivity to ensure we find enough matching candidates.
	// ef' = ef / p, where p = card/total.
	// But we rely on brute-force fallback for p < 0.02, so p >= 0.02.
	// Max expansion factor is 50x.
	if opts != nil && opts.Filter != nil {
		if bm, ok := opts.Filter.(segment.Bitmap); ok && totalCount > 0 {
			card := bm.Cardinality()
			if card > 0 && card < uint64(totalCount) {
				// Use floating point math for precision
				selectivity := float64(card) / float64(totalCount)
				if selectivity > 0 {
					expanded := float64(ef) / selectivity

					// Clamp to avoid excessive expansion that could hurt latency/memory
					// Just in case selectivity is extremely low but somehow we didn't take the brute-force path.
					// (The brute-force path threshold is defined in KNNSearchWithBuffer, but KNNSearchStream relies on this too)
					const maxExpansion = 20000
					if expanded > maxExpansion {
						expanded = maxExpansion
					}
					ef = int(expanded)
				}
			}
		}
	}

	if ef < k {
		ef = k
	}
	return ef
}

func (h *HNSW) greedySearch(g *graph, q []float32, epID model.RowID, distFunc DistFunc) (model.RowID, float32) {
	// 1. Greedy to layer 0
	currID := epID
	currDist := distFunc(currID)
	maxLevel := int(g.maxLevelAtomic.Load())

	for level := maxLevel; level > 0; level-- {
		changed := true
		for changed {
			changed = false
			node := h.getNode(g, currID)
			if !node.IsZero() {
				raw := node.GetConnectionsRaw(g.arena, level, h.maxConnectionsPerLayer, h.maxConnectionsLayer0)
				for i := 0; i < len(raw); i++ {
					v := atomic.LoadUint64(&raw[i])
					next := NeighborFromUint64(v)
					nextDist := distFunc(next.ID)
					if nextDist < currDist {
						currID = next.ID
						currDist = nextDist
						changed = true
					}
				}
			}
		}
	}
	return currID, currDist
}

// KNNSearchStream implements streaming searcher.
func (h *HNSW) KNNSearchStream(ctx context.Context, q []float32, k int, opts *SearchOptions) iter.Seq2[SearchResult, error] {
	return func(yield func(SearchResult, error) bool) {
		if err := ctx.Err(); err != nil {
			yield(SearchResult{}, err)
			return
		}
		g := h.currentGraph.Load()
		g.arena.IncRef()
		defer g.arena.DecRef()

		// Acquire Searcher
		s := searcher.Get()
		defer searcher.Put(s)

		q, err := h.prepareQuery(s, q)
		if err != nil {
			yield(SearchResult{}, err)
			return
		}

		// Create distFunc
		var distFunc DistFunc
		if snapshottable, ok := h.vectors.(interface {
			Snapshot() vectorstore.VectorSnapshot
		}); ok {
			snap := snapshottable.Snapshot()
			distFunc = func(nid model.RowID) float32 {
				d, _ := snap.ComputeDistance(nid, q, h.opts.DistanceType)
				return d
			}
		} else if h.distOp != nil {
			distFunc = func(nid model.RowID) float32 {
				return h.distOp(nid, q)
			}
		} else {
			distFunc = func(nid model.RowID) float32 {
				return h.dist(q, nid)
			}
		}

		epID := g.entryPointAtomic.Load()
		if h.getNode(g, model.RowID(epID)).IsZero() {
			return
		}

		ef := h.determineEF(k, opts, g.countAtomic.Load())

		currID, currDist := h.greedySearch(g, q, model.RowID(epID), distFunc)

		// 2. Search layer 0 with pre-filtering
		var filter segment.Filter
		if opts != nil && opts.Filter != nil {
			filter = opts.Filter
		}

		h.searchLayer(s, g, q, currID, currDist, 0, ef, filter, distFunc)
		h.yieldResults(s.Candidates, k, yield)
	}
}

func (h *HNSW) yieldResults(results *searcher.PriorityQueue, k int, yield func(SearchResult, error) bool) {
	// Extract K
	// MaxHeap pops worst first, so we need to pop (Len-k) items first
	for results.Len() > k {
		_, _ = results.PopItem()
	}

	// Collect results in reverse order (nearest first)
	tempResults := make([]SearchResult, 0, k)
	for results.Len() > 0 {
		item, _ := results.PopItem()
		tempResults = append(tempResults, SearchResult{ID: uint32(item.Node), Distance: item.Distance})
	}

	// Yield in reverse (since we popped worst first, the last item in tempResults is the best)
	for i := len(tempResults) - 1; i >= 0; i-- {
		if !yield(tempResults[i], nil) {
			return
		}
	}
}

// BruteSearch implements brute force searcher.
func (h *HNSW) BruteSearch(ctx context.Context, query []float32, k int, filter func(id model.RowID) bool) ([]SearchResult, error) {
	g := h.currentGraph.Load()
	// Simple scan
	pq := searcher.NewPriorityQueue(true)

	// Acquire Searcher just for scratch vector (for normalization)
	s := searcher.Get()
	defer searcher.Put(s)

	q, err := h.prepareQuery(s, query)
	if err != nil {
		return nil, err
	}

	// Create distFunc
	var distFunc DistFunc
	if snapshottable, ok := h.vectors.(interface {
		Snapshot() vectorstore.VectorSnapshot
	}); ok {
		snap := snapshottable.Snapshot()
		distFunc = func(nid model.RowID) float32 {
			d, _ := snap.ComputeDistance(nid, q, h.opts.DistanceType)
			return d
		}
	} else if h.distOp != nil {
		distFunc = func(nid model.RowID) float32 {
			return h.distOp(nid, q)
		}
	} else {
		distFunc = func(nid model.RowID) float32 {
			return h.dist(q, nid)
		}
	}

	nodes := g.nodes.Load()
	if nodes != nil {
		for i, seg := range *nodes {
			if seg == nil {
				continue
			}
			if err := h.scanSegment(seg, i, query, k, filter, pq, distFunc); err != nil {
				return nil, err
			}
		}
	}

	res := make([]SearchResult, pq.Len())
	for i := pq.Len() - 1; i >= 0; i-- {
		item, _ := pq.PopItem()
		res[i] = SearchResult{ID: uint32(item.Node), Distance: item.Distance}
	}
	return res, nil
}

func (h *HNSW) scanSegment(seg *NodeSegment, segIdx int, query []float32, k int, filter func(id model.RowID) bool, pq *searcher.PriorityQueue, distFunc DistFunc) error {
	for j := range seg {
		val := seg[j].Load()
		node := Node{ref: nodeRef(val)}
		if node.IsZero() {
			continue
		}
		iU64, err := conv.IntToUint64(segIdx)
		if err != nil {
			return err
		}
		jU64, err := conv.IntToUint64(j)
		if err != nil {
			return err
		}
		nodeID := iU64*nodeSegmentSize + jU64
		nodeIDU32, err := conv.Uint64ToUint32(nodeID)
		if err != nil {
			return err
		}
		id := model.RowID(nodeIDU32)
		if filter != nil && !filter(id) {
			continue
		}

		d := distFunc(id)
		if pq.Len() < k {
			pq.PushItem(searcher.PriorityQueueItem{Node: id, Distance: d})
		} else {
			top, _ := pq.TopItem()
			if d < top.Distance {
				_, _ = pq.PopItem()
				pq.PushItem(searcher.PriorityQueueItem{Node: id, Distance: d})
			}
		}
	}
	return nil
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

// VectorCount returns the number of vectors in the index.
func (h *HNSW) VectorCount() int {
	c := h.currentGraph.Load().countAtomic.Load()
	if c > int64(math.MaxInt) {
		return math.MaxInt
	}
	return int(c)
}

// Size returns the estimated memory usage in bytes.
func (h *HNSW) Size() int64 {
	return h.currentGraph.Load().Size()
}

// ContainsID checks if an ID exists in the index.
func (h *HNSW) ContainsID(id uint64) bool {
	g := h.currentGraph.Load()
	idU32, err := conv.Uint64ToUint32(id)
	if err != nil {
		return false
	}
	if g.tombstones.Test(idU32) {
		return false
	}
	return !h.getNode(g, model.RowID(idU32)).IsZero()
}

// ShardID returns the shard ID.
func (h *HNSW) ShardID() int { return 0 }

// NumShards returns the number of shards.
func (h *HNSW) NumShards() int { return 1 }

// NewSharded creates a new sharded HNSW index (stub).
func NewSharded(shardID, numShards int, optFns ...func(o *Options)) (*HNSW, error) {
	return New(optFns...)
}

// VectorByID returns vector.
func (h *HNSW) VectorByID(ctx context.Context, id model.RowID) ([]float32, error) {
	vec, ok := h.vectors.GetVector(id)
	if !ok {
		return nil, &ErrNodeNotFound{ID: id}
	}
	return vec, nil
}

func (h *HNSW) recoverEntryPoint(g *graph) {
	// Double check
	epID := g.entryPointAtomic.Load()
	if !h.getNode(g, model.RowID(epID)).IsZero() {
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
			val := segment[j].Load()
			node := Node{ref: nodeRef(val)}
			if g.checkAndSetEntryPoint(node, i, j) {
				return
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

// Metric returns the equivalent public distance metric for this index.
func (h *HNSW) Metric() distance.Metric {
	return h.opts.DistanceType
}

func (g *graph) checkAndSetEntryPoint(node Node, i, j int) bool {
	if node.IsZero() {
		return false
	}
	iU64, err := conv.IntToUint64(i)
	if err != nil {
		return false
	}
	jU64, err := conv.IntToUint64(j)
	if err != nil {
		return false
	}
	id := iU64*nodeSegmentSize + jU64
	idU32, err := conv.Uint64ToUint32(id)
	if err != nil {
		return false
	}

	if !g.tombstones.Test(idU32) {
		g.entryPointAtomic.Store(idU32)
		g.maxLevelAtomic.Store(int32(node.Level(g.arena)))
		return true
	}
	return false
}

func newDistanceFunc(metric distance.Metric) (distance.Func, error) {
	switch metric {
	case distance.MetricL2:
		return distance.SquaredL2, nil
	case distance.MetricCosine:
		// For normalized vectors, Cosine Distance = 1 - Dot Product.
		// We use 0.5 * SquaredL2 which is monotonic with Cosine Distance for unit vectors.
		return func(a, b []float32) float32 {
			return 0.5 * distance.SquaredL2(a, b)
		}, nil
	case distance.MetricDot:
		// Convert dot product similarity (higher is better) into a distance (lower is better).
		return func(a, b []float32) float32 {
			return -distance.Dot(a, b)
		}, nil
	default:
		return nil, fmt.Errorf("unsupported metric: %v", metric)
	}
}

// searchBitmap performs a brute-force search over the IDs in the bitmap.
// This is used for highly selective filters where graph traversal is inefficient.
func (h *HNSW) searchBitmap(ctx context.Context, s *searcher.Searcher, k int, bm segment.Bitmap, distFunc DistFunc) error {
	s.Candidates.Reset()

	g := h.currentGraph.Load()
	if g == nil {
		return fmt.Errorf("hnsw graph is nil")
	}

	bm.ForEach(func(idU32 uint32) bool {
		if ctx.Err() != nil {
			return false
		}

		// Check tombstones (must check if deleted)
		if g.tombstones.Test(idU32) {
			return true
		}

		id := model.RowID(idU32)
		dist := distFunc(id)

		// Maintain Heap of size K
		if s.Candidates.Len() < k {
			s.Candidates.PushItem(searcher.PriorityQueueItem{Node: id, Distance: dist})
		} else {
			top, _ := s.Candidates.TopItem()
			if dist < top.Distance {
				s.Candidates.PopItem()
				s.Candidates.PushItem(searcher.PriorityQueueItem{Node: id, Distance: dist})
			}
		}
		return true
	})

	return ctx.Err()
}
