// Package hnsw implements the Hierarchical Navigable Small World (HNSW) graph for approximate nearest neighbor search.
package hnsw

import (
	"container/heap"
	"math"
	"math/rand"
	"sync"
	"sync/atomic"

	"github.com/bits-and-blooms/bitset"
	"github.com/hupe1980/vecgo/index"
	"github.com/hupe1980/vecgo/queue"
)

// Node represents a node in the HNSW graph
type Node struct {
	Connections [][]uint32 // Links to other nodes
	Vector      []float32  // Vector (X dimensions)
	Layer       int        // Layer the node exists in the HNSW tree
	ID          uint32     // Unique identifier
}

// Options represents the options for configuring HNSW.
type Options struct {
	// M specifies the number of established connections for every new element during construction.
	// Reasonable range for M is 2-100. Higher M works better on datasets with high intrinsic dimensionality and/or high recall,
	// while low M works better for datasets with low intrinsic dimensionality and/or low recalls.
	// As an example, for dimension=4 random vectors, optimal M for search is somewhere around 6, while for high-dimensional datasets
	// (word embeddings, good face descriptors), higher M values are required (e.g., M=48-64) for optimal performance at high recall.
	// The range M=12-48 is ok for most use cases. When M is changed, one has to update the other parameters.
	// Nonetheless, ef and ef_construction parameters can be roughly estimated by assuming that M * ef_construction is a constant.
	M int

	// EF specifies the size of the dynamic candidate list.
	// EF is important for search time. Larger EF values can improve recall at the cost of increased search time.
	EF int

	// Heuristic indicates whether to use the heuristic algorithm (true) or the naive K-NN algorithm (false).
	// The heuristic algorithm is generally faster but may sacrifice some accuracy.
	Heuristic bool

	// DistanceType represents the type of distance function used for calculating distances between vectors.
	DistanceType index.DistanceType
}

var DefaultOptions = Options{
	M:            8,
	EF:           200,
	Heuristic:    true,
	DistanceType: index.DistanceTypeSquaredL2,
}

// HNSW represents the Hierarchical Navigable Small World graph
type HNSW struct {
	sync.RWMutex

	dimension int32
	mmax      int     // Max number of connections per element/per layer
	mmax0     int     // Max for the 0 layer
	ml        float64 // Normalization factor for level generation
	ep        uint32  // Top layer of HNSW
	maxLevel  int     // Track the current max level used

	nodes []*Node

	distanceFunc index.DistanceFunc

	opts Options

	initOnce   *sync.Once
	insertOnce *sync.Once
}

// New creates a new HNSW instance with the given options
func New(optFns ...func(o *Options)) *HNSW {
	opts := DefaultOptions

	for _, fn := range optFns {
		fn(&opts)
	}

	if opts.M == 1 {
		// M == 1 would result in division by zero
		// 1 / log(1.0 * M) = 1 / 0
		opts.M = 2 // TODO Add warning?
	}

	return &HNSW{
		mmax:         opts.M,
		mmax0:        2 * opts.M,
		ml:           1 / math.Log(1.0*float64(opts.M)),
		distanceFunc: index.NewDistanceFunc(opts.DistanceType),
		opts:         opts,

		initOnce:   &sync.Once{},
		insertOnce: &sync.Once{},
	}
}

// Insert inserts a new element into the HNSW graph
func (h *HNSW) Insert(v []float32) (uint32, error) {
	h.initOnce.Do(func() {
		if h.isEmpty() {
			atomic.StoreInt32(&h.dimension, int32(len(v)))
		}
	})

	// Check if dimensions of the input vector match the expected dimension
	dim := int(atomic.LoadInt32(&h.dimension))
	if len(v) != dim {
		return 0, &index.ErrDimensionMismatch{Expected: dim, Actual: len(v)}
	}

	// Make a copy of the vector to ensure changes outside this function don't affect the node
	vectorCopy := make([]float32, len(v))
	copy(vectorCopy, v)

	wasFirst := false

	h.insertOnce.Do(func() {
		if h.isEmpty() {
			h.Lock()
			defer h.Unlock()

			wasFirst = true

			h.ep = 0
			h.maxLevel = 0
			h.nodes = []*Node{{ID: 0, Layer: 0, Vector: vectorCopy, Connections: [][]uint32{
				make([]uint32, 0, h.mmax0),
			}}}
		}
	})

	if wasFirst {
		return 0, nil
	}

	h.Lock()
	defer h.Unlock()

	// next ID
	id := uint32(len(h.nodes))

	layer := int(math.Floor(-math.Log(rand.Float64()) * h.ml)) // nolint gosec

	node := &Node{
		ID:     id,
		Vector: vectorCopy,
		Layer:  layer,
	}

	node.Connections = make([][]uint32, layer+1)

	for i := layer; i >= 0; i-- {
		capacity := h.mmax
		if i == 0 {
			capacity = h.mmax0
		}

		node.Connections[i] = make([]uint32, 0, capacity)
	}

	// Find single shortest path from top layers above our current node, which will be our new starting-point
	currObj, currDist, err := h.findShortestPath(node)
	if err != nil {
		return 0, err
	}

	// For all levels equal and below our current node, find the top (closest) candidates and create a link
	for level := min(node.Layer, h.maxLevel); level >= 0; level-- {
		topCandidates, err := h.searchLayer(&searchParams{
			Query:      node.Vector,
			EntryPoint: &queue.PriorityQueueItem{Distance: currDist, Node: currObj.ID},
			EF:         h.opts.EF,
			Level:      level,
		})
		if err != nil {
			return 0, err
		}

		// Switch type, naive k-NN, or Heuristic HNSW for linking nearest neighbours
		if h.opts.Heuristic {
			h.selectNeighboursHeuristic(topCandidates, h.mmax, false)
		} else {
			h.selectNeighboursSimple(topCandidates, h.mmax)
		}

		node.Connections[level] = make([]uint32, topCandidates.Len())

		for i := topCandidates.Len() - 1; i >= 0; i-- {
			candidate, _ := heap.Pop(topCandidates).(*queue.PriorityQueueItem)
			node.Connections[level][i] = candidate.Node
		}
	}

	// Append new node
	h.nodes = append(h.nodes, node)

	// Next link the neighbour nodes to our new node, making it visible
	for level := min(node.Layer, h.maxLevel); level >= 0; level-- {
		for _, neighbourNode := range node.Connections[level] {
			if err := h.Link(neighbourNode, node.ID, level); err != nil {
				return 0, err
			}
		}
	}

	if node.Layer > h.maxLevel {
		h.ep = node.ID
		h.maxLevel = node.Layer
	}

	return node.ID, nil
}

// KNNSearch performs a k-nearest neighbor search in the HNSW graph
func (h *HNSW) KNNSearch(q []float32, k int, efSearch int, filter func(id uint32) bool) (*queue.PriorityQueue, error) {
	ep, currDist, err := h.findEP(q, h.nodes[h.ep])
	if err != nil {
		return nil, err
	}

	topCandidates, err := h.searchLayer(&searchParams{
		Query:      q,
		EntryPoint: &queue.PriorityQueueItem{Distance: currDist, Node: ep.ID},
		EF:         efSearch,
		Level:      0,
		Filter:     filter,
	})
	if err != nil {
		return nil, err
	}

	for topCandidates.Len() > k {
		_ = heap.Pop(topCandidates)
	}

	return topCandidates, nil
}

// BruteSearch performs a brute-force search in the HNSW graph
func (h *HNSW) BruteSearch(query []float32, k int, filter func(id uint32) bool) (*queue.PriorityQueue, error) {
	topCandidates := &queue.PriorityQueue{
		Order: true,
	}

	heap.Init(topCandidates)

	for _, node := range h.nodes {
		if !filter(node.ID) {
			continue
		}

		nodeDist, err := h.distanceFunc(query, node.Vector)
		if err != nil {
			return nil, err
		}

		if topCandidates.Len() < k {
			heap.Push(topCandidates, &queue.PriorityQueueItem{
				Node:     node.ID,
				Distance: nodeDist,
			})

			continue
		}

		largestDist, _ := topCandidates.Top().(*queue.PriorityQueueItem)

		if nodeDist < largestDist.Distance {
			heap.Pop(topCandidates)

			heap.Push(topCandidates, &queue.PriorityQueueItem{
				Node:     node.ID,
				Distance: nodeDist,
			})
		}
	}

	return topCandidates, nil
}

// Link adds links between nodes in the HNSW graph
func (h *HNSW) Link(first uint32, second uint32, level int) error {
	maxConnections := h.mmax
	// HNSW allows double the connections for the bottom level (0)
	if level == 0 {
		maxConnections = h.mmax0
	}

	node := h.nodes[first]

	node.Connections[level] = append(node.Connections[level], second)

	if len(node.Connections[level]) > maxConnections {
		// Add the new candidate to our queue
		topCandidates := &queue.PriorityQueue{
			Order: false,
		}

		heap.Init(topCandidates)

		for _, id := range node.Connections[level] {
			distance, err := h.distanceFunc(node.Vector, h.nodes[id].Vector)
			if err != nil {
				return err
			}

			heap.Push(topCandidates, &queue.PriorityQueueItem{Node: id, Distance: distance})
		}

		if h.opts.Heuristic {
			h.selectNeighboursHeuristic(topCandidates, maxConnections, true)
		} else {
			h.selectNeighboursSimple(topCandidates, maxConnections)
		}

		// Next, reorder our connected nodes with the improved lower distances within the graph
		node.Connections[level] = make([]uint32, maxConnections)

		// Order by best performing match (index 0) .. lowest
		for i := maxConnections - 1; i >= 0; i-- {
			item, _ := heap.Pop(topCandidates).(*queue.PriorityQueueItem)
			node.Connections[level][i] = item.Node
		}
	}

	return nil
}

func (h *HNSW) isEmpty() bool {
	h.RLock()
	defer h.RUnlock()

	return len(h.nodes) == 0
}

// findShortestPath finds the shortest path from the top layers to the specified node in the HNSW graph
func (h *HNSW) findShortestPath(node *Node) (*Node, float32, error) {
	// Current distance from our starting-point (ep)
	currObj := h.nodes[h.ep]

	currDist, err := h.distanceFunc(currObj.Vector, node.Vector)
	if err != nil {
		return nil, 0, err
	}

	for level := currObj.Layer; level > node.Layer; level-- {
		changed := true
		for changed {
			changed = false

			for _, nodeID := range currObj.Connections[level] {
				newObj := h.nodes[nodeID]

				newDist, err := h.distanceFunc(newObj.Vector, node.Vector)
				if err != nil {
					return nil, 0, err
				}

				if newDist < currDist {
					// Update the starting point to our new node
					currObj = newObj
					// Update the currently shortest distance
					currDist = newDist
					// If a smaller match found, continue
					changed = true
				}
			}
		}
	}

	return currObj, currDist, nil
}

// searchParams represents the parameters for performing a search operation in the HNSW graph.
type searchParams struct {
	// Query is the vector for which nearest neighbors are being searched.
	Query []float32

	// EntryPoint represents the starting point for the search operation.
	EntryPoint *queue.PriorityQueueItem

	// EF (Exploration Factor) specifies the size of the dynamic candidate list during the search.
	EF int

	// Level represents the level of the HNSW graph at which the search is performed.
	Level int

	// Filter is an optional filter function that can be applied to the search results.
	// If provided, only the nodes passing the filter condition will be considered in the search.
	Filter func(id uint32) bool
}

// searchLayer performs a search in a specified layer of the HNSW graph
func (h *HNSW) searchLayer(params *searchParams) (*queue.PriorityQueue, error) {
	var visited bitset.BitSet

	visited.Set(uint(params.EntryPoint.Node))

	// Add the new candidate to our queue
	candidates := &queue.PriorityQueue{
		Order: false, // min-heap
	}

	heap.Init(candidates)
	heap.Push(candidates, params.EntryPoint)

	topCandidates := &queue.PriorityQueue{
		Order: true, // max-heap
	}

	heap.Init(topCandidates)
	heap.Push(topCandidates, params.EntryPoint)

	for candidates.Len() > 0 {
		lowerBound := topCandidates.Top().(*queue.PriorityQueueItem).Distance

		candidate, _ := heap.Pop(candidates).(*queue.PriorityQueueItem)
		if candidate.Distance > lowerBound {
			break
		}

		node := h.nodes[candidate.Node]

		if len(node.Connections) > params.Level { // Check if level is within bounds
			conns := node.Connections[params.Level]

			for _, n := range conns {
				if !visited.Test(uint(n)) {
					visited.Set(uint(n))

					distance, err := h.distanceFunc(params.Query, h.nodes[n].Vector)
					if err != nil {
						return nil, err
					}

					item := &queue.PriorityQueueItem{
						Distance: distance,
						Node:     n,
					}

					if params.Filter != nil && !params.Filter(item.Node) {
						heap.Push(candidates, item)
						continue
					}

					topDistance := topCandidates.Top().(*queue.PriorityQueueItem).Distance

					// Add the element to topCandidates if size < EF
					if topCandidates.Len() < params.EF {
						heap.Push(topCandidates, item)
						heap.Push(candidates, item)
					} else if topDistance > distance {
						heap.Pop(topCandidates)
						heap.Push(topCandidates, item)
						heap.Push(candidates, item)
					}
				}
			}
		}
	}

	return topCandidates, nil
}

// selectNeighboursSimple selects the nearest neighbors using a simple approach
func (h *HNSW) selectNeighboursSimple(topCandidates *queue.PriorityQueue, m int) {
	for topCandidates.Len() > m {
		_ = heap.Pop(topCandidates)
	}
}

// selectNeighboursHeuristic selects the nearest neighbors using a heuristic approach
func (h *HNSW) selectNeighboursHeuristic(topCandidates *queue.PriorityQueue, m int, order bool) {
	// If results < m, return, nothing required
	if topCandidates.Len() < m {
		return
	}

	// Create our new priority queues
	newCandidates := &queue.PriorityQueue{}

	tmpCandidates := &queue.PriorityQueue{Order: order}

	heap.Init(tmpCandidates)

	items := make([]*queue.PriorityQueueItem, 0, m)

	if !order {
		newCandidates.Order = order
		heap.Init(newCandidates)

		// Add existing candidates to our new queue
		for topCandidates.Len() > 0 {
			item, _ := heap.Pop(topCandidates).(*queue.PriorityQueueItem)
			heap.Push(newCandidates, item)
		}
	} else {
		newCandidates = topCandidates
	}

	// Scan through our new queue (order changed from min-heap > max-heap or vice-versa depending on order arg)
	for newCandidates.Len() > 0 {
		// Finish if items reaches our desired length
		if len(items) >= m {
			break
		}

		item, _ := heap.Pop(newCandidates).(*queue.PriorityQueueItem)

		hit := true

		// Search through each item and determine if distance from node lower for items in set
		for _, v := range items {
			distance, _ := h.distanceFunc(h.nodes[v.Node].Vector, h.nodes[item.Node].Vector)
			if distance < item.Distance {
				hit = false
				break
			}
		}

		if hit {
			items = append(items, item)
		} else {
			heap.Push(tmpCandidates, item)
		}
	}

	// Add any additional items from tmpCandidates if current items < M
	for len(items) < m && tmpCandidates.Len() > 0 {
		item, _ := heap.Pop(tmpCandidates).(*queue.PriorityQueueItem)
		items = append(items, item)
	}

	// Last step, append our results into our original min/max-heap
	for _, item := range items {
		heap.Push(topCandidates, item)
	}
}

// findEP finds the entry-point (ep) in the HNSW graph
func (h *HNSW) findEP(q []float32, currObj *Node) (*Node, float32, error) {
	currDist, err := h.distanceFunc(q, currObj.Vector)
	if err != nil {
		return nil, 0, err
	}

	match := currObj

	// Find single shortest path from top layers above our current node, which will be our new starting-point
	for level := h.maxLevel; level > 0; level-- {
		scan := true

		for scan {
			scan = false

			for _, nodeID := range currObj.Connections[level] {
				nodeDist, err := h.distanceFunc(h.nodes[nodeID].Vector, q)
				if err != nil {
					return nil, 0, err
				}

				if nodeDist < currDist {
					// Update the starting point to our new node
					match = h.nodes[nodeID]

					// Update the currently shortest distance
					currDist = nodeDist

					// If a smaller match found, continue
					scan = true
				}
			}
		}
	}

	return match, currDist, nil
}
