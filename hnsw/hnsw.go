package hnsw

import (
	"container/heap"
	"fmt"
	"math"
	"math/rand"
	"sync"

	"log"

	"github.com/bits-and-blooms/bitset"
	"github.com/hupe1980/vecgo/metric"
)

// ErrDimensionMismatch is a named error type for dimension mismatch
type ErrDimensionMismatch struct {
	Expected int // Expected dimensions
	Actual   int // Actual dimensions
}

// Error returns the error message for dimension mismatch
func (e *ErrDimensionMismatch) Error() string {
	return fmt.Sprintf("dimension mismatch: expected %d, got %d", e.Expected, e.Actual)
}

// DistanceFunc represents a function for calculating the distance between two vectors
type DistanceFunc func(v1, v2 []float32) (float32, error)

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

	// DistanceFunc represents the distance function for calculating distance between vectors.
	// This function is essential for performing distance calculations during search operations.
	DistanceFunc DistanceFunc
}

var DefaultOptions = Options{
	M:            8,
	EF:           200,
	Heuristic:    true,
	DistanceFunc: metric.SquaredL2,
}

// HNSW represents the Hierarchical Navigable Small World graph
type HNSW struct {
	dimension int
	mmax      int     // Max number of connections per element/per layer
	mmax0     int     // Max for the 0 layer
	ml        float64 // Normalization factor for level generation
	ep        uint32  // Top layer of HNSW
	maxLevel  int     // Track the current max level used

	nodes []*Node

	opts Options

	mutex sync.Mutex
}

// New creates a new HNSW instance with the given dimension and options
func New(dimension int, optFns ...func(o *Options)) *HNSW {
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
		dimension: dimension,
		mmax:      opts.M,
		mmax0:     2 * opts.M,
		ep:        0,
		maxLevel:  0,
		ml:        1 / math.Log(1.0*float64(opts.M)),
		nodes:     []*Node{{ID: 0, Layer: 0, Vector: make([]float32, dimension), Connections: make([][]uint32, 2*opts.M+1)}},
		opts:      opts,
	}
}

// Insert inserts a new element into the HNSW graph
func (h *HNSW) Insert(v []float32) (uint32, error) {
	// Check if dimensions of the input vector match the expected dimension
	if len(v) != h.dimension {
		return 0, &ErrDimensionMismatch{Expected: h.dimension, Actual: len(v)}
	}

	// Make a copy of the vector to ensure changes outside this function don't affect the node
	vectorCopy := make([]float32, len(v))
	copy(vectorCopy, v)

	h.mutex.Lock()
	defer h.mutex.Unlock()

	// next ID
	id := uint32(len(h.nodes))

	node := &Node{
		ID:          id,
		Vector:      vectorCopy,
		Layer:       int(math.Floor(-math.Log(rand.Float64()) * h.ml)), // nolint gosec
		Connections: make([][]uint32, h.mmax+1),
	}

	// Find single shortest path from top layers above our current node, which will be our new starting-point
	currObj, currDist, err := h.findShortestPath(node)
	if err != nil {
		return 0, err
	}

	topCandidates := &PriorityQueue{
		Order: false,
	}

	// For all levels equal and below our current node, find the top (closest) candidates and create a link
	for level := min(node.Layer, h.maxLevel); level >= 0; level-- {
		err = h.searchLayer(vectorCopy, &PriorityQueueItem{Distance: currDist, Node: currObj.ID}, topCandidates, h.opts.EF, level)
		if err != nil {
			return 0, err
		}

		// Switch type, naive k-NN, or Heuristic HNSW for linking nearest neighbours
		if h.opts.Heuristic {
			h.selectNeighboursHeuristic(topCandidates, h.opts.M, false)
		} else {
			h.selectNeighboursSimple(topCandidates, h.opts.M)
		}

		node.Connections[level] = make([]uint32, topCandidates.Len())

		for i := topCandidates.Len() - 1; i >= 0; i-- {
			candidate, _ := heap.Pop(topCandidates).(*PriorityQueueItem)
			node.Connections[level][i] = candidate.Node
		}
	}

	// Append new node
	h.nodes = append(h.nodes, node)

	// Next link the neighbour nodes to our new node, making it visible
	for level := min(node.Layer, h.maxLevel); level >= 0; level-- {
		for _, neighbourNode := range node.Connections[level] {
			h.Link(neighbourNode, node.ID, level)
		}
	}

	if node.Layer > h.maxLevel {
		h.ep = node.ID
		h.maxLevel = node.Layer
	}

	return node.ID, nil
}

func (h *HNSW) findShortestPath(node *Node) (*Node, float32, error) {
	// Current distance from our starting-point (ep)
	currObj := h.nodes[h.ep]

	currDist, err := h.opts.DistanceFunc(currObj.Vector, node.Vector)
	if err != nil {
		return nil, 0, err
	}

	for level := currObj.Layer; level > node.Layer; level-- {
		changed := true
		for changed {
			changed = false

			for _, nodeID := range currObj.Connections[level] {
				newObj := h.nodes[nodeID]

				newDist, err := h.opts.DistanceFunc(newObj.Vector, node.Vector)
				if err != nil {
					return nil, 0, err
				}

				if newDist < currDist {
					// Update the starting point to our new node
					currObj = newObj
					// Update the currently shortest distance
					currDist = newDist
					changed = true
				}
			}
		}
	}

	return currObj, currDist, nil
}

// KNNSearch performs a k-nearest neighbor search in the HNSW graph
func (h *HNSW) KNNSearch(q []float32, k int, efSearch int) (*PriorityQueue, error) {
	topCandidates := &PriorityQueue{
		Order: true,
	}

	heap.Init(topCandidates)

	currObj := h.nodes[h.ep]

	match, currDist, err := h.findEp(q, currObj)
	if err != nil {
		return nil, err
	}

	var node uint32
	if match != nil {
		node = match.ID
	}

	if err := h.searchLayer(q, &PriorityQueueItem{Distance: currDist, Node: node}, topCandidates, efSearch, 0); err != nil {
		return nil, err
	}

	for topCandidates.Len() > k {
		_ = heap.Pop(topCandidates)
	}

	return topCandidates, nil
}

// BruteSearch performs a brute-force search in the HNSW graph
func (h *HNSW) BruteSearch(query []float32, k int) (*PriorityQueue, error) {
	topCandidates := &PriorityQueue{
		Order: true,
	}

	heap.Init(topCandidates)

	for _, node := range h.nodes {
		nodeDist, err := h.opts.DistanceFunc(query, node.Vector)
		if err != nil {
			return nil, err
		}

		if topCandidates.Len() < k {
			heap.Push(topCandidates, &PriorityQueueItem{
				Node:     node.ID,
				Distance: nodeDist,
			})

			continue
		}

		largestDist, _ := topCandidates.Top().(*PriorityQueueItem)

		if nodeDist < largestDist.Distance {
			heap.Pop(topCandidates)

			heap.Push(topCandidates, &PriorityQueueItem{
				Node:     node.ID,
				Distance: nodeDist,
			})
		}
	}

	return topCandidates, nil
}

// Link adds links between nodes in the HNSW graph
func (h *HNSW) Link(first uint32, second uint32, level int) {
	maxConnections := h.mmax
	// HNSW allows double the connections for the bottom level (0)
	if level == 0 {
		maxConnections = h.mmax0
	}

	node := h.nodes[first]
	node.Connections[level] = append(node.Connections[level], second)

	if len(node.Connections[level]) > maxConnections {
		// Add the new candidate to our queue
		topCandidates := &PriorityQueue{
			Order: false,
		}

		heap.Init(topCandidates)

		for _, id := range node.Connections[level] {
			distance, err := h.opts.DistanceFunc(node.Vector, h.nodes[id].Vector)
			if err != nil {
				log.Fatal(err)
			}

			heap.Push(topCandidates, &PriorityQueueItem{Node: id, Distance: distance})
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
			item, _ := heap.Pop(topCandidates).(*PriorityQueueItem)
			node.Connections[level][i] = item.Node
		}
	}
}

// searchLayer performs a search in a specified layer of the HNSW graph
func (h *HNSW) searchLayer(q []float32, ep *PriorityQueueItem, topCandidates *PriorityQueue, ef int, level int) error {
	var visited bitset.BitSet

	visited.Set(uint(ep.Node))

	// Add the new candidate to our queue
	candidates := &PriorityQueue{
		Order: false,
	}
	heap.Init(candidates)
	heap.Push(candidates, ep)

	topCandidates.Order = true // max-heap
	heap.Init(topCandidates)
	heap.Push(topCandidates, ep)

	for candidates.Len() > 0 {
		lowerBound := topCandidates.Top().(*PriorityQueueItem).Distance

		candidate, _ := heap.Pop(candidates).(*PriorityQueueItem)
		if candidate.Distance > lowerBound {
			break
		}

		node := h.nodes[candidate.Node]

		if len(node.Connections) > level { // Check if level is within bounds
			conns := node.Connections[level]

			for _, n := range conns {
				if !visited.Test(uint(n)) {
					visited.Set(uint(n))

					distance, err := h.opts.DistanceFunc(q, h.nodes[n].Vector)
					if err != nil {
						return err
					}

					topDistance := topCandidates.Top().(*PriorityQueueItem).Distance

					item := &PriorityQueueItem{
						Distance: distance,
						Node:     n,
					}

					// Add the element to topCandidates if size < EF
					if topCandidates.Len() < ef {
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

	return nil
}

// selectNeighboursSimple selects the nearest neighbors using a simple approach
func (h *HNSW) selectNeighboursSimple(topCandidates *PriorityQueue, M int) {
	for topCandidates.Len() > M {
		_ = heap.Pop(topCandidates)
	}
}

// selectNeighboursHeuristic selects the nearest neighbors using a heuristic approach
func (h *HNSW) selectNeighboursHeuristic(topCandidates *PriorityQueue, M int, order bool) {
	// If results < M, return, nothing required
	if topCandidates.Len() < M {
		return
	}

	// Create our new priority queues
	newCandidates := &PriorityQueue{}

	tmpCandidates := &PriorityQueue{Order: order}
	heap.Init(tmpCandidates)

	items := make([]*PriorityQueueItem, 0, M)

	if !order {
		newCandidates.Order = order
		heap.Init(newCandidates)

		// Add existing candidates to our new queue
		for topCandidates.Len() > 0 {
			item, _ := heap.Pop(topCandidates).(*PriorityQueueItem)
			heap.Push(newCandidates, item)
		}
	} else {
		newCandidates = topCandidates
	}

	// Scan through our new queue (order changed from min-heap > max-heap or vice-versa depending on order arg)
	for newCandidates.Len() > 0 {
		// Finish if items reaches our desired length
		if len(items) >= M {
			break
		}

		item, _ := heap.Pop(newCandidates).(*PriorityQueueItem)
		hit := true

		// Search through each item and determine if distance from node lower for items in set
		for _, v := range items {
			distance, _ := h.opts.DistanceFunc(h.nodes[v.Node].Vector, h.nodes[item.Node].Vector)
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
	for len(items) < M && tmpCandidates.Len() > 0 {
		item, _ := heap.Pop(tmpCandidates).(*PriorityQueueItem)
		items = append(items, item)
	}

	// Last step, append our results into our original min/max-heap
	for _, item := range items {
		heap.Push(topCandidates, item)
	}
}

// findEp finds the entry-point (ep) in the HNSW graph
func (h *HNSW) findEp(q []float32, currObj *Node) (*Node, float32, error) {
	currDist, err := h.opts.DistanceFunc(q, currObj.Vector)
	if err != nil {
		return nil, 0, err
	}

	var match *Node

	// Find single shortest path from top layers above our current node, which will be our new starting-point
	for level := h.maxLevel; level > 0; level-- {
		scan := true

		for scan {
			scan = false

			for _, nodeID := range currObj.Connections[level] {
				nodeDist, err := h.opts.DistanceFunc(h.nodes[nodeID].Vector, q)
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
