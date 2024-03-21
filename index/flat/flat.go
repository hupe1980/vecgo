// Package flat provides an implementation of a flat index for vector storage and search.
package flat

import (
	"container/heap"
	"sync"

	"github.com/hupe1980/vecgo/index"
	"github.com/hupe1980/vecgo/queue"
)

// Compile time check to ensure Flat satisfies the Index interface.
var _ index.Index = (*Flat)(nil)

// Node represents a node in the flat index with its vector and unique identifier.
type Node struct {
	Vector []float32 // Vector (X dimensions)
	ID     uint32    // Unique identifier
}

// Options contains configuration options for the flat index.
type Options struct {
	// DistanceType represents the type of distance function used for calculating distances between vectors.
	DistanceType index.DistanceType
}

// DefaultOptions contains the default configuration options for the flat index.
var DefaultOptions = Options{
	DistanceType: index.DistanceTypeSquaredL2,
}

// Flat represents a flat index for vector storage and search.
type Flat struct {
	dimension    int     // Dimension of vectors
	nodes        []*Node // Nodes in the index
	distanceFunc index.DistanceFunc
	opts         Options    // Options for the index
	mutex        sync.Mutex // Mutex for concurrent access
}

// New creates a new instance of the flat index with the given dimension and options.
func New(dimension int, optFns ...func(o *Options)) *Flat {
	opts := DefaultOptions

	for _, fn := range optFns {
		fn(&opts)
	}

	return &Flat{
		dimension:    dimension,
		nodes:        []*Node{{ID: 0, Vector: make([]float32, dimension)}},
		distanceFunc: index.NewDistanceFunc(opts.DistanceType),
		opts:         opts,
	}
}

// Insert inserts a vector into the flat index.
func (f *Flat) Insert(v []float32) (uint32, error) {
	// Check if dimensions of the input vector match the expected dimension
	if len(v) != f.dimension {
		return 0, &index.ErrDimensionMismatch{Expected: f.dimension, Actual: len(v)}
	}

	// Make a copy of the vector to ensure changes outside this function don't affect the node
	vectorCopy := make([]float32, len(v))
	copy(vectorCopy, v)

	f.mutex.Lock()
	defer f.mutex.Unlock()

	// next ID
	id := uint32(len(f.nodes))

	// Append new node
	f.nodes = append(f.nodes, &Node{ID: id, Vector: vectorCopy})

	return id, nil
}

// KNNSearch performs a K-nearest neighbor search in the flat index.
func (f *Flat) KNNSearch(q []float32, k int, efSearch int, filter func(id uint32) bool) (*queue.PriorityQueue, error) {
	return f.BruteSearch(q, k, filter)
}

// BruteSearch performs a brute-force search in the flat index.
func (f *Flat) BruteSearch(query []float32, k int, filter func(id uint32) bool) (*queue.PriorityQueue, error) {
	topCandidates := &queue.PriorityQueue{
		Order: true,
	}

	heap.Init(topCandidates)

	for _, node := range f.nodes {
		if !filter(node.ID) || node.ID == 0 {
			continue
		}

		nodeDist, err := f.distanceFunc(query, node.Vector)
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
