// Package flat provides an implementation of a flat index for vector storage and search.
package flat

import (
	"container/heap"
	"sync"
	"sync/atomic"

	"github.com/ylerby/vecgo/index"
	"github.com/ylerby/vecgo/queue"
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
	sync.RWMutex

	dimension    int32   // Dimension of vectors
	nodes        []*Node // Nodes in the index
	distanceFunc index.DistanceFunc
	opts         Options // Options for the index
	initOnce     *sync.Once
	insertOnce   *sync.Once
}

// New creates a new instance of the flat index with the given dimension and options.
func New(optFns ...func(o *Options)) *Flat {
	opts := DefaultOptions

	for _, fn := range optFns {
		fn(&opts)
	}

	return &Flat{
		distanceFunc: index.NewDistanceFunc(opts.DistanceType),
		opts:         opts,

		initOnce:   &sync.Once{},
		insertOnce: &sync.Once{},
	}
}

// Insert inserts a vector into the flat index.
func (f *Flat) Insert(v []float32) (uint32, error) {
	f.initOnce.Do(func() {
		if f.isEmpty() {
			atomic.StoreInt32(&f.dimension, int32(len(v)))
		}
	})

	// Check if dimensions of the input vector match the expected dimension
	dim := int(atomic.LoadInt32(&f.dimension))
	if len(v) != dim {
		return 0, &index.ErrDimensionMismatch{Expected: dim, Actual: len(v)}
	}

	// Make a copy of the vector to ensure changes outside this function don't affect the node
	vectorCopy := make([]float32, len(v))
	copy(vectorCopy, v)

	wasFirst := false

	f.insertOnce.Do(func() {
		if f.isEmpty() {
			f.Lock()
			defer f.Unlock()

			wasFirst = true
			f.nodes = []*Node{{ID: 0, Vector: vectorCopy}}
		}
	})

	if wasFirst {
		return 0, nil
	}

	f.Lock()
	defer f.Unlock()

	// next ID
	id := uint32(len(f.nodes))

	// Append new node
	f.nodes = append(f.nodes, &Node{ID: id, Vector: vectorCopy})

	return id, nil
}

// KNNSearch performs a K-nearest neighbor search in the flat index.
func (f *Flat) KNNSearch(q []float32, k int, efSearch int, filter func(id uint32) bool) ([]index.SearchResult, error) {
	return f.BruteSearch(q, k, filter)
}

// BruteSearch performs a brute-force search in the flat index.
func (f *Flat) BruteSearch(query []float32, k int, filter func(id uint32) bool) ([]index.SearchResult, error) {
	if f.isEmpty() {
		return nil, nil
	}

	topCandidates := queue.NewMax(k)
	heap.Init(topCandidates)

	for _, node := range f.nodes {
		if !filter(node.ID) {
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

	results := make([]index.SearchResult, topCandidates.Len())

	for i := topCandidates.Len() - 1; i >= 0; i-- {
		item, _ := heap.Pop(topCandidates).(*queue.PriorityQueueItem)
		results[i] = index.SearchResult{
			ID:       item.Node,
			Distance: item.Distance,
		}
	}

	return results, nil
}

func (f *Flat) isEmpty() bool {
	f.RLock()
	defer f.RUnlock()

	return len(f.nodes) == 0
}
