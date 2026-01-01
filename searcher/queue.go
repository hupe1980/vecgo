package searcher

import (
	"container/heap"

	"github.com/hupe1980/vecgo/core"
)

// Compile time check to ensure PriorityQueue satisfies the heap interface.
var _ heap.Interface = (*PriorityQueue)(nil)

// PriorityQueueItem represents an item in the priority queue.
// Optimized: value-based (no pointers), removed Index field (not needed)
type PriorityQueueItem struct {
	Node     core.LocalID // Node is the value of the item, which can be arbitrary.
	Distance float32      // Distance is the priority of the item in the queue.
}

// PriorityQueue implements heap.Interface and holds PriorityQueueItems.
// Optimized: value-based storage for better cache locality and zero allocations
type PriorityQueue struct {
	isMaxHeap bool                // true = max heap, false = min heap (renamed for clarity)
	items     []PriorityQueueItem // Value-based storage (no pointer indirection)
}

// TopItem returns the top element of the heap.
func (pq *PriorityQueue) TopItem() (PriorityQueueItem, bool) {
	if len(pq.items) == 0 {
		return PriorityQueueItem{}, false
	}
	return pq.items[0], true
}

// MinItem returns the item with the minimum distance in the queue.
// This is O(N) for a MaxHeap, but N (EF) is typically small.
func (pq *PriorityQueue) MinItem() (PriorityQueueItem, bool) {
	if len(pq.items) == 0 {
		return PriorityQueueItem{}, false
	}
	minItem := pq.items[0]
	for _, item := range pq.items[1:] {
		if item.Distance < minItem.Distance {
			minItem = item
		}
	}
	return minItem, true
}

// PushItem inserts an item while maintaining the heap invariant.
func (pq *PriorityQueue) PushItem(item PriorityQueueItem) {
	pq.items = append(pq.items, item)
	pq.siftUp(len(pq.items) - 1)
}

// PushItemBounded inserts an item into a bounded heap.
// If the heap is full and the new item is worse than the top, it is skipped.
// If the heap is full and the new item is better, the top is replaced.
func (pq *PriorityQueue) PushItemBounded(item PriorityQueueItem, capacity int) {
	if len(pq.items) < capacity {
		pq.PushItem(item)
		return
	}

	// Heap is full
	top, _ := pq.TopItem()
	if pq.isMaxHeap {
		// MaxHeap: Top is largest distance (worst candidate)
		// We want smallest distances.
		// If item.Distance < top.Distance, it's better.
		if item.Distance < top.Distance {
			pq.items[0] = item
			pq.siftDown(0)
		}
	} else {
		// MinHeap: Top is smallest distance (worst candidate)
		// We want largest distances.
		// If item.Distance > top.Distance, it's better.
		if item.Distance > top.Distance {
			pq.items[0] = item
			pq.siftDown(0)
		}
	}
}

// Len returns the number of elements in the heap.
func (pq *PriorityQueue) Len() int {
	return len(pq.items)
}

// Less reports whether the element with index i should sort before the element with index j.
func (pq *PriorityQueue) Less(i, j int) bool {
	if pq.isMaxHeap {
		return pq.items[i].Distance > pq.items[j].Distance
	}
	return pq.items[i].Distance < pq.items[j].Distance
}

// Swap swaps the elements with indexes i and j.
func (pq *PriorityQueue) Swap(i, j int) {
	pq.items[i], pq.items[j] = pq.items[j], pq.items[i]
}

// Push pushes the element x onto the heap.
func (pq *PriorityQueue) Push(x any) {
	item := x.(PriorityQueueItem)
	pq.items = append(pq.items, item)
}

// Pop removes and returns the minimum element (according to Less) from the heap.
func (pq *PriorityQueue) Pop() any {
	old := pq.items
	n := len(old)
	item := old[n-1]
	pq.items = old[0 : n-1]
	return item
}

// PopItem removes and returns the top element from the heap.
func (pq *PriorityQueue) PopItem() (PriorityQueueItem, bool) {
	n := len(pq.items)
	if n == 0 {
		return PriorityQueueItem{}, false
	}

	item := pq.items[0]
	pq.items[0] = pq.items[n-1]
	pq.items = pq.items[:n-1]

	if len(pq.items) > 0 {
		pq.siftDown(0)
	}

	return item, true
}

// NewPriorityQueue creates a new priority queue.
func NewPriorityQueue(isMaxHeap bool) *PriorityQueue {
	return &PriorityQueue{
		isMaxHeap: isMaxHeap,
		items:     make([]PriorityQueueItem, 0, 16),
	}
}

// siftUp moves the element at index i up the heap until the heap invariant is restored.
func (pq *PriorityQueue) siftUp(i int) {
	for i > 0 {
		parent := (i - 1) / 2
		if !pq.Less(i, parent) {
			break
		}
		pq.Swap(i, parent)
		i = parent
	}
}

// siftDown moves the element at index i down the heap until the heap invariant is restored.
func (pq *PriorityQueue) siftDown(i int) {
	n := len(pq.items)
	for {
		left := 2*i + 1
		if left >= n {
			break
		}
		child := left
		right := left + 1
		if right < n && pq.Less(right, left) {
			child = right
		}
		if !pq.Less(child, i) {
			break
		}
		pq.Swap(i, child)
		i = child
	}
}

// Reset clears the priority queue.
func (pq *PriorityQueue) Reset() {
	pq.items = pq.items[:0]
}
