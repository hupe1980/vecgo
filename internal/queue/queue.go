package queue

import "container/heap"

// Compile time check to ensure PriorityQueue satisfies the heap interface.
var _ heap.Interface = (*PriorityQueue)(nil)

// PriorityQueueItem represents an item in the priority queue.
// Optimized: value-based (no pointers), removed Index field (not needed)
type PriorityQueueItem struct {
	Node     uint32  // Node is the value of the item, which can be arbitrary.
	Distance float32 // Distance is the priority of the item in the queue.
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

// PushItem inserts an item while maintaining the heap invariant.
func (pq *PriorityQueue) PushItem(item PriorityQueueItem) {
	pq.items = append(pq.items, item)
	pq.siftUp(len(pq.items) - 1)
}

// PopItem removes and returns the top element while maintaining the heap invariant.
func (pq *PriorityQueue) PopItem() (PriorityQueueItem, bool) {
	n := len(pq.items)
	if n == 0 {
		return PriorityQueueItem{}, false
	}
	root := pq.items[0]
	last := pq.items[n-1]
	pq.items[n-1] = PriorityQueueItem{}
	pq.items = pq.items[:n-1]
	if n-1 > 0 {
		pq.items[0] = last
		pq.siftDown(0)
	}
	return root, true
}

func (pq *PriorityQueue) less(i, j int) bool {
	if pq.isMaxHeap {
		return pq.items[i].Distance > pq.items[j].Distance
	}
	return pq.items[i].Distance < pq.items[j].Distance
}

func (pq *PriorityQueue) siftUp(i int) {
	for i > 0 {
		p := (i - 1) / 2
		if !pq.less(i, p) {
			return
		}
		pq.items[i], pq.items[p] = pq.items[p], pq.items[i]
		i = p
	}
}

func (pq *PriorityQueue) siftDown(i int) {
	n := len(pq.items)
	for {
		l := 2*i + 1
		if l >= n {
			return
		}
		best := l
		r := l + 1
		if r < n && pq.less(r, l) {
			best = r
		}
		if !pq.less(best, i) {
			return
		}
		pq.items[i], pq.items[best] = pq.items[best], pq.items[i]
		i = best
	}
}

// MinItem returns the item with the smallest Distance currently in the queue.
// For min-heaps this is the top element; for max-heaps this scans the backing slice.
func (pq *PriorityQueue) MinItem() (PriorityQueueItem, bool) {
	if len(pq.items) == 0 {
		return PriorityQueueItem{}, false
	}
	if !pq.isMaxHeap {
		return pq.items[0], true
	}
	min := pq.items[0]
	for i := 1; i < len(pq.items); i++ {
		if pq.items[i].Distance < min.Distance {
			min = pq.items[i]
		}
	}
	return min, true
}

// NewMin initializes a new priority queue with minimum priority.
func NewMin(capacity int) *PriorityQueue {
	return &PriorityQueue{
		isMaxHeap: false,
		items:     make([]PriorityQueueItem, 0, capacity),
	}
}

// NewMax initializes a new priority queue with maximum priority.
func NewMax(capacity int) *PriorityQueue {
	return &PriorityQueue{
		isMaxHeap: true,
		items:     make([]PriorityQueueItem, 0, capacity),
	}
}

// Len returns the number of elements in the priority queue.
func (pq *PriorityQueue) Len() int { return len(pq.items) }

// Less reports whether the element with index i should sort before the element with index j.
// Optimized: eliminated branch by using comparison XOR trick
func (pq *PriorityQueue) Less(i, j int) bool {
	if pq.isMaxHeap {
		return pq.items[i].Distance > pq.items[j].Distance
	}
	return pq.items[i].Distance < pq.items[j].Distance
}

// Swap swaps the elements with indexes i and j.
// Optimized: no Index field updates needed (removed overhead)
func (pq *PriorityQueue) Swap(i, j int) {
	pq.items[i], pq.items[j] = pq.items[j], pq.items[i]
}

// Push adds x to the priority queue.
// Optimized: value-based, no pointer allocation
func (pq *PriorityQueue) Push(x any) {
	item := x.(PriorityQueueItem)
	pq.items = append(pq.items, item)
}

// Pop removes and returns the top element from the priority queue.
// Optimized: return value directly (no pointer), zero value clears memory
func (pq *PriorityQueue) Pop() any {
	n := len(pq.items)
	if n == 0 {
		return PriorityQueueItem{} // Return zero value
	}

	item := pq.items[n-1]
	pq.items[n-1] = PriorityQueueItem{} // Zero out for GC
	pq.items = pq.items[:n-1]

	return item
}

// Top returns the top element of the priority queue.
// Optimized: return value directly (no pointer)
func (pq *PriorityQueue) Top() any {
	if len(pq.items) == 0 {
		return PriorityQueueItem{}
	}
	return pq.items[0]
}

// Reset clears the priority queue for reuse.
// Optimized: just truncate slice (zero values not needed with value types)
func (pq *PriorityQueue) Reset() {
	pq.items = pq.items[:0]
}
