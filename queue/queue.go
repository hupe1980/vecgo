package queue

import "container/heap"

// Compile time check to ensure PriorityQueue satisfies the heap interface.
var _ heap.Interface = (*PriorityQueue)(nil)

// PriorityQueueItem represents an item in the priority queue.
type PriorityQueueItem struct {
	Node     uint32  // Node is the value of the item, which can be arbitrary.
	Distance float32 // Distance is the priority of the item in the queue.
	Index    int     // Index is needed by update and is maintained by the heap.Interface methods.
}

// PriorityQueue implements heap.Interface and holds PriorityQueueItems.
type PriorityQueue struct {
	Order bool                 // Order specifies whether the priority queue is in ascending or descending order.
	Items []*PriorityQueueItem // Items contains the elements of the priority queue.
}

// Len returns the number of elements in the priority queue.
func (pq *PriorityQueue) Len() int { return len(pq.Items) }

// Less reports whether the element with index i should sort before the element with index j.
func (pq *PriorityQueue) Less(i, j int) bool {
	if !pq.Order {
		return pq.Items[i].Distance < pq.Items[j].Distance
	} else {
		return pq.Items[i].Distance > pq.Items[j].Distance
	}
}

// Swap swaps the elements with indexes i and j.
func (pq *PriorityQueue) Swap(i, j int) {
	pq.Items[i], pq.Items[j] = pq.Items[j], pq.Items[i]
	pq.Items[i].Index, pq.Items[j].Index = i, j // Update indices
}

// Push adds x to the priority queue.
func (pq *PriorityQueue) Push(x any) {
	item, _ := x.(*PriorityQueueItem)
	item.Index = len(pq.Items)
	pq.Items = append(pq.Items, item)
}

// Pop removes and returns the top element from the priority queue.
func (pq *PriorityQueue) Pop() any {
	if len(pq.Items) == 0 {
		return nil // Or handle the error accordingly
	}

	old := pq.Items
	n := len(old)
	item := old[n-1]
	old[n-1] = nil       // Avoid memory leak
	item.Index = -1      // For safety
	pq.Items = old[:n-1] // Reslice without creating a new underlying array

	return item
}

// Top returns the top element of the priority queue.
func (pq *PriorityQueue) Top() any {
	return pq.Items[0]
}
