package searcher

import (
	"math/rand"
	"testing"
	"time"

	"github.com/hupe1980/vecgo/model"
)

func TestPriorityQueue(t *testing.T) {
	t.Run("MinHeap", func(t *testing.T) {
		pq := NewPriorityQueue(false) // false = MinHeap

		pq.PushItem(PriorityQueueItem{Node: 1, Distance: 10.0})
		pq.PushItem(PriorityQueueItem{Node: 2, Distance: 5.0})
		pq.PushItem(PriorityQueueItem{Node: 3, Distance: 20.0})

		if pq.Len() != 3 {
			t.Errorf("expected len 3, got %d", pq.Len())
		}

		// Top should be Min (5.0)
		top, ok := pq.TopItem()
		if !ok || top.Distance != 5.0 {
			t.Errorf("expected top 5.0, got %v", top.Distance)
		}

		// MinItem (O(N)) - for MinHeap it's the same as Top
		min, ok := pq.MinItem()
		if !ok || min.Distance != 5.0 {
			t.Errorf("expected min 5.0, got %v", min.Distance)
		}

		// Pop order: 5, 10, 20
		item, ok := pq.PopItem()
		if !ok || item.Distance != 5.0 {
			t.Errorf("pop 1: expected 5.0, got %v", item.Distance)
		}

		item, ok = pq.PopItem()
		if !ok || item.Distance != 10.0 {
			t.Errorf("pop 2: expected 10.0, got %v", item.Distance)
		}

		item, ok = pq.PopItem()
		if item.Distance != 20.0 {
			t.Errorf("pop 3: expected 20.0, got %v", item.Distance)
		}
	})

	t.Run("MaxHeap", func(t *testing.T) {
		pq := NewPriorityQueue(true) // true = MaxHeap

		pq.PushItem(PriorityQueueItem{Node: 1, Distance: 10.0})
		pq.PushItem(PriorityQueueItem{Node: 2, Distance: 5.0})
		pq.PushItem(PriorityQueueItem{Node: 3, Distance: 20.0})

		// Top should be Max (20.0)
		top, ok := pq.TopItem()
		if !ok || top.Distance != 20.0 {
			t.Errorf("expected top 20.0, got %v", top.Distance)
		}

		// MinItem (O(N)) should find 5.0
		min, ok := pq.MinItem()
		if !ok || min.Distance != 5.0 {
			t.Errorf("expected min 5.0, got %v", min.Distance)
		}

		// Pop order: 20, 10, 5
		item, _ := pq.PopItem()
		if item.Distance != 20.0 {
			t.Errorf("pop 1: expected 20.0, got %v", item.Distance)
		}
	})

	t.Run("PushItemBounded", func(t *testing.T) {
		// Verify behavior of PushItemBounded for MaxHeap (keeping smallest/best items, evicting largest)
		// Wait, PushItemBounded is usually for collecting "best" candidates.
		// If we want "closest k nodes", we use a MaxHeap of size k.
		// If new item < Top (max), we pop Top and push new item.

		pq := NewPriorityQueue(true) // MaxHeap
		capacity := 3

		// Fill up
		pq.PushItemBounded(PriorityQueueItem{Node: 1, Distance: 10.0}, capacity)
		pq.PushItemBounded(PriorityQueueItem{Node: 2, Distance: 20.0}, capacity)
		pq.PushItemBounded(PriorityQueueItem{Node: 3, Distance: 30.0}, capacity)

		// Heap: [30, 10, 20] (structure may vary but max is 30)
		top, _ := pq.TopItem()
		if top.Distance != 30.0 {
			t.Errorf("expected max 30.0, got %v", top.Distance)
		}

		// Push something smaller (better): 5.0
		// Should evict 30.0, result: {10, 20, 5} -> Max is 20
		pq.PushItemBounded(PriorityQueueItem{Node: 4, Distance: 5.0}, capacity)

		if pq.Len() != 3 {
			t.Errorf("expected len 3, got %d", pq.Len())
		}

		top, _ = pq.TopItem()
		if top.Distance != 20.0 {
			t.Errorf("expected max 20.0 after update, got %v", top.Distance)
		}

		// Push something larger (worse): 40.0
		// Should be ignored
		pq.PushItemBounded(PriorityQueueItem{Node: 5, Distance: 40.0}, capacity)

		top, _ = pq.TopItem()
		if top.Distance != 20.0 {
			t.Errorf("expected max 20.0 (ignored 40), got %v", top.Distance)
		}

		// MinHeap Bounded
		pqMin := NewPriorityQueue(false) // MinHeap
		pqMin.PushItemBounded(PriorityQueueItem{Distance: 10}, 2)
		pqMin.PushItemBounded(PriorityQueueItem{Distance: 20}, 2)

		// Push 30 (Better, larger). Replace 10 (Min).
		pqMin.PushItemBounded(PriorityQueueItem{Distance: 30}, 2)

		topMin, _ := pqMin.TopItem()
		if topMin.Distance != 20 {
			t.Errorf("MinHeap Bounded: expected top 20, got %v", topMin.Distance)
		}
	})

	t.Run("ZeroAllocations", func(t *testing.T) {
		// Reset logic
		pq := NewPriorityQueue(false)
		pq.PushItem(PriorityQueueItem{Node: 1, Distance: 1.0})
		pq.Reset()
		if pq.Len() != 0 {
			t.Error("expected 0 after reset")
		}

		// Test repeated reset/push doesn't realloc (can't easily test capacity, but functional check)
		for i := 0; i < 1000; i++ {
			pq.PushItem(PriorityQueueItem{Node: model.RowID(i), Distance: float32(i)})
		}
		pq.Reset()
		if pq.Len() != 0 {
			t.Error("expected 0")
		}
	})

	t.Run("Stress", func(t *testing.T) {
		pq := NewPriorityQueue(false) // MinHeap
		rng := rand.New(rand.NewSource(time.Now().UnixNano()))

		// Push 1000 randoms
		for i := 0; i < 1000; i++ {
			pq.PushItem(PriorityQueueItem{Node: model.RowID(i), Distance: rng.Float32()})
		}

		// Pop all and verify order
		var last float32 = -1.0
		for pq.Len() > 0 {
			item, _ := pq.PopItem()
			if last >= 0 && item.Distance < last {
				t.Fatalf("heap invariant violated: %v < %v", item.Distance, last)
			}
			last = item.Distance
		}
	})

	t.Run("Empty", func(t *testing.T) {
		pq := NewPriorityQueue(false)
		if _, ok := pq.TopItem(); ok {
			t.Error("TopItem on empty should return false")
		}
		if _, ok := pq.MinItem(); ok {
			t.Error("MinItem on empty should return false")
		}
		if _, ok := pq.PopItem(); ok {
			t.Error("PopItem on empty should return false")
		}
	})
}
