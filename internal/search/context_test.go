package search

import (
	"testing"

	"github.com/hupe1980/vecgo/internal/queue"
	"github.com/stretchr/testify/assert"
)

func TestSearcher_Reset(t *testing.T) {
	s := NewSearcher(100, 128)

	// Dirty the state
	s.Visited.Visit(1)
	s.Candidates.PushItem(queue.PriorityQueueItem{Node: 1, Distance: 0.5})
	s.ScratchCandidates.PushItem(queue.PriorityQueueItem{Node: 2, Distance: 0.6})
	s.OpsPerformed = 10

	// Reset
	s.Reset()

	// Verify clean state
	assert.False(t, s.Visited.Visited(1))
	assert.Equal(t, 0, s.Candidates.Len())
	assert.Equal(t, 0, s.ScratchCandidates.Len())
	assert.Equal(t, 0, s.OpsPerformed)
}

func TestSearcher_Heaps(t *testing.T) {
	s := NewSearcher(100, 128)

	// Candidates should be MaxHeap (for results - keep smallest)
	s.Candidates.PushItem(queue.PriorityQueueItem{Node: 1, Distance: 10.0})
	s.Candidates.PushItem(queue.PriorityQueueItem{Node: 2, Distance: 5.0})
	s.Candidates.PushItem(queue.PriorityQueueItem{Node: 3, Distance: 20.0})

	top, _ := s.Candidates.TopItem()
	assert.Equal(t, float32(20.0), top.Distance, "Candidates should be MaxHeap (top is largest)")

	// ScratchCandidates should be MinHeap (for exploration - explore closest)
	s.ScratchCandidates.PushItem(queue.PriorityQueueItem{Node: 1, Distance: 10.0})
	s.ScratchCandidates.PushItem(queue.PriorityQueueItem{Node: 2, Distance: 5.0})
	s.ScratchCandidates.PushItem(queue.PriorityQueueItem{Node: 3, Distance: 20.0})

	top, _ = s.ScratchCandidates.TopItem()
	assert.Equal(t, float32(5.0), top.Distance, "ScratchCandidates should be MinHeap (top is smallest)")
}
