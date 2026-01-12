package benchmark_test

import (
	"container/heap"
	"sort"

	"github.com/hupe1980/vecgo/distance"
	"github.com/hupe1980/vecgo/model"
)

type topKItem struct {
	pk   model.ID
	dist float32
}

// topKHeap is a max-heap by dist so the largest (worst) distance is popped first.
type topKHeap []topKItem

func (h topKHeap) Len() int           { return len(h) }
func (h topKHeap) Less(i, j int) bool { return h[i].dist > h[j].dist }
func (h topKHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *topKHeap) Push(x any)        { *h = append(*h, x.(topKItem)) }
func (h *topKHeap) Pop() any          { old := *h; n := len(old); x := old[n-1]; *h = old[:n-1]; return x }

func exactTopK_L2_WithIDs(data [][]float32, pks []model.ID, q []float32, k int) []model.ID {
	h := make(topKHeap, 0, k)
	for i, v := range data {
		d := distance.SquaredL2(q, v)
		pk := pks[i]
		if len(h) < k {
			heap.Push(&h, topKItem{pk: pk, dist: d})
			continue
		}
		if d < h[0].dist {
			h[0] = topKItem{pk: pk, dist: d}
			heap.Fix(&h, 0)
		}
	}

	out := make([]topKItem, len(h))
	copy(out, h)
	sort.Slice(out, func(i, j int) bool {
		return out[i].dist < out[j].dist
	})

	outPKs := make([]model.ID, len(out))
	for i := range out {
		outPKs[i] = out[i].pk
	}
	return outPKs
}

func recallAtK(results []model.Candidate, truth []model.ID) float64 {
	if len(truth) == 0 {
		return 0
	}

	set := make(map[model.ID]struct{}, len(truth))
	for _, pk := range truth {
		set[pk] = struct{}{}
	}
	var hit int
	for _, c := range results {
		if _, ok := set[c.ID]; ok {
			hit++
		}
	}
	return float64(hit) / float64(len(truth))
}
