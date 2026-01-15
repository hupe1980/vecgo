package memtable

import (
	"sync"

	"github.com/hupe1980/vecgo/internal/bitset"
	"github.com/hupe1980/vecgo/internal/segment"
	"github.com/hupe1980/vecgo/metadata"
)

type columnarFilterWrapper struct {
	parent   segment.Filter
	checks   []columnCheck
	rowCount uint32

	once   sync.Once
	bitmap *bitset.BitSet

	// Cached cardinality (computed cheaply without full bitmap)
	cardOnce  sync.Once
	cardCache uint64
}

type columnCheck struct {
	col    Column
	filter metadata.Filter
}

func newColumnarFilterWrapper(parent segment.Filter, meta *metadata.FilterSet, columns map[string]Column, rowCount uint32) *columnarFilterWrapper {
	checks := make([]columnCheck, 0, len(meta.Filters))
	for _, f := range meta.Filters {
		if col, ok := columns[f.Key]; ok {
			checks = append(checks, columnCheck{col: col, filter: f})
		} else {
			// Column doesn't exist, this filter will always fail (assuming implicit existence check)
			checks = append(checks, columnCheck{col: nil, filter: f})
		}
	}

	return &columnarFilterWrapper{
		parent:   parent,
		checks:   checks,
		rowCount: rowCount,
	}
}

func (w *columnarFilterWrapper) Matches(id uint32) bool {
	idInt := int(id)
	for _, check := range w.checks {
		if check.col == nil {
			return false
		}
		// Optimized direct column check
		if !check.col.Matches(idInt, check.filter.Value, check.filter.Operator) {
			return false
		}
	}

	if w.parent != nil && !w.parent.Matches(id) {
		return false
	}
	return true
}

func (w *columnarFilterWrapper) MatchesBatch(ids []uint32, out []bool) {
	// Naive implementation for now, can be vectorized later
	for i, id := range ids {
		out[i] = w.Matches(id)
	}
}

func (w *columnarFilterWrapper) compute() {
	if w.rowCount == 0 {
		w.bitmap = bitset.New(0)
		return
	}

	// Create lazy bitmap
	w.bitmap = bitset.New(w.rowCount)

	// Naive scan to populate
	// Optimization: This could be parallelized or use vector instructions if columns support it.
	for i := uint32(0); i < w.rowCount; i++ {
		if w.Matches(i) {
			w.bitmap.Set(i)
		}
	}
}

func (w *columnarFilterWrapper) Cardinality() uint64 {
	// Fast path: compute cardinality without materializing full bitmap
	// This is O(n) scan but avoids expensive bitmap allocation
	w.cardOnce.Do(func() {
		var count uint64
		for i := uint32(0); i < w.rowCount; i++ {
			if w.Matches(i) {
				count++
			}
		}
		w.cardCache = count
	})
	return w.cardCache
}

func (w *columnarFilterWrapper) Contains(id uint32) bool {
	w.once.Do(w.compute)
	return w.bitmap.Test(id)
}

func (w *columnarFilterWrapper) ForEach(fn func(id uint32) bool) {
	w.once.Do(w.compute)

	id, ok := w.bitmap.NextSetBit(0)
	for ok {
		if !fn(id) {
			return
		}
		id, ok = w.bitmap.NextSetBit(id + 1)
	}
}

// ToArrayInto copies all set bits into dst, returning the populated slice.
func (w *columnarFilterWrapper) ToArrayInto(dst []uint32) []uint32 {
	w.once.Do(w.compute)

	count := w.bitmap.Count()
	if cap(dst) < int(count) {
		dst = make([]uint32, count)
	} else {
		dst = dst[:count]
	}

	idx := 0
	id, ok := w.bitmap.NextSetBit(0)
	for ok && idx < len(dst) {
		dst[idx] = id
		idx++
		id, ok = w.bitmap.NextSetBit(id + 1)
	}
	return dst[:idx]
}

func (w *columnarFilterWrapper) AsBitmap() segment.Bitmap {
	w.once.Do(w.compute)
	// If selectivity is high, this bitmap is useful.
	// HNSW optimization checks "if bm, ok := filter.(segment.Bitmap)".
	// Does HNSW checking `filter.(segment.Bitmap)` work if `filter` is `segment.Filter` interface?
	// Yes, if the underlying struct implements it.
	// But `AsBitmap()` is separate.
	return w
}

func (w *columnarFilterWrapper) MatchesBlock(stats map[string]segment.FieldStats) bool {
	return true
}
