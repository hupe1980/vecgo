package memtable

import (
	"github.com/hupe1980/vecgo/internal/segment"
	"github.com/hupe1980/vecgo/metadata"
)

type columnarFilterWrapper struct {
	parent segment.Filter
	checks []columnCheck
}

type columnCheck struct {
	col    Column
	filter metadata.Filter
}

func newColumnarFilterWrapper(parent segment.Filter, meta *metadata.FilterSet, columns map[string]Column) *columnarFilterWrapper {
	checks := make([]columnCheck, 0, len(meta.Filters))
	for _, f := range meta.Filters {
		if col, ok := columns[f.Key]; ok {
			checks = append(checks, columnCheck{col: col, filter: f})
		} else {
			// Column doesn't exist, this filter will always fail (assuming implicit existence check)
			// Return a wrapper that always returns false (except maybe for NotEqual? strictness says false)
			// We can represent this as a nil col or ensure Matches returns false.
			// Let's rely on logic: if col missing, we can never satisfy "exists and matches".
			// So we add a check with nil col.
			checks = append(checks, columnCheck{col: nil, filter: f})
			// Optimization: if any AND filter is impossible, the whole thing is impossible.
			// We could return a "AlwaysFalse" filter.
		}
	}

	return &columnarFilterWrapper{
		parent: parent,
		checks: checks,
	}
}

func (w *columnarFilterWrapper) Matches(id uint32) bool {
	if w.parent != nil && !w.parent.Matches(id) {
		return false
	}
	idInt := int(id)
	for _, check := range w.checks {
		if check.col == nil {
			return false
		}
		val, ok := check.col.Get(idInt)
		if !ok {
			return false
		}
		if !check.filter.MatchesValue(val) {
			return false
		}
	}
	return true
}

func (w *columnarFilterWrapper) MatchesBatch(ids []uint32, out []bool) {
	// Naive implementation for now, can be vectorized later
	for i, id := range ids {
		out[i] = w.Matches(id)
	}
}

func (w *columnarFilterWrapper) AsBitmap() segment.Bitmap {
	return nil
}

func (w *columnarFilterWrapper) MatchesBlock(stats map[string]segment.FieldStats) bool {
	return true
}
