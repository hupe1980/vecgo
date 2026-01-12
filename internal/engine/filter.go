package engine

import (
	imetadata "github.com/hupe1980/vecgo/internal/metadata"
	"github.com/hupe1980/vecgo/internal/segment"
)

// tombstoneFilter implements segment.Filter for deleted rows.
type tombstoneFilter struct {
	ts *imetadata.LocalBitmap
}

func (f *tombstoneFilter) Matches(rowID uint32) bool {
	return !f.ts.Contains(rowID)
}

func (f *tombstoneFilter) MatchesBatch(ids []uint32, out []bool) {
	for i, id := range ids {
		out[i] = !f.ts.Contains(id)
	}
}

func (f *tombstoneFilter) AsBitmap() segment.Bitmap {
	return f.ts
}

func (f *tombstoneFilter) MatchesBlock(stats map[string]segment.FieldStats) bool {
	return true
}

// watermarkFilter excludes rows >= limit.
type watermarkFilter struct {
	limit uint32
	next  segment.Filter
}

func (f *watermarkFilter) Matches(rowID uint32) bool {
	if rowID >= f.limit {
		return false
	}
	if f.next != nil {
		return f.next.Matches(rowID)
	}
	return true
}

func (f *watermarkFilter) MatchesBatch(ids []uint32, out []bool) {
	if f.next != nil {
		f.next.MatchesBatch(ids, out)
	} else {
		for i := range out {
			out[i] = true
		}
	}

	for i, id := range ids {
		if id >= f.limit {
			out[i] = false
		}
	}
}

func (f *watermarkFilter) AsBitmap() segment.Bitmap {
	return nil
}

func (f *watermarkFilter) MatchesBlock(stats map[string]segment.FieldStats) bool {
	return true
}
