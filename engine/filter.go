package engine

import (
	"github.com/hupe1980/vecgo/internal/segment"
	"github.com/hupe1980/vecgo/metadata"
)

// compositeFilter combines a tombstone filter and a user metadata filter.
type compositeFilter struct {
	ts         *metadata.LocalBitmap
	userFilter *metadata.FilterSet
	// We need a way to resolve RowID -> Metadata to check the user filter.
	// This requires the segment to provide metadata access.
	// But segment.Filter.Matches(rowID) is called by the segment itself.
	// The segment knows how to look up metadata for a rowID.
	// Wait, the segment.Filter interface is:
	// Matches(id uint32) bool
	//
	// If we pass a *metadata.FilterSet as the filter, the segment needs to know how to apply it.
	// But the segment is generic.
	//
	// The Engine constructs the filter.
	// If the filter depends on metadata, the Engine needs to provide a way to access it?
	// Or the Segment implementation handles *metadata.FilterSet specially?
	//
	// In Vecgo, metadata is stored in the segment (or alongside it).
	// The Segment implementation (Flat, DiskANN, MemTable) is responsible for applying the filter.
	//
	// So, we should pass the *metadata.FilterSet to the segment via SearchOptions (which we did).
	// And the `filter` argument to `Search` is for *internal* filtering (like tombstones).
	//
	// However, we might want to combine them.
	// If the segment supports `segment.Filter`, it calls `Matches(rowID)`.
	// If we want to filter by metadata, we need to check metadata.
	//
	// Strategy:
	// 1. Pass user filter via `options.Filter`.
	// 2. Pass tombstone filter via `filter` argument.
	// 3. Segment implementation combines them.
	//
	// Let's verify this assumption by looking at `segment.Segment.Search` signature.
	// Search(ctx, q, k, filter Filter, opts model.SearchOptions)
	//
	// If we pass `options.Filter` (*metadata.FilterSet), the segment can use it.
	// The `filter` argument is typically for bitsets/tombstones.
	//
	// So, we don't need a composite filter in the Engine *if* the segments handle both.
	// But `tombstoneFilter` implements `segment.Filter`.
	//
	// Let's look at `tombstoneFilter` again.
}

// tombstoneFilter implements segment.Filter for deleted rows.
type tombstoneFilter struct {
	ts *metadata.LocalBitmap
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
