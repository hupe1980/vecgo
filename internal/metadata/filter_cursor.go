package imetadata

import (
	"github.com/hupe1980/vecgo/metadata"
)

// FilterCursor is a push-based iterator for filter results.
// Unlike materialized FilterResult (which builds a bitmap/array),
// FilterCursor pushes matching row IDs directly to the consumer.
//
// This design eliminates:
//   - Roaring bitmap allocations (OR, Clone, Iterator)
//   - Memory materialization (no intermediate arrays)
//   - Early termination overhead (consumer controls flow)
//
// Architecture: Filters push rows, search pulls nothing.
//
// Example usage:
//
//	cursor.ForEach(func(rowID uint32) bool {
//	    distance := computeDistance(query, vectors[rowID])
//	    heap.PushMaybe(rowID, distance)
//	    return heap.Len() < k*10 // early termination
//	})
type FilterCursor interface {
	// ForEach iterates over matching row IDs, calling fn for each.
	// Iteration stops early if fn returns false.
	// Zero allocations in the iteration path.
	ForEach(fn func(rowID uint32) bool)

	// EstimateCardinality returns an estimate of matching row IDs.
	// This is used for selectivity-based execution planning.
	// The estimate may be approximate (especially for complex filters).
	EstimateCardinality() int

	// IsEmpty returns true if no rows match (fast path).
	IsEmpty() bool

	// IsAll returns true if all rows match (fast path for no-filter case).
	IsAll() bool
}

// RowsCursor wraps a sorted []uint32 slice as a FilterCursor.
// This is the most common cursor type for low-cardinality results.
// Zero allocations - borrows the slice.
type RowsCursor struct {
	rows []uint32
}

// NewRowsCursor creates a FilterCursor from a sorted row slice.
// The slice is NOT copied - caller must ensure it outlives the cursor.
func NewRowsCursor(rows []uint32) *RowsCursor {
	return &RowsCursor{rows: rows}
}

func (c *RowsCursor) ForEach(fn func(rowID uint32) bool) {
	for _, id := range c.rows {
		if !fn(id) {
			return
		}
	}
}

func (c *RowsCursor) EstimateCardinality() int { return len(c.rows) }
func (c *RowsCursor) IsEmpty() bool            { return len(c.rows) == 0 }
func (c *RowsCursor) IsAll() bool              { return false }

// RangeCursor iterates over a contiguous range [start, end).
// Extremely efficient for temporal/sequential data.
// Zero storage, zero allocations.
type RangeCursor struct {
	start, end uint32
}

// NewRangeCursor creates a FilterCursor for a contiguous range.
func NewRangeCursor(start, end uint32) *RangeCursor {
	return &RangeCursor{start: start, end: end}
}

func (c *RangeCursor) ForEach(fn func(rowID uint32) bool) {
	for id := c.start; id < c.end; id++ {
		if !fn(id) {
			return
		}
	}
}

func (c *RangeCursor) EstimateCardinality() int { return int(c.end - c.start) }
func (c *RangeCursor) IsEmpty() bool            { return c.start >= c.end }
func (c *RangeCursor) IsAll() bool              { return false }

// AllCursor represents "all rows match" (unfiltered query).
// Iterates 0..size-1.
type AllCursor struct {
	size uint32
}

// NewAllCursor creates a FilterCursor that matches all rows in [0, size).
func NewAllCursor(size uint32) *AllCursor {
	return &AllCursor{size: size}
}

func (c *AllCursor) ForEach(fn func(rowID uint32) bool) {
	for id := uint32(0); id < c.size; id++ {
		if !fn(id) {
			return
		}
	}
}

func (c *AllCursor) EstimateCardinality() int { return int(c.size) }
func (c *AllCursor) IsEmpty() bool            { return c.size == 0 }
func (c *AllCursor) IsAll() bool              { return true }

// EmptyCursor represents "no rows match".
type EmptyCursor struct{}

func (c *EmptyCursor) ForEach(fn func(rowID uint32) bool) {}
func (c *EmptyCursor) EstimateCardinality() int           { return 0 }
func (c *EmptyCursor) IsEmpty() bool                      { return true }
func (c *EmptyCursor) IsAll() bool                        { return false }

var emptyCursor = &EmptyCursor{}

// GetEmptyCursor returns a singleton empty cursor.
func GetEmptyCursor() FilterCursor { return emptyCursor }

// ColumnFilterCursor evaluates filters directly against column data.
// This is the push-based replacement for EvaluateFilterResult.
// Filters are evaluated lazily during ForEach - no intermediate bitmap.
//
// For AND logic (default): rows must pass ALL filters.
// Short-circuit evaluation stops at first non-match.
type ColumnFilterCursor struct {
	columns      map[string]Column // borrowed from shard
	filters      []metadata.Filter // borrowed from FilterSet
	universeSize uint32            // total rows in segment
	estimate     int               // cached selectivity estimate
}

// Column is the interface for typed column data.
// This matches the existing memtable column interface.
type Column interface {
	// Matches checks if row at index i matches the filter condition.
	Matches(i int, val metadata.Value, op metadata.Operator) bool
}

// NewColumnFilterCursor creates a cursor that evaluates filters against columns.
// Columns and filters are borrowed - caller must ensure they outlive the cursor.
func NewColumnFilterCursor(columns map[string]Column, filters []metadata.Filter, universeSize uint32) *ColumnFilterCursor {
	// Compute rough selectivity estimate (assumes independence)
	estimate := int(universeSize)
	for _, f := range filters {
		// Heuristic: OpEqual ~10%, OpNotEqual ~90%, OpIn ~20%, Range ~30%
		switch f.Operator {
		case metadata.OpEqual:
			estimate = estimate * 10 / 100
		case metadata.OpNotEqual:
			estimate = estimate * 90 / 100
		case metadata.OpIn:
			estimate = estimate * 20 / 100
		case metadata.OpGreaterThan, metadata.OpGreaterEqual, metadata.OpLessThan, metadata.OpLessEqual:
			estimate = estimate * 30 / 100
		}
		if estimate < 1 {
			estimate = 1
		}
	}

	return &ColumnFilterCursor{
		columns:      columns,
		filters:      filters,
		universeSize: universeSize,
		estimate:     estimate,
	}
}

func (c *ColumnFilterCursor) ForEach(fn func(rowID uint32) bool) {
	if len(c.filters) == 0 {
		// No filters = all rows match
		for id := uint32(0); id < c.universeSize; id++ {
			if !fn(id) {
				return
			}
		}
		return
	}

	// Scan all rows, checking filters with short-circuit AND
rowLoop:
	for id := uint32(0); id < c.universeSize; id++ {
		// Check all filters (AND logic)
		for _, f := range c.filters {
			col, ok := c.columns[f.Key]
			if !ok {
				// Column doesn't exist - no match
				continue rowLoop
			}
			if !col.Matches(int(id), f.Value, f.Operator) {
				continue rowLoop
			}
		}
		// Passed all filters
		if !fn(id) {
			return
		}
	}
}

func (c *ColumnFilterCursor) EstimateCardinality() int { return c.estimate }
func (c *ColumnFilterCursor) IsEmpty() bool            { return c.universeSize == 0 }
func (c *ColumnFilterCursor) IsAll() bool              { return len(c.filters) == 0 }

// TombstoneFilterCursor wraps another cursor and filters out tombstoned rows.
// This is a decorator pattern - composes with any underlying cursor.
type TombstoneFilterCursor struct {
	inner     FilterCursor
	tombstone func(rowID uint32) bool // returns true if row is alive
}

// NewTombstoneFilterCursor wraps a cursor with tombstone filtering.
// The alive function returns true if the row is NOT tombstoned.
func NewTombstoneFilterCursor(inner FilterCursor, alive func(rowID uint32) bool) *TombstoneFilterCursor {
	return &TombstoneFilterCursor{inner: inner, tombstone: alive}
}

func (c *TombstoneFilterCursor) ForEach(fn func(rowID uint32) bool) {
	c.inner.ForEach(func(rowID uint32) bool {
		if c.tombstone(rowID) {
			return fn(rowID)
		}
		return true // continue, skip tombstoned
	})
}

func (c *TombstoneFilterCursor) EstimateCardinality() int { return c.inner.EstimateCardinality() }
func (c *TombstoneFilterCursor) IsEmpty() bool            { return c.inner.IsEmpty() }
func (c *TombstoneFilterCursor) IsAll() bool              { return false } // tombstones make it not "all"

// FilterResultCursor wraps an existing FilterResult as a cursor.
// This provides backward compatibility during migration.
type FilterResultCursor struct {
	fr FilterResult
}

// NewFilterResultCursor wraps a FilterResult as a FilterCursor.
func NewFilterResultCursor(fr FilterResult) *FilterResultCursor {
	return &FilterResultCursor{fr: fr}
}

func (c *FilterResultCursor) ForEach(fn func(rowID uint32) bool) {
	c.fr.ForEach(fn)
}

func (c *FilterResultCursor) EstimateCardinality() int { return c.fr.Cardinality() }
func (c *FilterResultCursor) IsEmpty() bool            { return c.fr.IsEmpty() }
func (c *FilterResultCursor) IsAll() bool              { return c.fr.IsAll() }
