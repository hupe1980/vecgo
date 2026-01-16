package engine

import (
	"github.com/hupe1980/vecgo/internal/manifest"
	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/model"
)

// canPruneSegment checks if a segment can be entirely skipped based on filter and stats.
// Returns true if the segment can definitely be skipped (no matches possible).
// Returns false if the segment might have matches and must be searched.
//
// This enables 2-10x query speedups by skipping segments before opening them.
// The check is O(1) per segment and has zero allocations.
func (e *Engine) canPruneSegment(segID model.SegmentID, filter *metadata.FilterSet) bool {
	if filter == nil || len(filter.Filters) == 0 {
		return false // No filter = can't prune
	}

	stats := e.getSegmentStats(segID)
	if stats == nil {
		return false // No stats = can't prune (legacy segment)
	}

	// Check each filter predicate against segment stats.
	// If ANY predicate can prune the segment, we can skip it.
	// For AND semantics, one false filter = whole segment matches nothing.
	for _, f := range filter.Filters {
		if canPruneByFilter(stats, f) {
			return true
		}
	}

	return false
}

// canPruneByFilter checks if a single filter predicate can prune the segment.
func canPruneByFilter(stats *manifest.SegmentStats, f metadata.Filter) bool {
	// Check if field exists in segment
	if !stats.HasField(f.Key) {
		// Field doesn't exist in segment.
		// For most operators (eq, gt, lt, etc.), missing field means no match.
		// Exception: "neq" on missing field might match all rows (depends on semantics).
		switch f.Operator {
		case metadata.OpNotEqual:
			return false // Can't prune: missing != X might be true
		default:
			return true // Can prune: field missing = no match
		}
	}

	// Extract numeric value for range checks
	val := f.Value
	if val.Kind == metadata.KindInt || val.Kind == metadata.KindFloat {
		var numVal float64
		if val.Kind == metadata.KindInt {
			numVal = float64(val.I64)
		} else {
			numVal = val.F64
		}

		// Use numeric pruning
		switch f.Operator {
		case metadata.OpEqual:
			return stats.CanPruneNumeric(f.Key, "eq", numVal)
		case metadata.OpNotEqual:
			return stats.CanPruneNumeric(f.Key, "neq", numVal)
		case metadata.OpLessThan:
			return stats.CanPruneNumeric(f.Key, "lt", numVal)
		case metadata.OpLessEqual:
			return stats.CanPruneNumeric(f.Key, "lte", numVal)
		case metadata.OpGreaterThan:
			return stats.CanPruneNumeric(f.Key, "gt", numVal)
		case metadata.OpGreaterEqual:
			return stats.CanPruneNumeric(f.Key, "gte", numVal)
		}
	}

	// For IN operator with array values, check if any value could match
	if f.Operator == metadata.OpIn && val.Kind == metadata.KindArray {
		allPruned := true
		for _, v := range val.A {
			if v.Kind == metadata.KindInt || v.Kind == metadata.KindFloat {
				var nv float64
				if v.Kind == metadata.KindInt {
					nv = float64(v.I64)
				} else {
					nv = v.F64
				}
				// If ANY value could match, can't prune
				if !stats.CanPruneNumeric(f.Key, "eq", nv) {
					allPruned = false
					break
				}
			} else {
				// Non-numeric value in IN list - can't prune
				allPruned = false
				break
			}
		}
		if allPruned {
			return true
		}
	}

	return false
}
