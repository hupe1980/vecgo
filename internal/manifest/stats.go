package manifest

import (
	"math"
)

// SegmentStats contains segment-level statistics for query pruning.
// These stats enable skipping entire segments before opening them.
// Size target: <2KB per segment for cache-friendliness.
//
// Design principles:
// - Immutable: computed at flush/compaction time
// - Cheap to check: O(1) pruning decisions
// - Small footprint: fits in manifest cache
// - Zero allocations at query time
type SegmentStats struct {
	// Numeric field min/max for range pruning
	// Key: field name, Value: min/max bounds
	// Enables: WHERE price > 100 → skip if MaxPrice < 100
	Numeric map[string]NumericFieldStats `json:"numeric,omitempty"`

	// Categorical field stats for selectivity estimation
	// Key: field name, Value: distinct count + top-k values
	// Enables: filter ordering, cardinality estimation
	Categorical map[string]CategoricalStats `json:"categorical,omitempty"`

	// Boolean existence flags for fast null/column checks
	// True if field exists in at least one row
	HasFields map[string]bool `json:"has_fields,omitempty"`

	// Vector statistics for distance pruning
	Vector *VectorStats `json:"vector,omitempty"`
}

// NumericFieldStats stores min/max bounds for a numeric field.
// Enables entire-segment pruning for range queries.
type NumericFieldStats struct {
	Min    float64 `json:"min"`
	Max    float64 `json:"max"`
	HasNaN bool    `json:"has_nan,omitempty"`
}

// CategoricalStats stores cardinality and top-k values for selectivity estimation.
type CategoricalStats struct {
	// DistinctCount is the number of distinct values in the segment
	DistinctCount uint32 `json:"distinct_count"`
	// TopK stores the most frequent values (up to 16)
	// Used for selectivity estimation without full scan
	TopK []ValueFreq `json:"top_k,omitempty"`
}

// ValueFreq stores a value and its frequency.
type ValueFreq struct {
	Value string `json:"value"`
	Count uint32 `json:"count"`
}

// VectorStats stores vector-level statistics for distance pruning.
type VectorStats struct {
	// Norm statistics for potential distance bounds
	MinNorm  float32 `json:"min_norm"`
	MaxNorm  float32 `json:"max_norm"`
	MeanNorm float32 `json:"mean_norm"`

	// Optional centroid for segment-level distance estimation
	// Stored quantized (int8) to save space. Empty if D > 256.
	Centroid []int8 `json:"centroid,omitempty"`
}

// CanPruneNumeric checks if a segment can be entirely skipped for a numeric range query.
// Returns true if the query range doesn't overlap with the segment's value range.
//
// Query semantics:
// - op == "gt" or "gte": queryVal is the lower bound
// - op == "lt" or "lte": queryVal is the upper bound
// - op == "eq": queryVal must be in [min, max]
// - op == "neq": can only prune if segment has single value equal to queryVal
func (ss *SegmentStats) CanPruneNumeric(field string, op string, queryVal float64) bool {
	if ss == nil || ss.Numeric == nil {
		return false
	}

	stats, ok := ss.Numeric[field]
	if !ok {
		// Field doesn't exist in this segment - depends on query semantics
		// For most queries, missing field means no match
		return true
	}

	switch op {
	case "gt":
		// WHERE field > queryVal
		// Prune if max <= queryVal (no value > queryVal)
		return stats.Max <= queryVal

	case "gte":
		// WHERE field >= queryVal
		// Prune if max < queryVal (no value >= queryVal)
		return stats.Max < queryVal

	case "lt":
		// WHERE field < queryVal
		// Prune if min >= queryVal (no value < queryVal)
		return stats.Min >= queryVal

	case "lte":
		// WHERE field <= queryVal
		// Prune if min > queryVal (no value <= queryVal)
		return stats.Min > queryVal

	case "eq":
		// WHERE field == queryVal
		// Prune if queryVal is outside [min, max]
		return queryVal < stats.Min || queryVal > stats.Max

	case "neq":
		// WHERE field != queryVal
		// Can only prune if entire segment has single value equal to queryVal
		isSingleValue := stats.Min == stats.Max
		return isSingleValue && stats.Min == queryVal

	case "between":
		// For BETWEEN queries, caller should use CanPruneNumericRange
		return false
	}

	return false
}

// CanPruneNumericRange checks if a segment can be pruned for a range query [lo, hi].
// Returns true if segment range doesn't overlap with query range.
func (ss *SegmentStats) CanPruneNumericRange(field string, lo, hi float64, includeLo, includeHi bool) bool {
	if ss == nil || ss.Numeric == nil {
		return false
	}

	stats, ok := ss.Numeric[field]
	if !ok {
		return true // Field missing = no match
	}

	// Check for no overlap
	// Segment range: [stats.Min, stats.Max]
	// Query range: [lo, hi] (with include flags)

	if includeLo {
		// Query: [lo, ...] — no overlap if stats.Max < lo
		if stats.Max < lo {
			return true
		}
	} else {
		// Query: (lo, ...] — no overlap if stats.Max <= lo
		if stats.Max <= lo {
			return true
		}
	}

	if includeHi {
		// Query: [..., hi] — no overlap if stats.Min > hi
		if stats.Min > hi {
			return true
		}
	} else {
		// Query: [..., hi) — no overlap if stats.Min >= hi
		if stats.Min >= hi {
			return true
		}
	}

	return false
}

// HasField checks if a field exists in the segment.
func (ss *SegmentStats) HasField(field string) bool {
	if ss == nil || ss.HasFields == nil {
		return false
	}
	return ss.HasFields[field]
}

// EstimateSelectivity estimates the selectivity of a numeric range query.
// Returns a value in [0, 1] representing the estimated fraction of matching rows.
// Uses uniform distribution assumption within segment bounds.
func (ss *SegmentStats) EstimateSelectivity(field string, lo, hi float64) float64 {
	if ss == nil || ss.Numeric == nil {
		return 1.0 // Conservative: assume all match
	}

	stats, ok := ss.Numeric[field]
	if !ok {
		return 0.0 // Field missing = no match
	}

	segmentRange := stats.Max - stats.Min
	if segmentRange <= 0 {
		// Single value in segment
		if lo <= stats.Min && stats.Max <= hi {
			return 1.0
		}
		return 0.0
	}

	// Clamp query range to segment range
	effectiveLo := math.Max(lo, stats.Min)
	effectiveHi := math.Min(hi, stats.Max)

	if effectiveLo > effectiveHi {
		return 0.0
	}

	// Uniform distribution assumption
	return (effectiveHi - effectiveLo) / segmentRange
}

// EstimateCategoricalSelectivity estimates selectivity for a categorical equality query.
// Uses top-k values if available, otherwise uses 1/distinctCount heuristic.
func (ss *SegmentStats) EstimateCategoricalSelectivity(field string, value string) float64 {
	if ss == nil || ss.Categorical == nil {
		return 1.0
	}

	stats, ok := ss.Categorical[field]
	if !ok {
		return 0.0 // Field missing = no match
	}

	// Check top-k for exact frequency
	for _, vf := range stats.TopK {
		if vf.Value == value {
			// We don't store total row count in CategoricalStats,
			// so we return a relative frequency based on distinct count
			// This is an approximation
			return float64(vf.Count) / float64(stats.DistinctCount*100) // Heuristic
		}
	}

	// Value not in top-k: use 1/distinctCount heuristic
	if stats.DistinctCount > 0 {
		return 1.0 / float64(stats.DistinctCount)
	}

	return 1.0
}

// NewSegmentStats creates an empty SegmentStats.
func NewSegmentStats() *SegmentStats {
	return &SegmentStats{
		Numeric:     make(map[string]NumericFieldStats),
		Categorical: make(map[string]CategoricalStats),
		HasFields:   make(map[string]bool),
	}
}
