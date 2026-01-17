package manifest

import (
	"math"
)

// SegmentStats contains segment-level statistics for query pruning and optimization.
// These stats enable skipping entire segments before opening them, adaptive execution
// strategies, and self-optimizing query planning.
//
// Size target: <2KB per segment for cache-friendliness.
//
// Design principles:
// - Immutable: computed at flush/compaction time (except runtime feedback)
// - Cheap to check: O(1) pruning decisions
// - Small footprint: fits in manifest cache
// - Zero allocations at query time
// - Self-optimizing: runtime feedback improves estimates
type SegmentStats struct {
	// === Core Counts (critical for all estimates) ===

	// TotalRows is the total number of rows in the segment (including deleted).
	TotalRows uint32 `json:"total_rows"`

	// LiveRows is the number of live (non-deleted) rows.
	// LiveRows <= TotalRows always.
	LiveRows uint32 `json:"live_rows"`

	// DeletedRatio = (TotalRows - LiveRows) / TotalRows.
	// High ratio (>0.3) suggests segment needs compaction.
	DeletedRatio float32 `json:"deleted_ratio,omitempty"`

	// === Numeric Field Statistics ===

	// Numeric field min/max for range pruning.
	// Key: field name, Value: min/max bounds + histogram.
	// Enables: WHERE price > 100 → skip if MaxPrice < 100.
	Numeric map[string]NumericFieldStats `json:"numeric,omitempty"`

	// === Categorical Field Statistics ===

	// Categorical field stats for selectivity estimation.
	// Key: field name, Value: distinct count + top-k values + purity.
	// Enables: filter ordering, cardinality estimation, purity-based skip.
	Categorical map[string]CategoricalStats `json:"categorical,omitempty"`

	// === Field Existence ===

	// HasFields maps field names to existence flags.
	// True if field exists in at least one row.
	HasFields map[string]bool `json:"has_fields,omitempty"`

	// === Vector Statistics ===

	// Vector statistics for distance pruning and segment ordering.
	Vector *VectorStats `json:"vector,omitempty"`

	// === Segment Shape Statistics ===

	// Shape describes segment structure for query optimization.
	Shape *ShapeStats `json:"shape,omitempty"`

	// === Filter Entropy Score ===

	// FilterEntropy measures overall filter randomness (0=pure, 1=random).
	// Computed as weighted average of field entropies.
	// Low entropy segments are preferred for early termination.
	FilterEntropy float32 `json:"filter_entropy,omitempty"`

	// === Runtime Feedback (not persisted, populated at load time) ===

	// Runtime holds execution feedback that improves with queries.
	// This is NOT persisted - it's computed from FeedbackStore.
	// Nil until first query feedback is recorded.
	Runtime *RuntimeStats `json:"-"`
}

// HistogramBins is the number of histogram bins.
const HistogramBins = 16

// NumericFieldStats stores min/max bounds and distribution for a numeric field.
// Enables entire-segment pruning for range queries.
type NumericFieldStats struct {
	Min    float64 `json:"min"`
	Max    float64 `json:"max"`
	HasNaN bool    `json:"has_nan,omitempty"`

	// Histogram stores distribution using log-scaled bins (16 bins).
	// More bins at dense regions (typically lower values for prices/counts).
	// Enables better selectivity estimation than equal-width bins.
	//
	// Bin boundaries are computed as:
	//   For i in [0, 15]:
	//     boundary[i] = Min + (Max-Min) * (log(1+i) / log(17))
	// This gives denser coverage at the low end.
	Histogram [HistogramBins]uint32 `json:"histogram,omitempty"`

	// HistogramMin/Max track actual min/max within each bin.
	// Enables tighter BETWEEN estimates.
	HistogramMin [HistogramBins]float64 `json:"hist_min,omitempty"`
	HistogramMax [HistogramBins]float64 `json:"hist_max,omitempty"`

	// Sum and SumSq for variance computation.
	Sum   float64 `json:"sum,omitempty"`
	SumSq float64 `json:"sum_sq,omitempty"`
	Count uint32  `json:"count,omitempty"`
}

// Variance returns the variance of the field values.
func (n *NumericFieldStats) Variance() float64 {
	if n.Count < 2 {
		return 0
	}
	mean := n.Sum / float64(n.Count)
	return (n.SumSq / float64(n.Count)) - (mean * mean)
}

// StdDev returns the standard deviation of the field values.
func (n *NumericFieldStats) StdDev() float64 {
	return math.Sqrt(n.Variance())
}

// CategoricalStats stores cardinality, purity, and top-k values for selectivity estimation.
type CategoricalStats struct {
	// DistinctCount is the number of distinct values in the segment.
	DistinctCount uint32 `json:"distinct_count"`

	// TopK stores the most frequent values (up to 16).
	// Used for selectivity estimation without full scan.
	TopK []ValueFreq `json:"top_k,omitempty"`

	// Purity: dominant value and its ratio (0.0-1.0).
	// If DominantRatio > 0.8, segment is "pure" for this value.
	// Enables: skip bitmap entirely if filter == dominant value.
	DominantValue string  `json:"dominant_value,omitempty"`
	DominantRatio float32 `json:"dominant_ratio,omitempty"` // count/totalRows

	// Entropy measures randomness (0=pure, log2(distinctCount)=uniform).
	// Normalized to [0,1] by dividing by log2(distinctCount).
	// Low entropy = good for pruning.
	Entropy float32 `json:"entropy,omitempty"`

	// Bloom filter for fast negative lookups (not serialized in JSON).
	// If Bloom says "definitely not here", we can skip the segment.
	// ~1% false positive rate, 10 bits per value.
	// Set to nil for low-cardinality fields where TopK covers all values.
	Bloom *BloomFilter `json:"-"`
}

// ValueFreq stores a value and its frequency.
type ValueFreq struct {
	Value string `json:"value"`
	Count uint32 `json:"count"`
}

// VectorStats stores vector-level statistics for distance pruning.
type VectorStats struct {
	// Norm statistics for potential distance bounds.
	MinNorm  float32 `json:"min_norm"`
	MaxNorm  float32 `json:"max_norm"`
	MeanNorm float32 `json:"mean_norm"`

	// === Distance-to-centroid statistics ===

	// AvgDistanceToCentroid is the mean L2 distance from each vector to the centroid.
	// Enables: segment ordering (search tight clusters first).
	AvgDistanceToCentroid float32 `json:"avg_dist_centroid,omitempty"`

	// Radius95 is the 95th percentile distance to centroid.
	// ~95% of vectors are within this radius.
	// Enables: early pruning if query is far from centroid.
	Radius95 float32 `json:"radius_95,omitempty"`

	// RadiusMax is the maximum distance to centroid.
	// Useful for absolute distance bounds.
	RadiusMax float32 `json:"radius_max,omitempty"`

	// Optional centroid for segment-level distance estimation.
	// Stored quantized (int8) to save space. Empty if D > 256.
	Centroid []int8 `json:"centroid,omitempty"`
}

// ShapeStats describes segment structure for query optimization.
type ShapeStats struct {
	// IsSortedByTimestamp indicates if rows are ordered by a timestamp field.
	// Enables: efficient range queries on time, streaming reads.
	IsSortedByTimestamp bool `json:"sorted_ts,omitempty"`

	// TimestampField is the name of the timestamp field (if sorted).
	TimestampField string `json:"ts_field,omitempty"`

	// IsAppendOnly indicates if segment was created via append-only writes.
	// No deletes or updates occurred.
	IsAppendOnly bool `json:"append_only,omitempty"`

	// IsClustered indicates if vectors are clustered (low intra-segment variance).
	// Computed from vector stats: StdDev(distance_to_centroid) < threshold.
	IsClustered bool `json:"clustered,omitempty"`

	// ClusterTightness measures how clustered vectors are (0=loose, 1=tight).
	// = 1 - (StdDev(dist_to_centroid) / AvgDist)
	ClusterTightness float32 `json:"cluster_tightness,omitempty"`
}

// RuntimeStats holds execution feedback that improves with queries.
// Not persisted - recomputed from FeedbackStore on load.
type RuntimeStats struct {
	// ObservedSelectivity maps field names to observed selectivity (0-1).
	// Updated via EMA from actual query results.
	// nil = no observations yet, use static estimates.
	ObservedSelectivity map[string]float32

	// PruneAccuracy measures how often pruning decisions are correct.
	// = (correct_prunes + correct_non_prunes) / total_decisions
	// Used to weight static vs observed estimates.
	PruneAccuracy float32

	// HotField is the most frequently filtered field.
	// Used for adaptive index building.
	HotField string

	// QueryCount is total queries touching this segment.
	QueryCount uint64
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
// Uses histogram if available, otherwise uniform distribution assumption.
func (ss *SegmentStats) EstimateSelectivity(field string, lo, hi float64) float64 {
	if ss == nil || ss.Numeric == nil {
		return 1.0 // Conservative: assume all match
	}

	stats, ok := ss.Numeric[field]
	if !ok {
		return 0.0 // Field missing = no match
	}

	// Prefer histogram-based estimate if available
	if histSel := ss.GetHistogramSelectivity(field, lo, hi); histSel >= 0 {
		return histSel
	}

	// Fallback to uniform distribution assumption
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

	return (effectiveHi - effectiveLo) / segmentRange
}

// EstimateCategoricalSelectivity estimates selectivity for a categorical equality query.
// Uses top-k values if available, otherwise uses 1/distinctCount heuristic.
// If runtime feedback is available, blends static and observed estimates.
func (ss *SegmentStats) EstimateCategoricalSelectivity(field string, value string) float64 {
	if ss == nil || ss.Categorical == nil {
		return 1.0
	}

	stats, ok := ss.Categorical[field]
	if !ok {
		return 0.0 // Field missing = no match
	}

	// Check runtime feedback first
	if ss.Runtime != nil && ss.Runtime.ObservedSelectivity != nil {
		if observed, ok := ss.Runtime.ObservedSelectivity[field]; ok {
			// Blend observed with static estimate based on accuracy
			staticEst := ss.staticCategoricalSelectivity(stats, value)
			weight := ss.Runtime.PruneAccuracy
			if weight < 0.5 {
				weight = 0.5 // Minimum 50% weight to static
			}
			return float64(observed)*(1-float64(weight)) + staticEst*float64(weight)
		}
	}

	return ss.staticCategoricalSelectivity(stats, value)
}

// staticCategoricalSelectivity computes selectivity without runtime feedback.
func (ss *SegmentStats) staticCategoricalSelectivity(stats CategoricalStats, value string) float64 {
	// Check top-k for exact frequency
	for _, vf := range stats.TopK {
		if vf.Value == value {
			if ss.LiveRows > 0 {
				return float64(vf.Count) / float64(ss.LiveRows)
			}
			// Fallback if LiveRows not set
			return float64(vf.Count) / float64(stats.DistinctCount*100)
		}
	}

	// Value not in top-k: use 1/distinctCount heuristic
	if stats.DistinctCount > 0 {
		return 1.0 / float64(stats.DistinctCount)
	}

	return 1.0
}

// IsPure returns true if the segment is "pure" for the given categorical field.
// A pure segment has > PurityThreshold of rows with the same value.
// Pure segments can skip bitmap operations entirely.
func (ss *SegmentStats) IsPure(field string, threshold float32) (string, bool) {
	if ss == nil || ss.Categorical == nil {
		return "", false
	}
	stats, ok := ss.Categorical[field]
	if !ok {
		return "", false
	}
	if stats.DominantRatio >= threshold {
		return stats.DominantValue, true
	}
	return "", false
}

// CanPruneCategorical checks if a segment can be entirely skipped for a categorical equality query.
// Returns true if:
// - Field doesn't exist in segment
// - Segment is pure with a different dominant value
// - Bloom filter says "definitely not present" (no false negatives)
// - Value is not in top-k and top-k covers all values
func (ss *SegmentStats) CanPruneCategorical(field, value string) bool {
	if ss == nil || ss.Categorical == nil {
		return false
	}
	stats, ok := ss.Categorical[field]
	if !ok {
		return true // Field missing = no match
	}

	// Fast path: If segment is pure with different value, skip entirely
	if stats.DominantRatio > 0.99 && stats.DominantValue != value {
		return true
	}

	// Bloom filter check: O(k) where k ≈ 7
	// If Bloom says "definitely not", we can skip (zero false negatives)
	if stats.Bloom != nil && !stats.Bloom.MayContain(value) {
		return true // Bloom says definitely not present
	}

	// TopK completeness check: if TopK covers all values and value not in TopK
	if int(stats.DistinctCount) <= len(stats.TopK) {
		// TopK contains all distinct values
		for _, vf := range stats.TopK {
			if vf.Value == value {
				return false // Value found in TopK
			}
		}
		return true // Value not in TopK and TopK is complete
	}

	return false
}

// GetHistogramSelectivity estimates selectivity using log-scaled histogram bins.
// Returns fraction of values in range [lo, hi], or -1 if histogram unavailable.
func (ss *SegmentStats) GetHistogramSelectivity(field string, lo, hi float64) float64 {
	if ss == nil || ss.Numeric == nil {
		return -1
	}
	stats, ok := ss.Numeric[field]
	if !ok {
		return 0.0
	}

	// Check if histogram is populated
	var totalCount uint32
	for _, c := range stats.Histogram {
		totalCount += c
	}
	if totalCount == 0 {
		return -1 // Histogram not available
	}

	segmentRange := stats.Max - stats.Min
	if segmentRange <= 0 {
		if lo <= stats.Min && stats.Max <= hi {
			return 1.0
		}
		return 0.0
	}

	// Compute log-scaled bin boundaries
	binBoundaries := make([]float64, HistogramBins+1)
	binBoundaries[0] = stats.Min
	for i := 1; i <= HistogramBins; i++ {
		// Log-scaled: boundary[i] = Min + (Max-Min) * (log(1+i) / log(17))
		t := math.Log(1+float64(i)) / math.Log(17)
		binBoundaries[i] = stats.Min + segmentRange*t
	}

	// Count values in range
	var inRangeCount uint32
	for i, count := range stats.Histogram {
		binLo := binBoundaries[i]
		binHi := binBoundaries[i+1]
		binWidth := binHi - binLo

		// Check overlap with query range
		if binHi <= lo || binLo >= hi {
			continue // No overlap
		}

		// Use per-bin min/max for tighter estimates if available
		actualBinLo := binLo
		actualBinHi := binHi
		if stats.HistogramMin[i] != 0 || stats.HistogramMax[i] != 0 {
			actualBinLo = stats.HistogramMin[i]
			actualBinHi = stats.HistogramMax[i]
		}

		// Full or partial overlap
		if actualBinLo >= lo && actualBinHi <= hi {
			// Full overlap
			inRangeCount += count
		} else if binWidth > 0 {
			// Partial overlap - proportional estimate
			overlapLo := math.Max(actualBinLo, lo)
			overlapHi := math.Min(actualBinHi, hi)
			actualBinWidth := actualBinHi - actualBinLo
			if actualBinWidth <= 0 {
				actualBinWidth = binWidth
			}
			overlapRatio := (overlapHi - overlapLo) / actualBinWidth
			if overlapRatio > 0 {
				inRangeCount += uint32(float64(count) * overlapRatio)
			}
		}
	}

	return float64(inRangeCount) / float64(totalCount)
}

// ShouldUseBitmap returns true if bitmap-based filtering is recommended.
// Based on cardinality, purity, and row count heuristics.
// Returns false if segment is pure enough for scan-only approach.
func (ss *SegmentStats) ShouldUseBitmap(field, value string, rowCount uint32) bool {
	if ss == nil || ss.Categorical == nil {
		return true // Default to bitmap
	}

	// Use LiveRows if available
	if ss.LiveRows > 0 {
		rowCount = ss.LiveRows
	}

	stats, ok := ss.Categorical[field]
	if !ok {
		return false // Field missing, no bitmap needed
	}

	// Pure segment: scan-only is faster
	if stats.DominantRatio > 0.8 && stats.DominantValue == value {
		return false
	}

	// Very low cardinality: scan may be faster
	if stats.DistinctCount <= 2 {
		return false
	}

	// Estimate match count
	var expectedMatches float64
	if stats.DominantValue == value {
		expectedMatches = float64(rowCount) * float64(stats.DominantRatio)
	} else {
		// Assume uniform distribution for non-dominant values
		expectedMatches = float64(rowCount) / float64(stats.DistinctCount)
	}

	// Bitmap is efficient when selectivity is low (few matches)
	// Scan is efficient when selectivity is high (many matches)
	return expectedMatches < float64(rowCount)*0.5
}

// NeedsCompaction returns true if the segment should be prioritized for compaction.
// Based on deleted ratio and efficiency metrics.
func (ss *SegmentStats) NeedsCompaction(deletedThreshold float32) bool {
	if ss == nil {
		return false
	}
	return ss.DeletedRatio > deletedThreshold
}

// IsLowEntropy returns true if the segment has low filter entropy.
// Low entropy segments are better for early termination.
func (ss *SegmentStats) IsLowEntropy(threshold float32) bool {
	if ss == nil {
		return false
	}
	return ss.FilterEntropy < threshold
}

// CanPruneByDistance checks if a segment can be pruned based on query distance to centroid.
// Returns true if the minimum possible distance exceeds maxDistance.
//
// Uses triangle inequality: dist(query, any_point) >= dist(query, centroid) - Radius95
func (ss *SegmentStats) CanPruneByDistance(queryCentroidDist, maxDistance float32) bool {
	if ss == nil || ss.Vector == nil || ss.Vector.Radius95 == 0 {
		return false
	}

	// Triangle inequality lower bound
	minPossibleDist := queryCentroidDist - ss.Vector.Radius95
	if minPossibleDist < 0 {
		minPossibleDist = 0
	}

	return minPossibleDist > maxDistance
}

// SegmentPriority computes a priority score for segment traversal order.
// Higher score = search this segment first.
// Considers: cluster tightness, distance to query centroid, entropy.
func (ss *SegmentStats) SegmentPriority(queryCentroidDist float32) float32 {
	if ss == nil {
		return 0
	}

	var score float32 = 1.0

	// Prefer tight clusters (vectors close to centroid)
	if ss.Shape != nil && ss.Shape.ClusterTightness > 0 {
		score += ss.Shape.ClusterTightness * 0.3
	}

	// Prefer low entropy segments
	if ss.FilterEntropy < 0.5 {
		score += (0.5 - ss.FilterEntropy) * 0.2
	}

	// Prefer segments closer to query
	if ss.Vector != nil && ss.Vector.AvgDistanceToCentroid > 0 {
		// Normalize by expected distance
		normalizedDist := queryCentroidDist / ss.Vector.AvgDistanceToCentroid
		if normalizedDist < 2.0 {
			score += (2.0 - normalizedDist) * 0.3
		}
	}

	// Penalize tombstone-heavy segments
	if ss.DeletedRatio > 0.1 {
		score -= ss.DeletedRatio * 0.2
	}

	return score
}

// NewSegmentStats creates an empty SegmentStats.
func NewSegmentStats() *SegmentStats {
	return &SegmentStats{
		Numeric:     make(map[string]NumericFieldStats),
		Categorical: make(map[string]CategoricalStats),
		HasFields:   make(map[string]bool),
	}
}
