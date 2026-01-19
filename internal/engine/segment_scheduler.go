package engine

import (
	"sort"
	"sync/atomic"

	"github.com/hupe1980/vecgo/distance"
	"github.com/hupe1980/vecgo/internal/manifest"
	"github.com/hupe1980/vecgo/model"
)

// SegmentScheduler computes optimal segment traversal order for queries.
// It uses manifest stats and centroid distances to prioritize segments
// most likely to contain top-k results.
//
// Key optimizations:
// - Triangle inequality pruning via centroid distance
// - Cluster tightness ordering (search tight clusters first)
// - Early termination when top segments yield sufficient results
type SegmentScheduler struct {
	// Stats provider for segment statistics
	statsProvider func(model.SegmentID) *manifest.SegmentStats

	// Metric for distance computation
	metric distance.Metric

	// Dimension for centroid comparison
	dim int

	// Track scheduling effectiveness
	totalScheduled  atomic.Uint64
	earlyTerminated atomic.Uint64
}

// SchedulerStats tracks scheduling effectiveness.
type SchedulerStats struct {
	TotalScheduled  uint64
	EarlyTerminated uint64
	TerminationRate float64
	AvgPriorityGain float64 // How much priority ordering helped
}

// NewSegmentScheduler creates a new segment scheduler.
func NewSegmentScheduler(
	statsProvider func(model.SegmentID) *manifest.SegmentStats,
	metric distance.Metric,
	dim int,
) *SegmentScheduler {
	return &SegmentScheduler{
		statsProvider: statsProvider,
		metric:        metric,
		dim:           dim,
	}
}

// SegmentPriority holds a segment with its computed priority score.
type SegmentPriority struct {
	SegmentID        model.SegmentID
	Segment          any     // The actual segment (interface to avoid import cycle)
	Priority         float32 // Higher = search first
	CentroidDistance float32 // Distance from query to segment centroid
	CanPrune         bool    // Whether segment can be completely skipped
	EstimatedHits    uint32  // Estimated matching rows (for filtered queries)
	Stats            *manifest.SegmentStats
}

// ScheduleParams contains parameters for scheduling.
type ScheduleParams struct {
	Query       []float32 // The query vector
	K           int       // Number of results requested
	MaxDistance float32   // Maximum acceptable distance (0 = no limit)
	Filter      any       // Optional filter (for selectivity estimation)
	FilterKeys  []string  // Filter field names for selectivity lookup
}

// Schedule computes priority-ordered segment list for a query.
// Returns segments ordered by decreasing priority (best first).
// Segments that can be pruned are excluded from the result.
func (ss *SegmentScheduler) Schedule(
	segments []SegmentPriority,
	params ScheduleParams,
) []SegmentPriority {
	if len(segments) == 0 {
		return segments
	}

	ss.totalScheduled.Add(1)

	// Phase 1: Compute centroid distances and priorities
	for i := range segments {
		seg := &segments[i]

		// Get stats for this segment
		if ss.statsProvider != nil {
			seg.Stats = ss.statsProvider(seg.SegmentID)
		}

		// Compute centroid distance
		seg.CentroidDistance = ss.computeCentroidDistance(params.Query, seg.Stats)

		// Check if segment can be pruned by distance
		if params.MaxDistance > 0 && seg.Stats != nil {
			seg.CanPrune = seg.Stats.CanPruneByDistance(seg.CentroidDistance, params.MaxDistance)
		}

		// Compute priority score
		seg.Priority = ss.computePriority(seg, params)
	}

	// Phase 2: Filter out prunable segments
	active := segments[:0]
	for _, seg := range segments {
		if !seg.CanPrune {
			active = append(active, seg)
		}
	}

	// Phase 3: Sort by priority (descending - highest first)
	sort.Slice(active, func(i, j int) bool {
		return active[i].Priority > active[j].Priority
	})

	return active
}

// computeCentroidDistance computes distance from query to segment centroid.
func (ss *SegmentScheduler) computeCentroidDistance(query []float32, stats *manifest.SegmentStats) float32 {
	if stats == nil || stats.Vector == nil || len(stats.Vector.Centroid) == 0 {
		return 0 // Unknown - treat as equidistant
	}

	// Dequantize centroid (stored as int8)
	centroid := stats.Vector.Centroid
	if len(centroid) != len(query) && len(centroid) != 0 {
		// Dimension mismatch - can't use centroid
		// This can happen if centroid was truncated for storage
		return 0
	}

	// Fast approximate L2 distance using quantized centroid
	// For int8 quantization: real_value ≈ int8_value * scale
	// We use mean norm as approximate scale
	scale := stats.Vector.MeanNorm / 127.0
	if scale == 0 {
		scale = 1.0
	}

	var sum float32
	for i, c := range centroid {
		if i >= len(query) {
			break
		}
		diff := query[i] - float32(c)*scale
		sum += diff * diff
	}

	return sum // Squared L2
}

// computePriority computes priority score for a segment.
// Higher score = search this segment first.
func (ss *SegmentScheduler) computePriority(seg *SegmentPriority, params ScheduleParams) float32 {
	stats := seg.Stats
	if stats == nil {
		return 1.0 // Default priority for unknown segments
	}

	// Base priority from manifest stats
	priority := stats.SegmentPriority(seg.CentroidDistance)

	// Boost for filter selectivity (if filter provided)
	if params.Filter != nil && len(params.FilterKeys) > 0 {
		selectivityBoost := ss.estimateFilterSelectivity(stats, params.FilterKeys)
		// Higher selectivity (fewer matches) = lower priority (less likely to have results)
		// But also could mean faster search within segment
		// Trade-off: prioritize segments with moderate selectivity
		if selectivityBoost < 0.1 {
			// Very selective filter - might miss this segment entirely
			priority -= 0.2
		} else if selectivityBoost > 0.5 {
			// Low selectivity - many matches expected
			priority += 0.1
		}
	}

	// Penalize tombstone-heavy segments (handled in SegmentPriority already, but reinforce)
	if stats.DeletedRatio > 0.3 {
		priority -= 0.1
	}

	return priority
}

// estimateFilterSelectivity estimates filter selectivity for a segment.
func (ss *SegmentScheduler) estimateFilterSelectivity(
	stats *manifest.SegmentStats,
	filterKeys []string,
) float32 {
	if stats == nil || len(filterKeys) == 0 {
		return 0.5 // Unknown
	}

	// Use static estimates from categorical stats
	for _, key := range filterKeys {
		if cat, ok := stats.Categorical[key]; ok {
			// Use entropy as proxy for selectivity
			// High entropy = uniform distribution = ~1/distinctCount selectivity
			if cat.DistinctCount > 0 {
				return 1.0 / float32(cat.DistinctCount)
			}
		}
	}

	return 0.5 // Unknown
}

// ShouldEarlyTerminate checks if search can terminate early based on results so far.
// This is called after searching the first N segments to see if we have enough
// high-quality results to skip remaining segments.
//
// Parameters:
// - searchedCount: number of segments already searched
// - totalCount: total number of segments
// - resultsFound: number of results found so far
// - k: target number of results
// - worstResultDist: distance of the k-th result (or max if < k results)
// - remainingSegments: segments not yet searched (sorted by priority)
//
// Returns true if remaining segments are unlikely to improve results.
func (ss *SegmentScheduler) ShouldEarlyTerminate(
	searchedCount, totalCount int,
	resultsFound, k int,
	worstResultDist float32,
	remainingSegments []SegmentPriority,
) bool {
	// Must have searched at least some segments
	if searchedCount < 2 || searchedCount < totalCount/4 {
		return false
	}

	// Must have found enough results
	if resultsFound < k {
		return false
	}

	// Check if all remaining segments can be pruned by distance
	canPruneAll := true
	for _, seg := range remainingSegments {
		if !seg.CanPrune {
			// Check triangle inequality
			if seg.Stats != nil && seg.Stats.Vector != nil {
				minDist := seg.CentroidDistance - seg.Stats.Vector.Radius95
				if minDist < 0 {
					minDist = 0
				}
				if minDist < worstResultDist {
					canPruneAll = false
					break
				}
			} else {
				canPruneAll = false
				break
			}
		}
	}

	if canPruneAll {
		ss.earlyTerminated.Add(1)
		return true
	}

	return false
}

// Stats returns scheduling statistics.
func (ss *SegmentScheduler) Stats() SchedulerStats {
	total := ss.totalScheduled.Load()
	terminated := ss.earlyTerminated.Load()

	var termRate float64
	if total > 0 {
		termRate = float64(terminated) / float64(total)
	}

	return SchedulerStats{
		TotalScheduled:  total,
		EarlyTerminated: terminated,
		TerminationRate: termRate,
	}
}

// Reset resets scheduling statistics.
func (ss *SegmentScheduler) Reset() {
	ss.totalScheduled.Store(0)
	ss.earlyTerminated.Store(0)
}
