// Package engine provides the core database engine.
package engine

import (
	"sync"
	"sync/atomic"
	"time"

	"github.com/hupe1980/vecgo/model"
)

// SegmentFeedback tracks query execution metrics for a segment.
// This enables query-feedback-driven segment evolution where
// poorly performing segments are prioritized for compaction/reshaping.
//
// All counters are atomic for lock-free updates during query execution.
// Periodically, the engine can read these metrics to identify
// segments that should be repacked or split.
type SegmentFeedback struct {
	SegmentID model.SegmentID

	// Query counters
	QueriesTotal  atomic.Uint64 // Total queries that touched this segment
	QueriesPruned atomic.Uint64 // Queries that pruned this segment entirely

	// Filter effectiveness
	FilterHits         atomic.Uint64 // Rows that passed filter
	FilterMisses       atomic.Uint64 // Rows that failed filter
	BitmapDensitySum   atomic.Uint64 // Sum of (bitmap_cardinality / segment_size) * 1000
	BitmapDensityCount atomic.Uint64 // Count for averaging

	// Search efficiency
	RowsVisited            atomic.Uint64 // Total rows visited during search
	RowsReturned           atomic.Uint64 // Total rows returned from this segment
	HNSWExpansionsRejected atomic.Uint64 // HNSW expansions rejected by filter

	// Distance computations
	DistanceOps           atomic.Uint64 // Total distance computations
	DistanceShortCircuits atomic.Uint64 // Distance computations short-circuited

	// Timing (microseconds)
	TotalSearchTimeMicros atomic.Uint64
	SearchCount           atomic.Uint64

	// Last update time (for decay/cleanup)
	LastQueryTime atomic.Int64 // Unix nano
}

// NewSegmentFeedback creates a new feedback tracker for a segment.
func NewSegmentFeedback(segmentID model.SegmentID) *SegmentFeedback {
	return &SegmentFeedback{
		SegmentID: segmentID,
	}
}

// RecordQuery records a query that touched this segment.
func (sf *SegmentFeedback) RecordQuery(pruned bool) {
	sf.QueriesTotal.Add(1)
	if pruned {
		sf.QueriesPruned.Add(1)
	}
	sf.LastQueryTime.Store(time.Now().UnixNano())
}

// RecordFilter records filter effectiveness metrics.
func (sf *SegmentFeedback) RecordFilter(hits, misses uint64, bitmapDensity float64) {
	sf.FilterHits.Add(hits)
	sf.FilterMisses.Add(misses)
	// Store density as integer (multiply by 1000 for precision)
	sf.BitmapDensitySum.Add(uint64(bitmapDensity * 1000))
	sf.BitmapDensityCount.Add(1)
}

// RecordSearch records search efficiency metrics.
func (sf *SegmentFeedback) RecordSearch(visited, returned, hnsWRejected uint64, durationMicros int64) {
	sf.RowsVisited.Add(visited)
	sf.RowsReturned.Add(returned)
	sf.HNSWExpansionsRejected.Add(hnsWRejected)
	sf.TotalSearchTimeMicros.Add(uint64(durationMicros))
	sf.SearchCount.Add(1)
}

// RecordDistance records distance computation metrics.
func (sf *SegmentFeedback) RecordDistance(ops, shortCircuits uint64) {
	sf.DistanceOps.Add(ops)
	sf.DistanceShortCircuits.Add(shortCircuits)
}

// FilterPassRate returns the filter pass rate (0.0-1.0).
// Returns 0.5 if no data available.
func (sf *SegmentFeedback) FilterPassRate() float64 {
	hits := sf.FilterHits.Load()
	misses := sf.FilterMisses.Load()
	total := hits + misses
	if total == 0 {
		return 0.5 // Unknown
	}
	return float64(hits) / float64(total)
}

// AverageBitmapDensity returns the average bitmap density (0.0-1.0).
// Higher density means more rows pass filter (less selective).
func (sf *SegmentFeedback) AverageBitmapDensity() float64 {
	count := sf.BitmapDensityCount.Load()
	if count == 0 {
		return 0.5 // Unknown
	}
	return float64(sf.BitmapDensitySum.Load()) / float64(count) / 1000.0
}

// VisitedReturnedRatio returns the ratio of visited/returned rows.
// Higher ratio means more wasted work (worse segment for this query pattern).
func (sf *SegmentFeedback) VisitedReturnedRatio() float64 {
	returned := sf.RowsReturned.Load()
	if returned == 0 {
		return 1.0 // Avoid division by zero
	}
	return float64(sf.RowsVisited.Load()) / float64(returned)
}

// AverageSearchTimeMicros returns the average search time in microseconds.
func (sf *SegmentFeedback) AverageSearchTimeMicros() float64 {
	count := sf.SearchCount.Load()
	if count == 0 {
		return 0
	}
	return float64(sf.TotalSearchTimeMicros.Load()) / float64(count)
}

// PruneRate returns the rate at which this segment is pruned (0.0-1.0).
// High prune rate = good (segment is efficiently skipped).
func (sf *SegmentFeedback) PruneRate() float64 {
	total := sf.QueriesTotal.Load()
	if total == 0 {
		return 0
	}
	return float64(sf.QueriesPruned.Load()) / float64(total)
}

// EfficiencyScore returns an overall efficiency score (0.0-1.0).
// Higher is better. Used for identifying segments to repack.
// Factors:
//   - Filter pass rate (higher = less selective = potentially worse)
//   - Visited/returned ratio (lower = more efficient)
//   - HNSW rejection rate (lower = better graph structure)
func (sf *SegmentFeedback) EfficiencyScore() float64 {
	// Start with perfect score
	score := 1.0

	// Penalize low filter selectivity (high pass rate with visited/returned imbalance)
	filterRate := sf.FilterPassRate()
	visitRatio := sf.VisitedReturnedRatio()
	if visitRatio > 10 {
		score -= 0.3 // Severe inefficiency
	} else if visitRatio > 5 {
		score -= 0.15
	}

	// Penalize high HNSW rejection rate
	distOps := sf.DistanceOps.Load()
	if distOps > 0 {
		rejectionRate := float64(sf.HNSWExpansionsRejected.Load()) / float64(distOps)
		if rejectionRate > 0.5 {
			score -= 0.2 // Half of expansions rejected = poor graph structure for filters
		}
	}

	// Bonus for high prune rate (good segment pruning)
	pruneRate := sf.PruneRate()
	score += pruneRate * 0.1

	// Bonus for high short-circuit rate (efficient distance computation)
	if distOps > 0 {
		scRate := float64(sf.DistanceShortCircuits.Load()) / float64(distOps)
		score += scRate * 0.1
	}

	// Apply filter rate penalty only if there's significant visited/returned imbalance
	if filterRate > 0.8 && visitRatio > 3 {
		score -= 0.1 // Non-selective filter causing extra work
	}

	// Clamp to [0, 1]
	if score < 0 {
		score = 0
	}
	if score > 1 {
		score = 1
	}
	return score
}

// Reset clears all counters.
func (sf *SegmentFeedback) Reset() {
	sf.QueriesTotal.Store(0)
	sf.QueriesPruned.Store(0)
	sf.FilterHits.Store(0)
	sf.FilterMisses.Store(0)
	sf.BitmapDensitySum.Store(0)
	sf.BitmapDensityCount.Store(0)
	sf.RowsVisited.Store(0)
	sf.RowsReturned.Store(0)
	sf.HNSWExpansionsRejected.Store(0)
	sf.DistanceOps.Store(0)
	sf.DistanceShortCircuits.Store(0)
	sf.TotalSearchTimeMicros.Store(0)
	sf.SearchCount.Store(0)
}

// FeedbackStore manages feedback for all segments.
// Thread-safe for concurrent query execution.
type FeedbackStore struct {
	mu       sync.RWMutex
	segments map[model.SegmentID]*SegmentFeedback
}

// NewFeedbackStore creates a new feedback store.
func NewFeedbackStore() *FeedbackStore {
	return &FeedbackStore{
		segments: make(map[model.SegmentID]*SegmentFeedback),
	}
}

// Get returns the feedback tracker for a segment, creating if needed.
func (fs *FeedbackStore) Get(segmentID model.SegmentID) *SegmentFeedback {
	fs.mu.RLock()
	fb, ok := fs.segments[segmentID]
	fs.mu.RUnlock()
	if ok {
		return fb
	}

	fs.mu.Lock()
	defer fs.mu.Unlock()
	// Double-check after acquiring write lock
	if fb, ok = fs.segments[segmentID]; ok {
		return fb
	}
	fb = NewSegmentFeedback(segmentID)
	fs.segments[segmentID] = fb
	return fb
}

// Remove removes feedback for a segment (call after segment is deleted/compacted).
func (fs *FeedbackStore) Remove(segmentID model.SegmentID) {
	fs.mu.Lock()
	delete(fs.segments, segmentID)
	fs.mu.Unlock()
}

// GetWorstSegments returns the N segments with lowest efficiency scores.
// These are candidates for repacking/compaction.
func (fs *FeedbackStore) GetWorstSegments(n int, minQueries uint64) []model.SegmentID {
	fs.mu.RLock()
	defer fs.mu.RUnlock()

	type scored struct {
		id    model.SegmentID
		score float64
	}
	candidates := make([]scored, 0, len(fs.segments))

	for id, fb := range fs.segments {
		// Skip segments with insufficient query history
		if fb.QueriesTotal.Load() < minQueries {
			continue
		}
		candidates = append(candidates, scored{id, fb.EfficiencyScore()})
	}

	// Sort by score ascending (worst first)
	for i := 0; i < len(candidates)-1; i++ {
		for j := i + 1; j < len(candidates); j++ {
			if candidates[j].score < candidates[i].score {
				candidates[i], candidates[j] = candidates[j], candidates[i]
			}
		}
	}

	// Return top N
	result := make([]model.SegmentID, 0, n)
	for i := 0; i < len(candidates) && i < n; i++ {
		result = append(result, candidates[i].id)
	}
	return result
}

// Snapshot returns a snapshot of all segment feedback for analysis.
func (fs *FeedbackStore) Snapshot() map[model.SegmentID]SegmentFeedbackSnapshot {
	fs.mu.RLock()
	defer fs.mu.RUnlock()

	result := make(map[model.SegmentID]SegmentFeedbackSnapshot, len(fs.segments))
	for id, fb := range fs.segments {
		result[id] = SegmentFeedbackSnapshot{
			SegmentID:            id,
			QueriesTotal:         fb.QueriesTotal.Load(),
			QueriesPruned:        fb.QueriesPruned.Load(),
			FilterPassRate:       fb.FilterPassRate(),
			AvgBitmapDensity:     fb.AverageBitmapDensity(),
			VisitedReturnedRatio: fb.VisitedReturnedRatio(),
			AvgSearchTimeMicros:  fb.AverageSearchTimeMicros(),
			EfficiencyScore:      fb.EfficiencyScore(),
			PruneRate:            fb.PruneRate(),
		}
	}
	return result
}

// SegmentFeedbackSnapshot is a point-in-time snapshot of segment feedback.
type SegmentFeedbackSnapshot struct {
	SegmentID            model.SegmentID
	QueriesTotal         uint64
	QueriesPruned        uint64
	FilterPassRate       float64
	AvgBitmapDensity     float64
	VisitedReturnedRatio float64
	AvgSearchTimeMicros  float64
	EfficiencyScore      float64
	PruneRate            float64
}
