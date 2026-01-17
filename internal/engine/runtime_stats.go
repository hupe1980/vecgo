package engine

import (
	"sync"

	"github.com/hupe1980/vecgo/internal/manifest"
	"github.com/hupe1980/vecgo/model"
)

// RuntimeStatsManager bridges FeedbackStore to SegmentStats.Runtime.
// It computes and updates runtime statistics that improve selectivity estimates
// and pruning accuracy based on actual query execution feedback.
//
// Thread-safety: All methods are safe for concurrent use.
type RuntimeStatsManager struct {
	mu       sync.RWMutex
	feedback *FeedbackStore

	// Observed selectivity EMA decay factor (0.1 = slow adapt, 0.5 = fast adapt)
	emaAlpha float32

	// Per-field selectivity observations
	// Key: segmentID -> field -> observed selectivity
	observed map[model.SegmentID]map[string]*selectivityTracker

	// Per-field query counts (for hot field detection)
	fieldQueryCounts map[model.SegmentID]map[string]uint64
}

// selectivityTracker tracks EMA of observed selectivity for a field.
type selectivityTracker struct {
	ema   float32 // Exponential moving average
	count uint64  // Number of observations
}

// NewRuntimeStatsManager creates a new runtime stats manager.
func NewRuntimeStatsManager(feedback *FeedbackStore) *RuntimeStatsManager {
	return &RuntimeStatsManager{
		feedback:         feedback,
		emaAlpha:         0.3, // Balance between stability and responsiveness
		observed:         make(map[model.SegmentID]map[string]*selectivityTracker),
		fieldQueryCounts: make(map[model.SegmentID]map[string]uint64),
	}
}

// RecordSelectivity records an observed selectivity for a field in a segment.
// Selectivity = (matching rows) / (total rows checked).
func (rsm *RuntimeStatsManager) RecordSelectivity(segmentID model.SegmentID, field string, selectivity float32) {
	rsm.mu.Lock()
	defer rsm.mu.Unlock()

	segFields := rsm.observed[segmentID]
	if segFields == nil {
		segFields = make(map[string]*selectivityTracker)
		rsm.observed[segmentID] = segFields
	}

	tracker := segFields[field]
	if tracker == nil {
		tracker = &selectivityTracker{ema: selectivity}
		segFields[field] = tracker
	} else {
		// EMA update: new_ema = alpha * observation + (1-alpha) * old_ema
		tracker.ema = rsm.emaAlpha*selectivity + (1-rsm.emaAlpha)*tracker.ema
	}
	tracker.count++

	// Track field query counts
	fieldCounts := rsm.fieldQueryCounts[segmentID]
	if fieldCounts == nil {
		fieldCounts = make(map[string]uint64)
		rsm.fieldQueryCounts[segmentID] = fieldCounts
	}
	fieldCounts[field]++
}

// RecordPruneAccuracy records whether a pruning decision was accurate.
// correct = true if the segment was correctly pruned (no matching results)
// or correctly NOT pruned (had matching results).
func (rsm *RuntimeStatsManager) RecordPruneAccuracy(_ model.SegmentID, _ bool) {
	// Prune accuracy is tracked implicitly via PruneRate and filter hit/miss ratios
	// in the FeedbackStore. This method is a hook for future explicit tracking.
}

// GetRuntimeStats computes RuntimeStats for a segment based on feedback.
// Returns nil if insufficient data is available.
func (rsm *RuntimeStatsManager) GetRuntimeStats(segmentID model.SegmentID) *manifest.RuntimeStats {
	rsm.mu.RLock()
	defer rsm.mu.RUnlock()

	fb := rsm.feedback.Get(segmentID)
	queryCount := fb.QueriesTotal.Load()

	// Need at least 10 queries for meaningful stats
	if queryCount < 10 {
		return nil
	}

	rt := &manifest.RuntimeStats{
		QueryCount: queryCount,
	}

	// Collect observed selectivities
	segFields := rsm.observed[segmentID]
	if len(segFields) > 0 {
		rt.ObservedSelectivity = make(map[string]float32, len(segFields))
		for field, tracker := range segFields {
			if tracker.count >= 5 { // Need at least 5 observations
				rt.ObservedSelectivity[field] = tracker.ema
			}
		}
	}

	// Compute prune accuracy from feedback
	// If prune rate is high and efficiency is good, pruning is accurate
	pruneRate := fb.PruneRate()
	efficiency := fb.EfficiencyScore()
	rt.PruneAccuracy = float32((pruneRate + efficiency) / 2)

	// Find hot field (most frequently filtered)
	fieldCounts := rsm.fieldQueryCounts[segmentID]
	if fieldCounts != nil {
		var hotField string
		var maxCount uint64
		for field, count := range fieldCounts {
			if count > maxCount {
				maxCount = count
				hotField = field
			}
		}
		if maxCount >= 10 {
			rt.HotField = hotField
		}
	}

	return rt
}

// AttachRuntimeStats attaches computed runtime stats to a SegmentStats.
// Call this when loading segments or before executing queries.
func (rsm *RuntimeStatsManager) AttachRuntimeStats(segmentID model.SegmentID, stats *manifest.SegmentStats) {
	if stats == nil {
		return
	}
	stats.Runtime = rsm.GetRuntimeStats(segmentID)
}

// RemoveSegment removes all tracked data for a segment (after compaction/deletion).
func (rsm *RuntimeStatsManager) RemoveSegment(segmentID model.SegmentID) {
	rsm.mu.Lock()
	delete(rsm.observed, segmentID)
	delete(rsm.fieldQueryCounts, segmentID)
	rsm.mu.Unlock()

	rsm.feedback.Remove(segmentID)
}

// SetEMAAlpha sets the EMA decay factor for selectivity tracking.
// Lower values (0.1) = slower adaptation, more stable.
// Higher values (0.5) = faster adaptation, more reactive.
func (rsm *RuntimeStatsManager) SetEMAAlpha(alpha float32) {
	rsm.mu.Lock()
	rsm.emaAlpha = alpha
	rsm.mu.Unlock()
}

// Snapshot returns a snapshot of all runtime stats for analysis.
func (rsm *RuntimeStatsManager) Snapshot() map[model.SegmentID]*manifest.RuntimeStats {
	rsm.mu.RLock()
	defer rsm.mu.RUnlock()

	result := make(map[model.SegmentID]*manifest.RuntimeStats)
	for segID := range rsm.observed {
		rt := rsm.GetRuntimeStats(segID)
		if rt != nil {
			result[segID] = rt
		}
	}
	return result
}

// SegmentSelectivityReport returns a report of selectivity observations for a segment.
func (rsm *RuntimeStatsManager) SegmentSelectivityReport(segmentID model.SegmentID) map[string]SelectivityInfo {
	rsm.mu.RLock()
	defer rsm.mu.RUnlock()

	segFields := rsm.observed[segmentID]
	if segFields == nil {
		return nil
	}

	result := make(map[string]SelectivityInfo, len(segFields))
	for field, tracker := range segFields {
		result[field] = SelectivityInfo{
			Field:            field,
			Selectivity:      tracker.ema,
			ObservationCount: tracker.count,
		}
	}
	return result
}

// SelectivityInfo contains selectivity information for a field.
type SelectivityInfo struct {
	Field            string
	Selectivity      float32
	ObservationCount uint64
}
