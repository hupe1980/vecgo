// Package engine provides the core database engine.
package engine

import (
	"sync"
	"sync/atomic"

	"github.com/hupe1980/vecgo/metadata"
)

// FilterOrderLearner learns the optimal filter evaluation order based on
// observed selectivity. Filters that reduce cardinality most are evaluated first.
//
// This is inspired by database query optimizers but uses simple exponential
// moving averages instead of complex cardinality estimation.
//
// Key insight: The order of filters matters more than the filters themselves.
// Evaluating the most selective filter first can reduce bitmap ops by 2-5x.
type FilterOrderLearner struct {
	mu    sync.RWMutex
	stats map[string]*filterFieldStats // field -> stats

	// Global stats
	totalQueries atomic.Uint64
	reorderings  atomic.Uint64
}

type filterFieldStats struct {
	// Exponential moving average of selectivity (0.0-1.0)
	// Lower = more selective = should be first
	selectivityEMA float64

	// Count of observations
	observations uint64

	// Sum of selectivity values for initial bootstrap
	selectivitySum float64

	// Per-operator stats for more precise ordering
	operatorStats map[metadata.Operator]*operatorSelectivity
}

type operatorSelectivity struct {
	selectivityEMA float64
	observations   uint64
}

// FilterOrderLearnerConfig configures the filter order learner.
type FilterOrderLearnerConfig struct {
	// EMAAlpha is the exponential moving average alpha (0.0-1.0).
	// Higher = more weight on recent observations.
	// Default: 0.1
	EMAAlpha float64

	// MinObservations is the minimum observations before using learned order.
	// Default: 10
	MinObservations int

	// Enabled controls whether filter reordering is active.
	// Default: true
	Enabled bool
}

// DefaultFilterOrderLearnerConfig returns the default configuration.
func DefaultFilterOrderLearnerConfig() FilterOrderLearnerConfig {
	return FilterOrderLearnerConfig{
		EMAAlpha:        0.1,
		MinObservations: 10,
		Enabled:         true,
	}
}

// NewFilterOrderLearner creates a new filter order learner.
func NewFilterOrderLearner() *FilterOrderLearner {
	return &FilterOrderLearner{
		stats: make(map[string]*filterFieldStats),
	}
}

// RecordSelectivity records the observed selectivity for a filter field.
// selectivity is the ratio of rows that passed the filter (0.0-1.0).
// Lower selectivity = more rows filtered out = should be evaluated first.
func (fol *FilterOrderLearner) RecordSelectivity(field string, op metadata.Operator, selectivity float64, alpha float64) {
	if alpha <= 0 || alpha > 1 {
		alpha = 0.1
	}

	fol.mu.Lock()
	defer fol.mu.Unlock()

	stats, ok := fol.stats[field]
	if !ok {
		stats = &filterFieldStats{
			operatorStats: make(map[metadata.Operator]*operatorSelectivity),
		}
		fol.stats[field] = stats
	}

	// Update field-level EMA
	if stats.observations == 0 {
		stats.selectivityEMA = selectivity
	} else {
		stats.selectivityEMA = alpha*selectivity + (1-alpha)*stats.selectivityEMA
	}
	stats.observations++
	stats.selectivitySum += selectivity

	// Update operator-level stats
	opStats, ok := stats.operatorStats[op]
	if !ok {
		opStats = &operatorSelectivity{}
		stats.operatorStats[op] = opStats
	}
	if opStats.observations == 0 {
		opStats.selectivityEMA = selectivity
	} else {
		opStats.selectivityEMA = alpha*selectivity + (1-alpha)*opStats.selectivityEMA
	}
	opStats.observations++

	fol.totalQueries.Add(1)
}

// OptimalOrder returns the optimal filter evaluation order based on learned selectivity.
// Filters are reordered so the most selective (lowest selectivity) comes first.
// Returns a new slice; original is not modified.
func (fol *FilterOrderLearner) OptimalOrder(filters []metadata.Filter, minObs int) []metadata.Filter {
	if len(filters) <= 1 {
		return filters
	}

	fol.mu.RLock()
	defer fol.mu.RUnlock()

	// Build scored list
	type scored struct {
		filter      metadata.Filter
		selectivity float64
		hasData     bool
	}
	scored_filters := make([]scored, len(filters))

	for i, f := range filters {
		scored_filters[i].filter = f
		scored_filters[i].selectivity = 0.5 // Default: unknown

		if stats, ok := fol.stats[f.Key]; ok && stats.observations >= uint64(minObs) {
			// Try operator-specific stats first
			if opStats, ok := stats.operatorStats[f.Operator]; ok && opStats.observations >= uint64(minObs) {
				scored_filters[i].selectivity = opStats.selectivityEMA
				scored_filters[i].hasData = true
			} else {
				// Fall back to field-level stats
				scored_filters[i].selectivity = stats.selectivityEMA
				scored_filters[i].hasData = true
			}
		}
	}

	// Check if we have enough data to reorder
	hasLearned := false
	for _, sf := range scored_filters {
		if sf.hasData {
			hasLearned = true
			break
		}
	}
	if !hasLearned {
		return filters // No learned data, keep original order
	}

	// Sort by selectivity (lowest first = most selective)
	// Using simple insertion sort for small arrays
	for i := 1; i < len(scored_filters); i++ {
		j := i
		for j > 0 && scored_filters[j].selectivity < scored_filters[j-1].selectivity {
			scored_filters[j], scored_filters[j-1] = scored_filters[j-1], scored_filters[j]
			j--
		}
	}

	// Check if order changed
	orderChanged := false
	for i, sf := range scored_filters {
		if sf.filter.Key != filters[i].Key || sf.filter.Operator != filters[i].Operator {
			orderChanged = true
			break
		}
	}
	if orderChanged {
		fol.reorderings.Add(1)
	}

	// Extract reordered filters
	result := make([]metadata.Filter, len(filters))
	for i, sf := range scored_filters {
		result[i] = sf.filter
	}
	return result
}

// GetSelectivity returns the learned selectivity for a field.
// Returns 0.5 if not enough data.
func (fol *FilterOrderLearner) GetSelectivity(field string, op metadata.Operator, minObs int) float64 {
	fol.mu.RLock()
	defer fol.mu.RUnlock()

	stats, ok := fol.stats[field]
	if !ok || stats.observations < uint64(minObs) {
		return 0.5 // Unknown
	}

	// Try operator-specific first
	if opStats, ok := stats.operatorStats[op]; ok && opStats.observations >= uint64(minObs) {
		return opStats.selectivityEMA
	}

	return stats.selectivityEMA
}

// EstimateSelectivity estimates selectivity for a single filter.
// Combines field-level and operator-level statistics.
func (fol *FilterOrderLearner) EstimateSelectivity(f metadata.Filter) float64 {
	if fol == nil {
		return 0.5
	}
	return fol.GetSelectivity(f.Key, f.Operator, 1)
}

// Stats returns learner statistics.
func (fol *FilterOrderLearner) Stats() FilterOrderStats {
	fol.mu.RLock()
	numFields := len(fol.stats)
	fol.mu.RUnlock()

	return FilterOrderStats{
		TotalQueries:  fol.totalQueries.Load(),
		Reorderings:   fol.reorderings.Load(),
		FieldsTracked: numFields,
	}
}

// FilterOrderStats contains learner statistics.
type FilterOrderStats struct {
	TotalQueries  uint64
	Reorderings   uint64
	FieldsTracked int
}

// ReorderingRate returns the rate of queries where filters were reordered.
func (s FilterOrderStats) ReorderingRate() float64 {
	if s.TotalQueries == 0 {
		return 0
	}
	return float64(s.Reorderings) / float64(s.TotalQueries)
}

// FieldSelectivitySnapshot returns a snapshot of all field selectivities.
func (fol *FilterOrderLearner) FieldSelectivitySnapshot() map[string]float64 {
	fol.mu.RLock()
	defer fol.mu.RUnlock()

	result := make(map[string]float64, len(fol.stats))
	for field, stats := range fol.stats {
		result[field] = stats.selectivityEMA
	}
	return result
}

// Reset clears all learned data.
func (fol *FilterOrderLearner) Reset() {
	fol.mu.Lock()
	defer fol.mu.Unlock()

	fol.stats = make(map[string]*filterFieldStats)
	fol.totalQueries.Store(0)
	fol.reorderings.Store(0)
}
