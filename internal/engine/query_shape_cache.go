// Package engine provides the core database engine.
package engine

import (
	"hash/maphash"
	"sync"
	"sync/atomic"
	"time"

	"github.com/hupe1980/vecgo/metadata"
)

// QueryShapeCache caches execution plans for hot query patterns.
// Instead of re-analyzing filters and estimating costs for every query,
// we cache the execution plan for queries with the same "shape".
//
// Shape = fields involved + operators (not values).
// Example: "category=? AND price>?" is one shape, regardless of actual values.
//
// Benefits:
//   - Skip filter analysis for repeated query patterns (common in apps)
//   - Pre-computed optimal filter order
//   - Cached selectivity estimates
//   - Pre-computed segment pruning decisions
type QueryShapeCache struct {
	mu      sync.RWMutex
	plans   map[uint64]*ExecutionPlan
	seed    maphash.Seed
	maxSize int

	// Stats
	hits   atomic.Uint64
	misses atomic.Uint64
	evicts atomic.Uint64
}

// ExecutionPlan is a cached plan for a query shape.
type ExecutionPlan struct {
	// Shape signature (for validation)
	shapeSignature uint64

	// Optimal filter evaluation order (field indices)
	filterOrder []int

	// Estimated selectivities per filter
	selectivities []float64

	// Combined estimated selectivity
	combinedSelectivity float64

	// Recommended EF (search expansion factor)
	recommendedEF int

	// Whether to use speculative search
	useSpeculative bool

	// Execution hints
	hints ExecutionHints

	// Plan freshness
	createdAt time.Time
	lastUsed  atomic.Value // time.Time
	useCount  atomic.Uint64
}

// ExecutionHints provides execution guidance based on historical patterns.
type ExecutionHints struct {
	// Use microindexes for these field indices
	UseMicroindex []int

	// Expected result count (for pre-allocation)
	ExpectedResults int

	// Typical latency for this shape (for budget allocation)
	TypicalLatency time.Duration

	// Whether bitmap is better than row-list for results
	PreferBitmap bool

	// Whether to enable distance short-circuiting
	EnableShortCircuit bool

	// Recommended per-segment budget split
	BudgetSplitRatio float64
}

// QueryShapeCacheConfig configures the query shape cache.
type QueryShapeCacheConfig struct {
	// MaxSize is the maximum number of cached plans.
	// Default: 1000
	MaxSize int

	// PlanTTL is how long plans are valid.
	// Default: 1 hour
	PlanTTL time.Duration

	// Enabled controls whether caching is active.
	Enabled bool
}

// DefaultQueryShapeCacheConfig returns the default configuration.
func DefaultQueryShapeCacheConfig() QueryShapeCacheConfig {
	return QueryShapeCacheConfig{
		MaxSize: 1000,
		PlanTTL: time.Hour,
		Enabled: true,
	}
}

// NewQueryShapeCache creates a new query shape cache.
func NewQueryShapeCache(cfg QueryShapeCacheConfig) *QueryShapeCache {
	return &QueryShapeCache{
		plans:   make(map[uint64]*ExecutionPlan),
		seed:    maphash.MakeSeed(),
		maxSize: cfg.MaxSize,
	}
}

// ComputeShape computes the shape signature for a filter set.
// Same fields + operators = same shape, regardless of values.
func (c *QueryShapeCache) ComputeShape(filters []metadata.Filter) uint64 {
	if len(filters) == 0 {
		return 0
	}

	var h maphash.Hash
	h.SetSeed(c.seed)

	// Sort-independent hashing: XOR field+operator hashes
	var combined uint64
	for _, f := range filters {
		h.Reset()
		h.WriteString(f.Key)
		h.WriteString(string(f.Operator))
		combined ^= h.Sum64()
	}

	// Add filter count to distinguish shapes
	h.Reset()
	h.WriteByte(byte(len(filters)))
	combined ^= h.Sum64()

	return combined
}

// Get retrieves a cached plan for a query shape.
// Returns nil if not found or expired.
func (c *QueryShapeCache) Get(shape uint64) *ExecutionPlan {
	if c == nil || shape == 0 {
		return nil
	}

	c.mu.RLock()
	plan := c.plans[shape]
	c.mu.RUnlock()

	if plan == nil {
		c.misses.Add(1)
		return nil
	}

	// Update usage stats
	plan.lastUsed.Store(time.Now())
	plan.useCount.Add(1)
	c.hits.Add(1)

	return plan
}

// Put stores a plan for a query shape.
func (c *QueryShapeCache) Put(shape uint64, plan *ExecutionPlan) {
	if c == nil || shape == 0 || plan == nil {
		return
	}

	plan.shapeSignature = shape
	plan.createdAt = time.Now()
	plan.lastUsed.Store(time.Now())

	c.mu.Lock()
	defer c.mu.Unlock()

	// Evict if at capacity
	if len(c.plans) >= c.maxSize {
		c.evictLRU()
	}

	c.plans[shape] = plan
}

// evictLRU evicts the least recently used plan.
// Must be called with lock held.
func (c *QueryShapeCache) evictLRU() {
	var oldestKey uint64
	var oldestTime time.Time

	for k, v := range c.plans {
		lastUsed, ok := v.lastUsed.Load().(time.Time)
		if !ok {
			lastUsed = v.createdAt
		}

		if oldestKey == 0 || lastUsed.Before(oldestTime) {
			oldestKey = k
			oldestTime = lastUsed
		}
	}

	if oldestKey != 0 {
		delete(c.plans, oldestKey)
		c.evicts.Add(1)
	}
}

// BuildPlan creates an execution plan for a filter set.
func (c *QueryShapeCache) BuildPlan(
	filters []metadata.Filter,
	learner *FilterOrderLearner,
	stats *SegmentStatsAggregate,
) *ExecutionPlan {
	plan := &ExecutionPlan{
		filterOrder:   make([]int, len(filters)),
		selectivities: make([]float64, len(filters)),
		hints: ExecutionHints{
			EnableShortCircuit: true,
			BudgetSplitRatio:   1.0,
		},
	}

	// Compute selectivities and optimal order
	type filterWithIndex struct {
		idx         int
		selectivity float64
	}

	indexed := make([]filterWithIndex, len(filters))
	combinedSel := 1.0

	for i, f := range filters {
		sel := 0.5 // Default selectivity

		// Use learner if available
		if learner != nil {
			sel = learner.EstimateSelectivity(f)
		}

		// Use segment stats if available
		if stats != nil {
			if statsSel := stats.EstimateSelectivity(f); statsSel > 0 {
				// Combine with learned selectivity (weighted average)
				sel = sel*0.4 + statsSel*0.6
			}
		}

		indexed[i] = filterWithIndex{idx: i, selectivity: sel}
		plan.selectivities[i] = sel
		combinedSel *= sel
	}

	// Sort by selectivity (most selective first)
	for i := 0; i < len(indexed)-1; i++ {
		minIdx := i
		for j := i + 1; j < len(indexed); j++ {
			if indexed[j].selectivity < indexed[minIdx].selectivity {
				minIdx = j
			}
		}
		indexed[i], indexed[minIdx] = indexed[minIdx], indexed[i]
	}

	// Store optimal order
	for i, fi := range indexed {
		plan.filterOrder[i] = fi.idx
	}

	plan.combinedSelectivity = combinedSel

	// Determine if speculative search is worthwhile
	plan.useSpeculative = combinedSel > 0.01 && combinedSel < 0.5

	// Recommend EF based on selectivity
	if combinedSel < 0.01 {
		plan.recommendedEF = 200 // Very selective: need more exploration
	} else if combinedSel < 0.1 {
		plan.recommendedEF = 100
	} else {
		plan.recommendedEF = 50
	}

	// Microindex hints: use for equality filters on low-selectivity fields
	for i, f := range filters {
		if f.Operator == metadata.OpEqual && plan.selectivities[i] < 0.1 {
			plan.hints.UseMicroindex = append(plan.hints.UseMicroindex, i)
		}
	}

	// Bitmap vs rows hint
	plan.hints.PreferBitmap = combinedSel > 0.1

	// Expected results (heuristic)
	plan.hints.ExpectedResults = int(combinedSel * 10000)
	if plan.hints.ExpectedResults < 10 {
		plan.hints.ExpectedResults = 10
	}
	if plan.hints.ExpectedResults > 1000 {
		plan.hints.ExpectedResults = 1000
	}

	return plan
}

// InvalidateAll clears all cached plans.
// Call after schema changes or significant data changes.
func (c *QueryShapeCache) InvalidateAll() {
	if c == nil {
		return
	}

	c.mu.Lock()
	c.plans = make(map[uint64]*ExecutionPlan)
	c.mu.Unlock()
}

// Stats returns cache statistics.
func (c *QueryShapeCache) Stats() QueryShapeCacheStats {
	if c == nil {
		return QueryShapeCacheStats{}
	}

	c.mu.RLock()
	size := len(c.plans)
	c.mu.RUnlock()

	return QueryShapeCacheStats{
		Size:   size,
		Hits:   c.hits.Load(),
		Misses: c.misses.Load(),
		Evicts: c.evicts.Load(),
	}
}

// QueryShapeCacheStats contains cache statistics.
type QueryShapeCacheStats struct {
	Size   int
	Hits   uint64
	Misses uint64
	Evicts uint64
}

// HitRate returns the cache hit rate (0.0-1.0).
func (s QueryShapeCacheStats) HitRate() float64 {
	total := s.Hits + s.Misses
	if total == 0 {
		return 0
	}
	return float64(s.Hits) / float64(total)
}

// SegmentStatsAggregate provides aggregated segment statistics for planning.
type SegmentStatsAggregate struct {
	// Per-field statistics across all segments
	fieldSelectivity map[string]float64

	// Minimum/maximum values per field
	numericRanges map[string][2]float64 // [min, max]

	// Total row count
	totalRows int64
}

// NewSegmentStatsAggregate creates an aggregate from segment stats.
func NewSegmentStatsAggregate() *SegmentStatsAggregate {
	return &SegmentStatsAggregate{
		fieldSelectivity: make(map[string]float64),
		numericRanges:    make(map[string][2]float64),
	}
}

// AddFieldSelectivity adds field selectivity information.
func (a *SegmentStatsAggregate) AddFieldSelectivity(field string, selectivity float64) {
	if existing, ok := a.fieldSelectivity[field]; ok {
		// Average with existing
		a.fieldSelectivity[field] = (existing + selectivity) / 2
	} else {
		a.fieldSelectivity[field] = selectivity
	}
}

// AddNumericRange adds numeric range information.
func (a *SegmentStatsAggregate) AddNumericRange(field string, minVal, maxVal float64) {
	if existing, ok := a.numericRanges[field]; ok {
		if minVal < existing[0] {
			existing[0] = minVal
		}
		if maxVal > existing[1] {
			existing[1] = maxVal
		}
		a.numericRanges[field] = existing
	} else {
		a.numericRanges[field] = [2]float64{minVal, maxVal}
	}
}

// SetTotalRows sets the total row count.
func (a *SegmentStatsAggregate) SetTotalRows(rows int64) {
	a.totalRows = rows
}

// EstimateSelectivity estimates selectivity for a filter.
func (a *SegmentStatsAggregate) EstimateSelectivity(f metadata.Filter) float64 {
	if a == nil {
		return 0
	}

	// Check field selectivity
	if sel, ok := a.fieldSelectivity[f.Key]; ok {
		return sel
	}

	// For range queries, use numeric range
	if r, ok := a.numericRanges[f.Key]; ok {
		rangeSize := r[1] - r[0]
		if rangeSize > 0 {
			switch f.Operator {
			case metadata.OpGreaterThan, metadata.OpGreaterEqual:
				val := f.Value.F64
				if f.Value.Kind == metadata.KindInt {
					val = float64(f.Value.I64)
				}
				return (r[1] - val) / rangeSize
			case metadata.OpLessThan, metadata.OpLessEqual:
				val := f.Value.F64
				if f.Value.Kind == metadata.KindInt {
					val = float64(f.Value.I64)
				}
				return (val - r[0]) / rangeSize
			}
		}
	}

	return 0
}
