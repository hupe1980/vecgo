// Package engine provides the core database engine.
package engine

import (
	"context"
	"sync/atomic"
	"time"
)

// QueryBudget enforces resource limits during query execution.
// It provides predictable P99 latency by capping:
//   - Total distance computations
//   - Wall-clock time
//   - Nodes visited per segment
//   - Memory allocations
//
// Design: Pass budget through context, check periodically in hot loops.
// Overhead: ~1ns per check (atomic load + branch).
type QueryBudget struct {
	// Distance computation budget
	maxDistanceOps  int64
	usedDistanceOps atomic.Int64

	// Time budget
	deadline time.Time
	started  time.Time

	// Node visit budget (per segment)
	maxNodesPerSegment int64

	// Memory budget (bytes)
	maxMemory  int64
	usedMemory atomic.Int64

	// Early termination flag
	exhausted atomic.Bool

	// Stats for observability
	exhaustedReason atomic.Value // string
}

// BudgetConfig configures query resource limits.
type BudgetConfig struct {
	// MaxDistanceOps limits total distance computations.
	// 0 = unlimited. Typical: 10K-100K for sub-50ms queries.
	MaxDistanceOps int64

	// MaxDuration limits wall-clock time.
	// 0 = unlimited. Typical: 100ms-1s.
	MaxDuration time.Duration

	// MaxNodesPerSegment limits node visits per segment.
	// 0 = unlimited. Typical: 1K-10K.
	MaxNodesPerSegment int64

	// MaxMemory limits memory allocations (bytes).
	// 0 = unlimited. Typical: 10MB-100MB.
	MaxMemory int64
}

// DefaultBudgetConfig returns a permissive budget for general use.
func DefaultBudgetConfig() BudgetConfig {
	return BudgetConfig{
		MaxDistanceOps:     0, // unlimited
		MaxDuration:        0, // unlimited
		MaxNodesPerSegment: 0, // unlimited
		MaxMemory:          0, // unlimited
	}
}

// StrictBudgetConfig returns a strict budget for predictable latency.
func StrictBudgetConfig(targetLatency time.Duration) BudgetConfig {
	// Heuristic: at 1μs per distance op, 50K ops ≈ 50ms
	distanceOps := targetLatency.Microseconds()
	if distanceOps < 1000 {
		distanceOps = 1000
	}

	return BudgetConfig{
		MaxDistanceOps:     distanceOps,
		MaxDuration:        targetLatency,
		MaxNodesPerSegment: distanceOps / 10,
		MaxMemory:          50 * 1024 * 1024, // 50MB
	}
}

// NewQueryBudget creates a budget from configuration.
func NewQueryBudget(cfg BudgetConfig) *QueryBudget {
	qb := &QueryBudget{
		maxDistanceOps:     cfg.MaxDistanceOps,
		maxNodesPerSegment: cfg.MaxNodesPerSegment,
		maxMemory:          cfg.MaxMemory,
		started:            time.Now(),
	}

	if cfg.MaxDuration > 0 {
		qb.deadline = qb.started.Add(cfg.MaxDuration)
	}

	return qb
}

// budgetKey is the context key for QueryBudget.
type budgetKey struct{}

// WithBudget attaches a budget to a context.
func WithBudget(ctx context.Context, budget *QueryBudget) context.Context {
	return context.WithValue(ctx, budgetKey{}, budget)
}

// BudgetFromContext retrieves the budget from context, or nil if none.
func BudgetFromContext(ctx context.Context) *QueryBudget {
	if b, ok := ctx.Value(budgetKey{}).(*QueryBudget); ok {
		return b
	}
	return nil
}

// CheckDistance checks if we can perform n distance operations.
// Returns false if budget exhausted. Call before distance computation.
//
// Hot path: ~1ns overhead (atomic add + compare).
func (qb *QueryBudget) CheckDistance(n int) bool {
	if qb == nil || qb.maxDistanceOps == 0 {
		return true
	}

	if qb.exhausted.Load() {
		return false
	}

	used := qb.usedDistanceOps.Add(int64(n))
	if used > qb.maxDistanceOps {
		qb.markExhausted("distance_ops")
		return false
	}

	return true
}

// CheckDeadline checks if the time budget is exhausted.
// Call periodically (every ~100 ops) to avoid overhead.
//
// Hot path: ~5ns overhead (time comparison).
func (qb *QueryBudget) CheckDeadline() bool {
	if qb == nil || qb.deadline.IsZero() {
		return true
	}

	if qb.exhausted.Load() {
		return false
	}

	if time.Now().After(qb.deadline) {
		qb.markExhausted("deadline")
		return false
	}

	return true
}

// CheckNodes checks segment node visit budget.
// Returns false if segment budget exhausted.
func (qb *QueryBudget) CheckNodes(nodesInSegment int) bool {
	if qb == nil || qb.maxNodesPerSegment == 0 {
		return true
	}

	if qb.exhausted.Load() {
		return false
	}

	if int64(nodesInSegment) > qb.maxNodesPerSegment {
		qb.markExhausted("nodes_per_segment")
		return false
	}

	return true
}

// CheckMemory checks memory budget and tracks usage.
// Returns false if memory budget exhausted.
func (qb *QueryBudget) CheckMemory(bytes int64) bool {
	if qb == nil || qb.maxMemory == 0 {
		return true
	}

	if qb.exhausted.Load() {
		return false
	}

	used := qb.usedMemory.Add(bytes)
	if used > qb.maxMemory {
		qb.markExhausted("memory")
		return false
	}

	return true
}

// ReleaseMemory returns memory to the budget.
func (qb *QueryBudget) ReleaseMemory(bytes int64) {
	if qb == nil || qb.maxMemory == 0 {
		return
	}
	qb.usedMemory.Add(-bytes)
}

// IsExhausted returns true if any budget limit was exceeded.
func (qb *QueryBudget) IsExhausted() bool {
	if qb == nil {
		return false
	}
	return qb.exhausted.Load()
}

// ExhaustedReason returns why the budget was exhausted.
func (qb *QueryBudget) ExhaustedReason() string {
	if qb == nil {
		return ""
	}
	if r := qb.exhaustedReason.Load(); r != nil {
		return r.(string)
	}
	return ""
}

func (qb *QueryBudget) markExhausted(reason string) {
	if qb.exhausted.CompareAndSwap(false, true) {
		qb.exhaustedReason.Store(reason)
	}
}

// Stats returns budget usage statistics.
func (qb *QueryBudget) Stats() BudgetStats {
	if qb == nil {
		return BudgetStats{}
	}

	return BudgetStats{
		DistanceOpsUsed:  qb.usedDistanceOps.Load(),
		DistanceOpsLimit: qb.maxDistanceOps,
		MemoryUsed:       qb.usedMemory.Load(),
		MemoryLimit:      qb.maxMemory,
		Elapsed:          time.Since(qb.started),
		TimeLimit:        qb.deadline.Sub(qb.started),
		Exhausted:        qb.exhausted.Load(),
		ExhaustedReason:  qb.ExhaustedReason(),
		NodesPerSegLimit: qb.maxNodesPerSegment,
	}
}

// BudgetStats contains budget usage statistics.
type BudgetStats struct {
	DistanceOpsUsed  int64
	DistanceOpsLimit int64
	MemoryUsed       int64
	MemoryLimit      int64
	Elapsed          time.Duration
	TimeLimit        time.Duration
	Exhausted        bool
	ExhaustedReason  string
	NodesPerSegLimit int64
}

// UtilizationPercent returns the highest budget utilization percentage.
func (s BudgetStats) UtilizationPercent() float64 {
	var maxUtil float64

	if s.DistanceOpsLimit > 0 {
		util := float64(s.DistanceOpsUsed) / float64(s.DistanceOpsLimit) * 100
		if util > maxUtil {
			maxUtil = util
		}
	}

	if s.MemoryLimit > 0 {
		util := float64(s.MemoryUsed) / float64(s.MemoryLimit) * 100
		if util > maxUtil {
			maxUtil = util
		}
	}

	if s.TimeLimit > 0 {
		util := float64(s.Elapsed) / float64(s.TimeLimit) * 100
		if util > maxUtil {
			maxUtil = util
		}
	}

	return maxUtil
}

// RemainingBudget returns a new budget with remaining limits.
// Useful for splitting budget across parallel segments.
func (qb *QueryBudget) RemainingBudget() *QueryBudget {
	if qb == nil {
		return nil
	}

	remaining := &QueryBudget{
		maxNodesPerSegment: qb.maxNodesPerSegment,
		started:            qb.started,
	}

	// Remaining distance ops
	if qb.maxDistanceOps > 0 {
		used := qb.usedDistanceOps.Load()
		remaining.maxDistanceOps = qb.maxDistanceOps - used
		if remaining.maxDistanceOps < 0 {
			remaining.maxDistanceOps = 0
		}
	}

	// Remaining time
	if !qb.deadline.IsZero() {
		remaining.deadline = qb.deadline
	}

	// Remaining memory
	if qb.maxMemory > 0 {
		used := qb.usedMemory.Load()
		remaining.maxMemory = qb.maxMemory - used
		if remaining.maxMemory < 0 {
			remaining.maxMemory = 0
		}
	}

	return remaining
}

// SplitBudget splits the budget across n segments for parallel execution.
// Each segment gets an equal share of the remaining budget.
func (qb *QueryBudget) SplitBudget(n int) []*QueryBudget {
	if qb == nil || n <= 0 {
		return nil
	}

	budgets := make([]*QueryBudget, n)
	remaining := qb.RemainingBudget()

	for i := range budgets {
		budgets[i] = &QueryBudget{
			maxNodesPerSegment: remaining.maxNodesPerSegment,
			started:            remaining.started,
			deadline:           remaining.deadline,
		}

		// Split distance ops evenly
		if remaining.maxDistanceOps > 0 {
			budgets[i].maxDistanceOps = remaining.maxDistanceOps / int64(n)
		}

		// Split memory evenly
		if remaining.maxMemory > 0 {
			budgets[i].maxMemory = remaining.maxMemory / int64(n)
		}
	}

	return budgets
}
