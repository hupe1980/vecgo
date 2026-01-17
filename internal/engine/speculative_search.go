// Package engine provides the core database engine.
package engine

import (
	"context"
	"sync"
	"sync/atomic"
)

// SpeculativeSearch runs filtered and unfiltered searches in parallel,
// merging results. This hides filter uncertainty when selectivity is unknown.
//
// Why: Pre-filtering can be slower than post-filtering for low-selectivity filters.
// Running both in parallel and taking the best result ensures good worst-case latency.
//
// Design:
//   - Launch filtered search with predicate
//   - Launch unfiltered search (rerank with filter later)
//   - Return first to complete with enough results
//   - Cancel the other
//
// Cost: 2x CPU in worst case, but bounded by budget.
// Benefit: Hides filter uncertainty, better P99.
type SpeculativeSearch struct {
	// Configuration
	minResultsFiltered   int     // Minimum results before filtered can "win"
	minResultsUnfiltered int     // Minimum results before unfiltered can "win"
	selectivityThreshold float64 // Don't speculate if selectivity known to be very low

	// Stats
	filteredWins   atomic.Uint64
	unfilteredWins atomic.Uint64
	speculations   atomic.Uint64
}

// SpeculativeConfig configures speculative search.
type SpeculativeConfig struct {
	// MinResultsFiltered is the minimum results from filtered search before it can win.
	// Default: k (the requested result count)
	MinResultsFiltered int

	// MinResultsUnfiltered is the minimum results from unfiltered search before it can win.
	// Default: k * 2 (need extra for post-filtering)
	MinResultsUnfiltered int

	// SelectivityThreshold: don't speculate if estimated selectivity < this.
	// Default: 0.01 (1%)
	SelectivityThreshold float64

	// Enabled controls whether speculative search is active.
	Enabled bool
}

// DefaultSpeculativeConfig returns the default configuration.
func DefaultSpeculativeConfig() SpeculativeConfig {
	return SpeculativeConfig{
		MinResultsFiltered:   10,
		MinResultsUnfiltered: 20,
		SelectivityThreshold: 0.01,
		Enabled:              true,
	}
}

// NewSpeculativeSearch creates a new speculative search coordinator.
func NewSpeculativeSearch(cfg SpeculativeConfig) *SpeculativeSearch {
	return &SpeculativeSearch{
		minResultsFiltered:   cfg.MinResultsFiltered,
		minResultsUnfiltered: cfg.MinResultsUnfiltered,
		selectivityThreshold: cfg.SelectivityThreshold,
	}
}

// SearchResult represents results from one search branch.
type SearchResult[T any] struct {
	Results   []T
	Filtered  bool // true if results are pre-filtered
	Exhausted bool // true if search exhausted its budget
	Error     error
}

// SpeculativeRunner coordinates parallel search execution.
type SpeculativeRunner[T any] struct {
	spec   *SpeculativeSearch
	k      int
	ctx    context.Context
	cancel context.CancelFunc

	mu         sync.Mutex
	winner     *SearchResult[T]
	winnerCond *sync.Cond
}

// NewSpeculativeRunner creates a new speculative runner for a search.
func NewSpeculativeRunner[T any](ctx context.Context, s *SpeculativeSearch, k int) *SpeculativeRunner[T] {
	ctx, cancel := context.WithCancel(ctx)
	runner := &SpeculativeRunner[T]{
		spec:   s,
		k:      k,
		ctx:    ctx,
		cancel: cancel,
	}
	runner.winnerCond = sync.NewCond(&runner.mu)
	return runner
}

// ShouldSpeculate returns true if speculative search is worthwhile.
// Don't speculate when:
//   - Selectivity is very low (filter will definitely be faster)
//   - Selectivity is very high (filter has no effect)
//   - Budget is too tight
func (s *SpeculativeSearch) ShouldSpeculate(estimatedSelectivity float64, budgetStats BudgetStats) bool {
	if s == nil {
		return false
	}

	// Don't speculate if selectivity is very low (filter will win)
	if estimatedSelectivity > 0 && estimatedSelectivity < s.selectivityThreshold {
		return false
	}

	// Don't speculate if selectivity is very high (no filter effect)
	if estimatedSelectivity > 0.9 {
		return false
	}

	// Don't speculate if budget is already >50% used
	if budgetStats.UtilizationPercent() > 50 {
		return false
	}

	return true
}

// ReportFiltered reports results from the filtered search branch.
func (r *SpeculativeRunner[T]) ReportFiltered(results []T, exhausted bool, err error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.winner != nil {
		return // Already have a winner
	}

	// Check if filtered search has enough results
	if len(results) >= r.spec.minResultsFiltered || exhausted || err != nil {
		r.winner = &SearchResult[T]{
			Results:   results,
			Filtered:  true,
			Exhausted: exhausted,
			Error:     err,
		}
		r.spec.filteredWins.Add(1)
		r.cancel()
		r.winnerCond.Broadcast()
	}
}

// ReportUnfiltered reports results from the unfiltered search branch.
func (r *SpeculativeRunner[T]) ReportUnfiltered(results []T, exhausted bool, err error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.winner != nil {
		return // Already have a winner
	}

	// Check if unfiltered search has enough results
	if len(results) >= r.spec.minResultsUnfiltered || exhausted || err != nil {
		r.winner = &SearchResult[T]{
			Results:   results,
			Filtered:  false,
			Exhausted: exhausted,
			Error:     err,
		}
		r.spec.unfilteredWins.Add(1)
		r.cancel()
		r.winnerCond.Broadcast()
	}
}

// Wait waits for a winner and returns the result.
func (r *SpeculativeRunner[T]) Wait() *SearchResult[T] {
	r.mu.Lock()
	defer r.mu.Unlock()

	for r.winner == nil {
		r.winnerCond.Wait()
	}

	return r.winner
}

// Context returns the runner's context (cancelled when winner found).
func (r *SpeculativeRunner[T]) Context() context.Context {
	return r.ctx
}

// Cancel cancels the speculative search.
func (r *SpeculativeRunner[T]) Cancel() {
	r.cancel()
}

// Stats returns speculative search statistics.
func (s *SpeculativeSearch) Stats() SpeculativeStats {
	if s == nil {
		return SpeculativeStats{}
	}

	return SpeculativeStats{
		Speculations:   s.speculations.Load(),
		FilteredWins:   s.filteredWins.Load(),
		UnfilteredWins: s.unfilteredWins.Load(),
	}
}

// SpeculativeStats contains speculative search statistics.
type SpeculativeStats struct {
	Speculations   uint64
	FilteredWins   uint64
	UnfilteredWins uint64
}

// WinRate returns the filtered search win rate (0.0-1.0).
func (s SpeculativeStats) WinRate() float64 {
	total := s.FilteredWins + s.UnfilteredWins
	if total == 0 {
		return 0
	}
	return float64(s.FilteredWins) / float64(total)
}

// DualResultMerger merges results from filtered and unfiltered searches.
type DualResultMerger[T any] struct {
	filtered   []T
	unfiltered []T
	k          int
	scoreFn    func(T) float32
	filterFn   func(T) bool
}

// NewDualResultMerger creates a merger for combining speculative results.
func NewDualResultMerger[T any](k int, scoreFn func(T) float32, filterFn func(T) bool) *DualResultMerger[T] {
	return &DualResultMerger[T]{
		k:        k,
		scoreFn:  scoreFn,
		filterFn: filterFn,
	}
}

// AddFiltered adds results from the filtered search.
func (m *DualResultMerger[T]) AddFiltered(results []T) {
	m.filtered = results
}

// AddUnfiltered adds results from the unfiltered search.
func (m *DualResultMerger[T]) AddUnfiltered(results []T) {
	m.unfiltered = results
}

// Merge combines results, preferring filtered, falling back to post-filtered unfiltered.
func (m *DualResultMerger[T]) Merge() []T {
	// If filtered has enough results, use them
	if len(m.filtered) >= m.k {
		return m.filtered[:m.k]
	}

	// Post-filter unfiltered results
	var postFiltered []T
	for _, r := range m.unfiltered {
		if m.filterFn(r) {
			postFiltered = append(postFiltered, r)
		}
	}

	// Merge: use filtered results, fill gaps with post-filtered
	result := make([]T, 0, m.k)

	// Add all filtered results
	result = append(result, m.filtered...)

	// Add post-filtered results not already in filtered
	seen := make(map[float32]bool)
	for _, r := range result {
		seen[m.scoreFn(r)] = true
	}

	for _, r := range postFiltered {
		if !seen[m.scoreFn(r)] && len(result) < m.k {
			result = append(result, r)
		}
	}

	return result
}
