package imetadata

import (
	"sync"

	"github.com/hupe1980/vecgo/metadata"
)

// FilterMatcher is the zero-allocation interface for filter evaluation.
// Replaces closures with interface dispatch to eliminate heap escapes.
//
// DESIGN RATIONALE:
// Go closures that capture variables escape to heap (~50 bytes each).
// For filtered search with N filters, this causes N allocations per query.
// Interface dispatch uses vtables (static) instead of closures (heap).
//
// Performance:
//   - Closure: ~5ns + 1 alloc
//   - Interface: ~2ns + 0 alloc (vtable is static)
type FilterMatcher interface {
	// Matches returns true if the rowID passes the filter.
	// MUST be safe for concurrent use.
	Matches(rowID uint32) bool

	// Release returns the matcher to its pool.
	// MUST be called when done to avoid pool exhaustion.
	Release()
}

// -----------------------------------------------------------------------------
// Bitmap Matcher - For equality and IN filters
// -----------------------------------------------------------------------------

// BitmapMatcher checks if rowID exists in a bitmap.
// Used for OpEqual and OpIn filters.
type BitmapMatcher struct {
	bitmap *LocalBitmap
}

var bitmapMatcherPool = sync.Pool{
	New: func() any { return &BitmapMatcher{} },
}

// GetBitmapMatcher returns a pooled BitmapMatcher.
func GetBitmapMatcher(bm *LocalBitmap) *BitmapMatcher {
	m := bitmapMatcherPool.Get().(*BitmapMatcher)
	m.bitmap = bm
	return m
}

func (m *BitmapMatcher) Matches(rowID uint32) bool {
	return m.bitmap != nil && m.bitmap.Contains(rowID)
}

func (m *BitmapMatcher) Release() {
	m.bitmap = nil
	bitmapMatcherPool.Put(m)
}

// -----------------------------------------------------------------------------
// Not-Bitmap Matcher - For NotEqual filters
// -----------------------------------------------------------------------------

// NotBitmapMatcher checks if rowID does NOT exist in a bitmap.
// Used for OpNotEqual filters.
type NotBitmapMatcher struct {
	bitmap *LocalBitmap
}

var notBitmapMatcherPool = sync.Pool{
	New: func() any { return &NotBitmapMatcher{} },
}

// GetNotBitmapMatcher returns a pooled NotBitmapMatcher.
func GetNotBitmapMatcher(bm *LocalBitmap) *NotBitmapMatcher {
	m := notBitmapMatcherPool.Get().(*NotBitmapMatcher)
	m.bitmap = bm
	return m
}

func (m *NotBitmapMatcher) Matches(rowID uint32) bool {
	return m.bitmap == nil || !m.bitmap.Contains(rowID)
}

func (m *NotBitmapMatcher) Release() {
	m.bitmap = nil
	notBitmapMatcherPool.Put(m)
}

// -----------------------------------------------------------------------------
// Multi-Bitmap Matcher - For IN filters with multiple values
// -----------------------------------------------------------------------------

// MultiBitmapMatcher checks if rowID exists in any of the bitmaps.
// Used for OpIn filters with multiple values.
type MultiBitmapMatcher struct {
	bitmaps []*LocalBitmap
}

var multiBitmapMatcherPool = sync.Pool{
	New: func() any { return &MultiBitmapMatcher{bitmaps: make([]*LocalBitmap, 0, 8)} },
}

// GetMultiBitmapMatcher returns a pooled MultiBitmapMatcher.
func GetMultiBitmapMatcher(bitmaps []*LocalBitmap) *MultiBitmapMatcher {
	m := multiBitmapMatcherPool.Get().(*MultiBitmapMatcher)
	m.bitmaps = append(m.bitmaps[:0], bitmaps...)
	return m
}

func (m *MultiBitmapMatcher) Matches(rowID uint32) bool {
	for _, b := range m.bitmaps {
		if b.Contains(rowID) {
			return true
		}
	}
	return false
}

func (m *MultiBitmapMatcher) Release() {
	m.bitmaps = m.bitmaps[:0]
	multiBitmapMatcherPool.Put(m)
}

// -----------------------------------------------------------------------------
// Numeric Matchers - For range filters (pre-captured field data)
// -----------------------------------------------------------------------------

// NumericOp represents a numeric comparison operation.
type NumericOp uint8

const (
	NumericOpEq NumericOp = iota
	NumericOpNe
	NumericOpLt
	NumericOpLe
	NumericOpGt
	NumericOpGe
)

// NumericOpFromOperator converts metadata.Operator to NumericOp.
func NumericOpFromOperator(op metadata.Operator) NumericOp {
	switch op {
	case metadata.OpEqual:
		return NumericOpEq
	case metadata.OpNotEqual:
		return NumericOpNe
	case metadata.OpLessThan:
		return NumericOpLt
	case metadata.OpLessEqual:
		return NumericOpLe
	case metadata.OpGreaterThan:
		return NumericOpGt
	case metadata.OpGreaterEqual:
		return NumericOpGe
	default:
		return NumericOpEq
	}
}

// DenseNumericMatcher uses array indexing for dense rowID ranges.
// Optimal for immutable segments where rowIDs are contiguous.
type DenseNumericMatcher struct {
	rowIDToIndex []int32   // rowID -> index in values (-1 = not present)
	values       []float64 // indexed values
	op           NumericOp
	filterVal    float64
}

var denseNumericMatcherPool = sync.Pool{
	New: func() any { return &DenseNumericMatcher{} },
}

// GetDenseNumericMatcher returns a pooled DenseNumericMatcher.
func GetDenseNumericMatcher(rowIDToIndex []int32, values []float64, op NumericOp, filterVal float64) *DenseNumericMatcher {
	m := denseNumericMatcherPool.Get().(*DenseNumericMatcher)
	m.rowIDToIndex = rowIDToIndex
	m.values = values
	m.op = op
	m.filterVal = filterVal
	return m
}

func (m *DenseNumericMatcher) Matches(rowID uint32) bool {
	if int(rowID) >= len(m.rowIDToIndex) {
		return false
	}
	idx := m.rowIDToIndex[rowID]
	if idx < 0 {
		return false
	}
	return m.compare(m.values[idx])
}

func (m *DenseNumericMatcher) compare(value float64) bool {
	switch m.op {
	case NumericOpEq:
		return value == m.filterVal
	case NumericOpNe:
		return value != m.filterVal
	case NumericOpLt:
		return value < m.filterVal
	case NumericOpLe:
		return value <= m.filterVal
	case NumericOpGt:
		return value > m.filterVal
	case NumericOpGe:
		return value >= m.filterVal
	default:
		return false
	}
}

func (m *DenseNumericMatcher) Release() {
	m.rowIDToIndex = nil
	m.values = nil
	denseNumericMatcherPool.Put(m)
}

// SparseNumericMatcher uses map lookup for sparse rowID sets.
// Used for memtables and segments with non-contiguous rowIDs.
type SparseNumericMatcher struct {
	rowIDToIndex map[uint32]int
	values       []float64
	op           NumericOp
	filterVal    float64
}

var sparseNumericMatcherPool = sync.Pool{
	New: func() any { return &SparseNumericMatcher{} },
}

// GetSparseNumericMatcher returns a pooled SparseNumericMatcher.
func GetSparseNumericMatcher(rowIDToIndex map[uint32]int, values []float64, op NumericOp, filterVal float64) *SparseNumericMatcher {
	m := sparseNumericMatcherPool.Get().(*SparseNumericMatcher)
	m.rowIDToIndex = rowIDToIndex
	m.values = values
	m.op = op
	m.filterVal = filterVal
	return m
}

func (m *SparseNumericMatcher) Matches(rowID uint32) bool {
	idx, ok := m.rowIDToIndex[rowID]
	if !ok {
		return false
	}
	return m.compare(m.values[idx])
}

func (m *SparseNumericMatcher) compare(value float64) bool {
	switch m.op {
	case NumericOpEq:
		return value == m.filterVal
	case NumericOpNe:
		return value != m.filterVal
	case NumericOpLt:
		return value < m.filterVal
	case NumericOpLe:
		return value <= m.filterVal
	case NumericOpGt:
		return value > m.filterVal
	case NumericOpGe:
		return value >= m.filterVal
	default:
		return false
	}
}

func (m *SparseNumericMatcher) Release() {
	m.rowIDToIndex = nil
	m.values = nil
	sparseNumericMatcherPool.Put(m)
}

// -----------------------------------------------------------------------------
// Composite Matcher - AND of multiple matchers
// -----------------------------------------------------------------------------

// CompositeMatcher evaluates multiple matchers with AND semantics.
// Short-circuits on first false (most selective first for best perf).
type CompositeMatcher struct {
	matchers []FilterMatcher
}

var compositeMatcherPool = sync.Pool{
	New: func() any { return &CompositeMatcher{matchers: make([]FilterMatcher, 0, 8)} },
}

// GetCompositeMatcher returns a pooled CompositeMatcher.
func GetCompositeMatcher(matchers []FilterMatcher) *CompositeMatcher {
	m := compositeMatcherPool.Get().(*CompositeMatcher)
	m.matchers = append(m.matchers[:0], matchers...)
	return m
}

func (m *CompositeMatcher) Matches(rowID uint32) bool {
	for _, matcher := range m.matchers {
		if !matcher.Matches(rowID) {
			return false
		}
	}
	return true
}

func (m *CompositeMatcher) Release() {
	// Release all child matchers
	for _, matcher := range m.matchers {
		matcher.Release()
	}
	m.matchers = m.matchers[:0]
	compositeMatcherPool.Put(m)
}

// -----------------------------------------------------------------------------
// Null Matchers - For trivial cases
// -----------------------------------------------------------------------------

// AlwaysTrueMatcher always returns true (no filter).
type AlwaysTrueMatcher struct{}

var alwaysTrueInstance = &AlwaysTrueMatcher{}

func GetAlwaysTrueMatcher() *AlwaysTrueMatcher   { return alwaysTrueInstance }
func (m *AlwaysTrueMatcher) Matches(uint32) bool { return true }
func (m *AlwaysTrueMatcher) Release()            {} // Singleton, no-op

// AlwaysFalseMatcher always returns false (impossible filter).
type AlwaysFalseMatcher struct{}

var alwaysFalseInstance = &AlwaysFalseMatcher{}

func GetAlwaysFalseMatcher() *AlwaysFalseMatcher  { return alwaysFalseInstance }
func (m *AlwaysFalseMatcher) Matches(uint32) bool { return false }
func (m *AlwaysFalseMatcher) Release()            {} // Singleton, no-op

// -----------------------------------------------------------------------------
// Matcher Scratch - Pooled slice for building composite matchers
// -----------------------------------------------------------------------------

// MatcherScratch is a pooled slice for collecting matchers.
type MatcherScratch struct {
	Matchers []FilterMatcher
	Bitmaps  []*LocalBitmap // Scratch for collecting IN bitmaps
}

var matcherScratchPool = sync.Pool{
	New: func() any {
		return &MatcherScratch{
			Matchers: make([]FilterMatcher, 0, 8),
			Bitmaps:  make([]*LocalBitmap, 0, 8),
		}
	},
}

// GetMatcherScratch returns a pooled MatcherScratch.
func GetMatcherScratch() *MatcherScratch {
	s := matcherScratchPool.Get().(*MatcherScratch)
	s.Matchers = s.Matchers[:0]
	s.Bitmaps = s.Bitmaps[:0]
	return s
}

// PutMatcherScratch returns a MatcherScratch to the pool.
func PutMatcherScratch(s *MatcherScratch) {
	s.Matchers = s.Matchers[:0]
	s.Bitmaps = s.Bitmaps[:0]
	matcherScratchPool.Put(s)
}
