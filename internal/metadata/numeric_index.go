package imetadata

import (
	"bufio"
	"encoding/binary"
	"io"
	"math"
	"slices"
	"sort"
	"sync"

	"github.com/hupe1980/vecgo/internal/simd"
	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/model"
)

// LowCardinalityThreshold is the maximum distinct values for bitmap precomputation.
// Below this threshold, we precompute value→bitmap maps for O(1) equality lookups.
// Aligned with roaring bitmap container boundaries (array vs bitmap cutoff).
const LowCardinalityThreshold = 512

// NumericIndex provides O(log n) range queries for numeric fields.
// Uses a columnar layout for optimal cache locality during binary search.
//
// Architecture:
//   - Columnar storage: values []float64 and rowIDs []uint32 (aligned)
//   - Binary search on values for range boundaries
//   - Batch bitmap add via AddMany for efficient result building
//   - Precomputed bitmaps for low-cardinality fields (O(1) equality)
//
// Performance:
//   - Range queries: O(log n + m) where m is the number of matches
//   - Equality (low cardinality): O(1) bitmap lookup
//   - Cache-optimal: binary search touches only values array
//   - Memory: 12 bytes per entry (8 float64 + 4 uint32) + bitmaps for low-card fields
//
// When to use NumericIndex vs InvertedIndex:
//   - NumericIndex: high cardinality fields (timestamps, prices, scores)
//   - InvertedIndex: low cardinality fields (categories, status, types)
type NumericIndex struct {
	// fields maps field name to columnar numeric data
	fields map[string]*numericField

	// sealed indicates whether the index has been sealed.
	// After sealing, reads can proceed without locks on immutable structures.
	sealed bool
}

// fieldStats tracks statistics for adaptive query execution.
type fieldStats struct {
	min         float64 // Minimum value
	max         float64 // Maximum value
	cardinality int     // Number of distinct values (computed on Seal)
}

// numericField stores columnar (value, rowID) data for a single field.
// Invariant: len(values) == len(rowIDs)
// Invariant when sorted: values[i] belongs to rowIDs[i], sorted ascending by value
//
// After Seal(): The following fields are immutable and can be read lock-free:
//   - values, rowIDs (sorted arrays)
//   - bitmapIndex, sortedUniqueValues, prefixBitmaps (precomputed bitmaps)
//   - rowIDToIndex or rowIDToIndexSlice (reverse lookup for O(1) MatchRowID)
//   - stats (computed statistics)
//
// The pendingDeletes map is the only mutable field after Seal().
type numericField struct {
	// values is the column of numeric values (sorted when sealed)
	values []float64

	// rowIDs is the column of row IDs (aligned with values)
	rowIDs []uint32

	// rowIDToIndex maps rowID → index into values/rowIDs arrays (sparse rowIDs).
	// Built on Seal() for O(1) reverse lookup in MatchRowID.
	// IMMUTABLE after Seal() - can be read lock-free.
	// Used when rowIDs are sparse (max > 2*len).
	rowIDToIndex map[uint32]int

	// rowIDToIndexSlice is the dense alternative to rowIDToIndex.
	// Uses array indexing (no hash) for faster lookup when rowIDs are dense.
	// Value of -1 means rowID not present in this field.
	// Used when rowIDs are dense (max < 2*len).
	rowIDToIndexSlice []int32

	// sorted indicates whether the field is sorted (false during bulk load)
	sorted bool

	// stats contains min/max/cardinality for adaptive execution
	stats fieldStats

	// bitmapIndex is precomputed for low-cardinality fields (nil if high cardinality)
	// Maps value → bitmap of rowIDs. Built on Seal() if cardinality <= LowCardinalityThreshold.
	// IMMUTABLE after Seal() - can be read lock-free.
	bitmapIndex map[float64]*LocalBitmap

	// sortedUniqueValues contains unique values in sorted order for deterministic iteration.
	// Built alongside bitmapIndex on Seal(). Enables cache-friendly range queries.
	// IMMUTABLE after Seal() - can be read lock-free.
	sortedUniqueValues []float64

	// prefixBitmaps[i] contains the cumulative OR of bitmaps for values[0..i] (inclusive).
	// This enables O(1) range queries: OpLessEqual(v) → just return prefixBitmaps[idx].
	// Built alongside bitmapIndex on Seal(). Only for low-cardinality fields.
	// IMMUTABLE after Seal() - can be read lock-free.
	prefixBitmaps []*LocalBitmap

	// pendingDeletes tracks rowIDs to remove on next Seal() (deferred deletion)
	// Using map for O(1) lookup during queries.
	// This is the only mutable field after Seal().
	pendingDeletes map[uint32]struct{}
}

// NewNumericIndex creates a new empty numeric index.
func NewNumericIndex() *NumericIndex {
	return &NumericIndex{
		fields: make(map[string]*numericField),
	}
}

// newNumericField creates a new numericField with initial capacity.
func newNumericField(capacity int) *numericField {
	return &numericField{
		values: make([]float64, 0, capacity),
		rowIDs: make([]uint32, 0, capacity),
		sorted: true,
		stats: fieldStats{
			min: math.Inf(1),
			max: math.Inf(-1),
		},
	}
}

// Add adds a numeric value for a field and rowID.
// If the value is not numeric (int or float), it is ignored.
// Thread-safety: Caller must hold write lock.
func (ni *NumericIndex) Add(fieldKey string, value metadata.Value, rowID model.RowID) {
	var numVal float64
	switch value.Kind {
	case metadata.KindInt:
		numVal = float64(value.I64)
	case metadata.KindFloat:
		numVal = value.F64
	default:
		return // Not numeric, ignore
	}

	field, ok := ni.fields[fieldKey]
	if !ok {
		field = newNumericField(64)
		ni.fields[fieldKey] = field
	}

	field.values = append(field.values, numVal)
	field.rowIDs = append(field.rowIDs, uint32(rowID))

	// Update stats incrementally
	if numVal < field.stats.min {
		field.stats.min = numVal
	}
	if numVal > field.stats.max {
		field.stats.max = numVal
	}

	// If this breaks sort order, mark as unsorted
	if field.sorted && len(field.values) > 1 {
		prevVal := field.values[len(field.values)-2]
		currVal := field.values[len(field.values)-1]
		if currVal < prevVal {
			field.sorted = false
		}
	}

	// Invalidate precomputed bitmaps (will be rebuilt on Seal)
	field.bitmapIndex = nil
}

// Remove marks a rowID for deferred deletion.
// Actual removal happens on Seal() to avoid hot-path slice reallocations.
// The value parameter is used only for type validation (must be numeric).
// Thread-safety: Caller must hold write lock.
func (ni *NumericIndex) Remove(fieldKey string, value metadata.Value, rowID model.RowID) {
	// Validate numeric type
	switch value.Kind {
	case metadata.KindInt, metadata.KindFloat:
		// OK
	default:
		return
	}

	field, ok := ni.fields[fieldKey]
	if !ok {
		return
	}

	rid := uint32(rowID)

	// Deferred deletion: mark for removal on next Seal()
	if field.pendingDeletes == nil {
		field.pendingDeletes = make(map[uint32]struct{})
	}
	field.pendingDeletes[rid] = struct{}{}

	// Invalidate precomputed bitmaps
	field.bitmapIndex = nil
}

// Seal sorts all fields, compacts pending deletes, and builds bitmap indexes.
// Call this after bulk loading or before persisting.
// Uses parallel processing for large fields.
//
// After Seal():
//   - The sorted arrays (values, rowIDs) are immutable
//   - The bitmap indexes (bitmapIndex, prefixBitmaps) are immutable
//   - Only pendingDeletes can be modified
//
// Thread-safety: Caller must hold write lock.
func (ni *NumericIndex) Seal() {
	if len(ni.fields) == 0 {
		ni.sealed = true
		return
	}

	// For small number of fields, process sequentially
	if len(ni.fields) <= 2 {
		for _, field := range ni.fields {
			ni.sealField(field)
		}
		ni.sealed = true
		return
	}

	// Parallel processing for multiple fields
	var wg sync.WaitGroup
	for _, field := range ni.fields {
		wg.Add(1)
		go func(f *numericField) {
			defer wg.Done()
			ni.sealField(f)
		}(field)
	}
	wg.Wait()
	ni.sealed = true
}

// IsSealed returns true if the index has been sealed.
// After sealing, the index's sorted arrays and bitmap indexes are immutable
// and can be read without locks (except for pendingDeletes).
func (ni *NumericIndex) IsSealed() bool {
	return ni.sealed
}

// sealField processes a single field: compact deletes, sort, compute stats, build bitmaps.
func (ni *NumericIndex) sealField(field *numericField) {
	// 1. Apply pending deletes using swap-delete (O(n) single pass, no allocations)
	if len(field.pendingDeletes) > 0 {
		writeIdx := 0
		for readIdx := 0; readIdx < len(field.values); readIdx++ {
			rid := field.rowIDs[readIdx]
			if _, deleted := field.pendingDeletes[rid]; !deleted {
				if writeIdx != readIdx {
					field.values[writeIdx] = field.values[readIdx]
					field.rowIDs[writeIdx] = rid
				}
				writeIdx++
			}
		}
		field.values = field.values[:writeIdx]
		field.rowIDs = field.rowIDs[:writeIdx]
		field.pendingDeletes = nil
		field.sorted = false // Order may have changed
	}

	// 2. Sort if needed
	if !field.sorted && len(field.values) > 0 {
		ni.sortField(field)
	}

	// 3. Compute cardinality and update stats
	if len(field.values) > 0 {
		field.stats.min = field.values[0]
		field.stats.max = field.values[len(field.values)-1]
		field.stats.cardinality = countDistinct(field.values)
	}

	// 4. Build rowID → index lookup for O(1) MatchRowID
	// Check if rowIDs are dense (0 to N-1) - if so, use slice for faster lookup
	maxRowID := uint32(0)
	for _, rid := range field.rowIDs {
		if rid > maxRowID {
			maxRowID = rid
		}
	}

	// If rowIDs are reasonably dense (max < 2*len), use slice; otherwise use map
	if int(maxRowID) < len(field.rowIDs)*2 {
		// Dense: use slice for O(1) array indexing (no hash)
		field.rowIDToIndexSlice = make([]int32, maxRowID+1)
		for i := range field.rowIDToIndexSlice {
			field.rowIDToIndexSlice[i] = -1 // Sentinel for "not present"
		}
		for i, rid := range field.rowIDs {
			field.rowIDToIndexSlice[rid] = int32(i)
		}
		field.rowIDToIndex = nil // Clear map
	} else {
		// Sparse: use map
		field.rowIDToIndex = make(map[uint32]int, len(field.rowIDs))
		for i, rid := range field.rowIDs {
			field.rowIDToIndex[rid] = i
		}
		field.rowIDToIndexSlice = nil // Clear slice
	}

	// 5. Build bitmap index for low-cardinality fields
	if field.stats.cardinality > 0 && field.stats.cardinality <= LowCardinalityThreshold {
		ni.buildBitmapIndex(field)
	} else {
		field.bitmapIndex = nil // High cardinality, use column scan
	}
}

// countDistinct counts distinct values in a sorted slice.
func countDistinct(values []float64) int {
	if len(values) == 0 {
		return 0
	}
	count := 1
	prev := values[0]
	for i := 1; i < len(values); i++ {
		if values[i] != prev {
			count++
			prev = values[i]
		}
	}
	return count
}

// buildBitmapIndex creates value→bitmap map for O(1) equality lookups.
// Also builds sortedUniqueValues for deterministic, cache-friendly iteration.
// Also builds prefixBitmaps for O(1) range queries (OpLessEqual, OpLessThan, etc.).
//
// All bitmaps are run-optimized for better compression and query performance.
func (ni *NumericIndex) buildBitmapIndex(field *numericField) {
	// Preallocate with known cardinality
	field.bitmapIndex = make(map[float64]*LocalBitmap, field.stats.cardinality)
	field.sortedUniqueValues = make([]float64, 0, field.stats.cardinality)
	field.prefixBitmaps = make([]*LocalBitmap, 0, field.stats.cardinality)

	// Single pass: group rowIDs by value
	// Since values are sorted, we can batch consecutive equal values
	if len(field.values) == 0 {
		return
	}

	startIdx := 0
	currentVal := field.values[0]

	for i := 1; i <= len(field.values); i++ {
		// Check if we've reached a new value or end of array
		if i == len(field.values) || field.values[i] != currentVal {
			// Batch add all rowIDs for this value
			// NOTE: Use NewLocalBitmap(), NOT GetPooledBitmap() - these are stored in the index
			bm := NewLocalBitmap()
			bm.AddMany(field.rowIDs[startIdx:i])
			bm.RunOptimize() // Compress for faster queries
			field.bitmapIndex[currentVal] = bm
			field.sortedUniqueValues = append(field.sortedUniqueValues, currentVal)

			// Build cumulative prefix bitmap for O(1) range queries
			// prefixBitmaps[i] = OR of bitmaps for all values[0..i]
			if len(field.prefixBitmaps) == 0 {
				// First value: just clone its bitmap
				prefix := bm.Clone()
				prefix.RunOptimize() // Prefix bitmaps benefit greatly from RLE
				field.prefixBitmaps = append(field.prefixBitmaps, prefix)
			} else {
				// Cumulative: OR previous prefix with current bitmap
				prefix := field.prefixBitmaps[len(field.prefixBitmaps)-1].Clone()
				prefix.Or(bm)
				prefix.RunOptimize() // Re-optimize after OR
				field.prefixBitmaps = append(field.prefixBitmaps, prefix)
			}

			if i < len(field.values) {
				startIdx = i
				currentVal = field.values[i]
			}
		}
	}
}

// sortField sorts a numeric field by value.
// Uses indirect sort to keep values and rowIDs arrays aligned.
func (ni *NumericIndex) sortField(field *numericField) {
	if field.sorted || len(field.values) == 0 {
		return
	}

	// Sort by value using indirect sort to keep rowIDs aligned
	indices := make([]int, len(field.values))
	for i := range indices {
		indices[i] = i
	}
	slices.SortFunc(indices, func(a, b int) int {
		va, vb := field.values[a], field.values[b]
		if va < vb {
			return -1
		}
		if va > vb {
			return 1
		}
		// Stable sort by rowID for equal values
		if field.rowIDs[a] < field.rowIDs[b] {
			return -1
		}
		if field.rowIDs[a] > field.rowIDs[b] {
			return 1
		}
		return 0
	})

	// Apply permutation
	newValues := make([]float64, len(field.values))
	newRowIDs := make([]uint32, len(field.rowIDs))
	for i, idx := range indices {
		newValues[i] = field.values[idx]
		newRowIDs[i] = field.rowIDs[idx]
	}
	field.values = newValues
	field.rowIDs = newRowIDs
	field.sorted = true
}

// HasField returns true if the field has numeric entries.
func (ni *NumericIndex) HasField(fieldKey string) bool {
	field, ok := ni.fields[fieldKey]
	return ok && len(field.values) > 0
}

// Len returns the number of entries for a field.
func (ni *NumericIndex) Len(fieldKey string) int {
	field, ok := ni.fields[fieldKey]
	if !ok {
		return 0
	}
	return len(field.values)
}

// GetStats returns minVal, maxVal, and cardinality for a field.
// Returns zeros if field doesn't exist or is empty.
func (ni *NumericIndex) GetStats(fieldKey string) (minVal, maxVal float64, cardinality int) {
	field, ok := ni.fields[fieldKey]
	if !ok || len(field.values) == 0 {
		return 0, 0, 0
	}
	return field.stats.min, field.stats.max, field.stats.cardinality
}

// IsLowCardinality returns true if the field has precomputed bitmap index.
func (ni *NumericIndex) IsLowCardinality(fieldKey string) bool {
	field, ok := ni.fields[fieldKey]
	return ok && field.bitmapIndex != nil
}

// QueryRange finds all rowIDs where fieldKey is in [minVal, maxVal].
// Set includeMin/includeMax to control boundary inclusion.
// Results are added to dst using batch operations for efficiency.
// Skips pending deletes if any.
//
// Thread-safety: Caller must hold read lock.
// Note: For unsorted data (before Seal()), this uses SIMD full-column scan.
// For sorted data (after Seal()), this uses binary search which is faster.
func (ni *NumericIndex) QueryRange(
	fieldKey string,
	minVal, maxVal float64,
	includeMin, includeMax bool,
	dst *LocalBitmap,
) {
	field, ok := ni.fields[fieldKey]
	if !ok || len(field.values) == 0 {
		return
	}

	// Adaptive execution: use SIMD scan for unsorted data, binary search for sorted
	if !field.sorted {
		// Unsorted: use SIMD full-column scan
		// Adjust bounds for exclusive queries
		scanMin := minVal
		scanMax := maxVal
		if !includeMin {
			scanMin = math.Nextafter(minVal, math.Inf(1))
		}
		if !includeMax {
			scanMax = math.Nextafter(maxVal, math.Inf(-1))
		}
		ni.queryRangeSIMDInternal(field, scanMin, scanMax, dst)
		return
	}

	// Binary search for lower bound using stdlib (cache-optimal)
	var lo int
	if includeMin {
		lo = sort.SearchFloat64s(field.values, minVal)
	} else {
		// Find first value > minVal
		lo = sort.SearchFloat64s(field.values, minVal)
		for lo < len(field.values) && field.values[lo] == minVal {
			lo++
		}
	}

	// Binary search for upper bound
	var hi int
	if includeMax {
		// Find first value > maxVal
		hi = sort.SearchFloat64s(field.values, maxVal)
		for hi < len(field.values) && field.values[hi] == maxVal {
			hi++
		}
	} else {
		hi = sort.SearchFloat64s(field.values, maxVal)
	}

	if hi <= lo {
		return // No matches
	}

	// Fast path: no pending deletes, use batch AddMany
	if len(field.pendingDeletes) == 0 {
		dst.AddMany(field.rowIDs[lo:hi])
		return
	}

	// Slow path: filter out pending deletes using pooled scratch buffer
	qs := GetQueryScratch()
	defer PutQueryScratch(qs)

	// Ensure capacity
	matchCount := hi - lo
	if cap(qs.TmpRowIDs) < matchCount {
		qs.TmpRowIDs = make([]uint32, 0, matchCount)
	}

	// Collect non-deleted rowIDs
	for i := lo; i < hi; i++ {
		rid := field.rowIDs[i]
		if _, deleted := field.pendingDeletes[rid]; !deleted {
			qs.TmpRowIDs = append(qs.TmpRowIDs, rid)
		}
	}

	if len(qs.TmpRowIDs) > 0 {
		dst.AddMany(qs.TmpRowIDs)
	}
}

// queryRangeSIMDInternal is the internal SIMD implementation used by QueryRange
// for unsorted data. It uses vectorized comparisons and gather operations.
func (ni *NumericIndex) queryRangeSIMDInternal(field *numericField, minVal, maxVal float64, dst *LocalBitmap) {
	n := len(field.values)

	// Get scratch buffers from pool
	qs := GetQueryScratch()
	defer PutQueryScratch(qs)

	// Ensure TmpIndices has enough capacity
	if cap(qs.TmpIndices) < n {
		qs.TmpIndices = make([]int32, n)
	}

	// SIMD: find all indices where minVal <= values[i] <= maxVal
	matchingIndices := simd.FilterRangeF64Indices(field.values, minVal, maxVal, qs.TmpIndices[:n])

	if len(matchingIndices) == 0 {
		return
	}

	// Ensure TmpRowIDs has enough capacity
	if cap(qs.TmpRowIDs) < len(matchingIndices) {
		qs.TmpRowIDs = make([]uint32, len(matchingIndices))
	}

	// SIMD: gather rowIDs at matching indices
	gatheredRowIDs := simd.GatherU32(field.rowIDs, matchingIndices, qs.TmpRowIDs[:len(matchingIndices)])

	// Filter out pending deletes if any
	// This is scalar but pendingDeletes is typically small, so overhead is minimal
	if len(field.pendingDeletes) > 0 {
		// Filter in-place to avoid allocation
		writeIdx := 0
		for _, rid := range gatheredRowIDs {
			if _, deleted := field.pendingDeletes[rid]; !deleted {
				gatheredRowIDs[writeIdx] = rid
				writeIdx++
			}
		}
		gatheredRowIDs = gatheredRowIDs[:writeIdx]
	}

	if len(gatheredRowIDs) > 0 {
		dst.AddMany(gatheredRowIDs)
	}
}

// QueryRangeSIMD performs a SIMD-accelerated full-column scan for range queries.
// Unlike QueryRange which uses binary search on sorted data, this scans all values
// using vectorized comparisons.
//
// When to use QueryRangeSIMD vs QueryRange:
//   - QueryRangeSIMD: unsorted data (during bulk load, before Seal())
//   - QueryRange: sorted data (after Seal()) - binary search is faster for sorted data
//
// Benchmarks show binary search outperforms SIMD scan on sorted data because:
//   - Binary search touches O(log n) cache lines vs O(n) for full scan
//   - Modern CPUs have excellent branch prediction for binary search
//
// The range is always inclusive: [minVal, maxVal].
// For exclusive bounds, adjust minVal/maxVal with math.Nextafter.
// Thread-safety: Caller must hold read lock.
func (ni *NumericIndex) QueryRangeSIMD(
	fieldKey string,
	minVal, maxVal float64,
	dst *LocalBitmap,
) {
	field, ok := ni.fields[fieldKey]
	if !ok || len(field.values) == 0 {
		return
	}

	n := len(field.values)

	// Get scratch buffers from pool
	qs := GetQueryScratch()
	defer PutQueryScratch(qs)

	// Ensure TmpIndices has enough capacity
	if cap(qs.TmpIndices) < n {
		qs.TmpIndices = make([]int32, n)
	}

	// SIMD: find all indices where minVal <= values[i] <= maxVal
	matchingIndices := simd.FilterRangeF64Indices(field.values, minVal, maxVal, qs.TmpIndices[:n])

	if len(matchingIndices) == 0 {
		return
	}

	// Ensure TmpRowIDs has enough capacity
	if cap(qs.TmpRowIDs) < len(matchingIndices) {
		qs.TmpRowIDs = make([]uint32, len(matchingIndices))
	}

	// SIMD: gather rowIDs at matching indices
	gatheredRowIDs := simd.GatherU32(field.rowIDs, matchingIndices, qs.TmpRowIDs[:len(matchingIndices)])

	// Filter out pending deletes if any
	if len(field.pendingDeletes) > 0 {
		// Filter in-place
		writeIdx := 0
		for _, rid := range gatheredRowIDs {
			if _, deleted := field.pendingDeletes[rid]; !deleted {
				gatheredRowIDs[writeIdx] = rid
				writeIdx++
			}
		}
		gatheredRowIDs = gatheredRowIDs[:writeIdx]
	}

	if len(gatheredRowIDs) > 0 {
		dst.AddMany(gatheredRowIDs)
	}
}

// ForEachMatch iterates over matching row IDs without allocating a bitmap.
// This is the zero-allocation path for cursor-based filter evaluation.
// Returns true if the iteration was completed, false if fn returned false (early termination).
func (ni *NumericIndex) ForEachMatch(
	fieldKey string,
	op metadata.Operator,
	filterVal float64,
	fn func(rowID uint32) bool,
) bool {
	field, ok := ni.fields[fieldKey]
	if !ok || len(field.values) == 0 {
		return true // No matches, iteration complete
	}

	// Must be sorted for binary search
	if !field.sorted {
		return true // Cannot evaluate unsorted data with this method
	}

	var lo, hi int

	switch op {
	case metadata.OpEqual:
		// Find exact range
		lo = sort.SearchFloat64s(field.values, filterVal)
		hi = lo
		for hi < len(field.values) && field.values[hi] == filterVal {
			hi++
		}

	case metadata.OpLessThan:
		lo = 0
		hi = sort.SearchFloat64s(field.values, filterVal)

	case metadata.OpLessEqual:
		lo = 0
		hi = sort.SearchFloat64s(field.values, filterVal)
		for hi < len(field.values) && field.values[hi] == filterVal {
			hi++
		}

	case metadata.OpGreaterThan:
		lo = sort.SearchFloat64s(field.values, filterVal)
		for lo < len(field.values) && field.values[lo] == filterVal {
			lo++
		}
		hi = len(field.values)

	case metadata.OpGreaterEqual:
		lo = sort.SearchFloat64s(field.values, filterVal)
		hi = len(field.values)

	case metadata.OpNotEqual:
		// Iterate all except matching
		eq_lo := sort.SearchFloat64s(field.values, filterVal)
		eq_hi := eq_lo
		for eq_hi < len(field.values) && field.values[eq_hi] == filterVal {
			eq_hi++
		}
		// Iterate [0, eq_lo) then [eq_hi, len)
		for i := 0; i < eq_lo; i++ {
			rid := field.rowIDs[i]
			if _, deleted := field.pendingDeletes[rid]; deleted {
				continue
			}
			if !fn(rid) {
				return false
			}
		}
		for i := eq_hi; i < len(field.values); i++ {
			rid := field.rowIDs[i]
			if _, deleted := field.pendingDeletes[rid]; deleted {
				continue
			}
			if !fn(rid) {
				return false
			}
		}
		return true

	default:
		return true // Unsupported operator
	}

	if hi <= lo {
		return true // No matches
	}

	// Iterate matching range
	for i := lo; i < hi; i++ {
		rid := field.rowIDs[i]
		if _, deleted := field.pendingDeletes[rid]; deleted {
			continue
		}
		if !fn(rid) {
			return false
		}
	}
	return true
}

// EstimateSelectivity estimates the selectivity of a filter on a field.
// Returns the fraction of rows that match (0.0 to 1.0).
func (ni *NumericIndex) EstimateSelectivity(fieldKey string, op metadata.Operator, filterVal float64) float64 {
	field, ok := ni.fields[fieldKey]
	if !ok || len(field.values) == 0 {
		return 0.0
	}

	total := float64(len(field.values))
	if total == 0 {
		return 0.0
	}

	// Use binary search to estimate count
	switch op {
	case metadata.OpEqual:
		lo := sort.SearchFloat64s(field.values, filterVal)
		hi := lo
		for hi < len(field.values) && field.values[hi] == filterVal {
			hi++
		}
		return float64(hi-lo) / total

	case metadata.OpLessThan:
		count := sort.SearchFloat64s(field.values, filterVal)
		return float64(count) / total

	case metadata.OpLessEqual:
		hi := sort.SearchFloat64s(field.values, filterVal)
		for hi < len(field.values) && field.values[hi] == filterVal {
			hi++
		}
		return float64(hi) / total

	case metadata.OpGreaterThan:
		lo := sort.SearchFloat64s(field.values, filterVal)
		for lo < len(field.values) && field.values[lo] == filterVal {
			lo++
		}
		return float64(len(field.values)-lo) / total

	case metadata.OpGreaterEqual:
		lo := sort.SearchFloat64s(field.values, filterVal)
		return float64(len(field.values)-lo) / total

	case metadata.OpNotEqual:
		lo := sort.SearchFloat64s(field.values, filterVal)
		hi := lo
		for hi < len(field.values) && field.values[hi] == filterVal {
			hi++
		}
		return 1.0 - float64(hi-lo)/total

	default:
		return 0.5 // Unknown operator
	}
}

// MatchRowID checks if a specific rowID matches a filter condition.
// Uses O(1) lookup via rowIDToIndex map (built during Seal).
// This is the zero-allocation hot path for FilterCursor range filters.
//
// Returns false if:
//   - Field doesn't exist
//   - RowID not found in field
//   - RowID has a pending delete
//   - Value doesn't match the filter condition
//
// Thread-safety: Caller must hold read lock.
func (ni *NumericIndex) MatchRowID(fieldKey string, op metadata.Operator, filterVal float64, rowID uint32) bool {
	field, ok := ni.fields[fieldKey]
	if !ok {
		return false
	}

	// O(1) lookup: rowID → index
	var idx int
	var found bool

	if field.rowIDToIndexSlice != nil {
		// Dense path: array indexing (no hash)
		if int(rowID) >= len(field.rowIDToIndexSlice) {
			return false
		}
		idxVal := field.rowIDToIndexSlice[rowID]
		if idxVal < 0 {
			return false
		}
		idx = int(idxVal)
		found = true
	} else {
		// Sparse path: map lookup
		idx, found = field.rowIDToIndex[rowID]
	}

	if !found {
		return false
	}

	// Check pending deletes
	if _, deleted := field.pendingDeletes[rowID]; deleted {
		return false
	}

	// Get the value for this rowID
	value := field.values[idx]

	// Compare against filter
	switch op {
	case metadata.OpEqual:
		return value == filterVal
	case metadata.OpNotEqual:
		return value != filterVal
	case metadata.OpLessThan:
		return value < filterVal
	case metadata.OpLessEqual:
		return value <= filterVal
	case metadata.OpGreaterThan:
		return value > filterVal
	case metadata.OpGreaterEqual:
		return value >= filterVal
	default:
		return false
	}
}

// GetNumericMatcher returns a pooled FilterMatcher for zero-allocation filter evaluation.
// This replaces closure-based GetFieldMatcher to eliminate heap escapes.
//
// Returns nil if the field doesn't exist or isn't sealed.
// Caller MUST call Release() on the returned matcher when done.
func (ni *NumericIndex) GetNumericMatcher(fieldKey string, op metadata.Operator, filterVal float64) FilterMatcher {
	field, ok := ni.fields[fieldKey]
	if !ok {
		return nil
	}

	// Pre-capture slices/maps
	rowIDToIndexSlice := field.rowIDToIndexSlice
	rowIDToIndex := field.rowIDToIndex
	values := field.values

	// If neither lookup structure is populated, the index isn't sealed.
	if rowIDToIndexSlice == nil && rowIDToIndex == nil {
		return nil
	}

	// Note: We don't support pendingDeletes in the zero-alloc path.
	// Immutable segments (the hot path) never have pending deletes.
	// Memtables use a different code path.

	numOp := NumericOpFromOperator(op)

	if rowIDToIndexSlice != nil {
		// Dense path: array indexing (optimal for immutable segments)
		return GetDenseNumericMatcher(rowIDToIndexSlice, values, numOp, filterVal)
	}

	// Sparse path: map lookup
	return GetSparseNumericMatcher(rowIDToIndex, values, numOp, filterVal)
}

// GetFieldMatcher returns a closure that matches rowIDs against a filter condition.
// The field is looked up ONCE at call time, eliminating per-rowID string map lookups.
// This is the optimized hot path for FilterCursor.
//
// Returns nil if the field doesn't exist.
func (ni *NumericIndex) GetFieldMatcher(fieldKey string, op metadata.Operator, filterVal float64) func(rowID uint32) bool {
	field, ok := ni.fields[fieldKey]
	if !ok {
		return nil
	}

	// Pre-capture slices/maps to avoid field access in hot path
	rowIDToIndexSlice := field.rowIDToIndexSlice
	rowIDToIndex := field.rowIDToIndex
	values := field.values
	pendingDeletes := field.pendingDeletes
	hasPendingDeletes := len(pendingDeletes) > 0

	// If neither lookup structure is populated, the index isn't sealed.
	// Return nil to signal caller should use fallback (document lookup).
	if rowIDToIndexSlice == nil && rowIDToIndex == nil {
		return nil
	}

	// Select optimal path based on data structure at closure creation time
	// This avoids branching inside the hot loop

	if rowIDToIndexSlice != nil {
		// Dense path: array indexing (no hash)
		if hasPendingDeletes {
			// Dense + deletes (rare for immutable segments)
			return func(rowID uint32) bool {
				if int(rowID) >= len(rowIDToIndexSlice) {
					return false
				}
				idxVal := rowIDToIndexSlice[rowID]
				if idxVal < 0 {
					return false
				}
				if _, deleted := pendingDeletes[rowID]; deleted {
					return false
				}
				return compareFloat64(values[idxVal], op, filterVal)
			}
		}
		// Dense + no deletes (optimal path for immutable segments)
		return func(rowID uint32) bool {
			if int(rowID) >= len(rowIDToIndexSlice) {
				return false
			}
			idxVal := rowIDToIndexSlice[rowID]
			if idxVal < 0 {
				return false
			}
			return compareFloat64(values[idxVal], op, filterVal)
		}
	}

	// Sparse path: map lookup
	if hasPendingDeletes {
		return func(rowID uint32) bool {
			idx, found := rowIDToIndex[rowID]
			if !found {
				return false
			}
			if _, deleted := pendingDeletes[rowID]; deleted {
				return false
			}
			return compareFloat64(values[idx], op, filterVal)
		}
	}
	return func(rowID uint32) bool {
		idx, found := rowIDToIndex[rowID]
		if !found {
			return false
		}
		return compareFloat64(values[idx], op, filterVal)
	}
}

// compareFloat64 is an inlined comparison helper.
// Using switch here lets the compiler generate efficient branch tables.
func compareFloat64(value float64, op metadata.Operator, filterVal float64) bool {
	switch op {
	case metadata.OpEqual:
		return value == filterVal
	case metadata.OpNotEqual:
		return value != filterVal
	case metadata.OpLessThan:
		return value < filterVal
	case metadata.OpLessEqual:
		return value <= filterVal
	case metadata.OpGreaterThan:
		return value > filterVal
	case metadata.OpGreaterEqual:
		return value >= filterVal
	default:
		return false
	}
}

// EvaluateFilter evaluates a numeric filter using adaptive execution.
// - Low cardinality fields: O(1) bitmap lookup (precomputed)
// - High cardinality fields: O(log n) binary search on column
// Returns a bitmap of matching rowIDs.
// Supports: OpLessThan, OpLessEqual, OpGreaterThan, OpGreaterEqual, OpNotEqual, OpEqual
// Thread-safety: Caller must hold read lock.
//
// NOTE: The returned bitmap is from a pool. Caller should call PutPooledBitmap when done.
// For zero-alloc hot paths, use EvaluateFilterInto instead.
func (ni *NumericIndex) EvaluateFilter(f metadata.Filter) *LocalBitmap {
	result := GetPooledBitmap()
	ni.EvaluateFilterInto(f, result)
	return result
}

// EvaluateFilterInto is the zero-allocation version of EvaluateFilter.
// Results are added (OR'd) into the provided destination bitmap.
// The destination is NOT cleared first - this allows building up results from multiple filters.
// For a fresh result, caller should clear dst before calling.
//
// Performance: This is the hot-path optimized version.
// - No allocations in steady state
// - Caller controls bitmap lifecycle (pooled or owned)
//
// Thread-safety: Caller must hold read lock.
func (ni *NumericIndex) EvaluateFilterInto(f metadata.Filter, dst *LocalBitmap) {
	field, ok := ni.fields[f.Key]
	if !ok || len(field.values) == 0 {
		return // No matches, dst unchanged
	}

	// Get filter value as float64
	var filterVal float64
	switch f.Value.Kind {
	case metadata.KindInt:
		filterVal = float64(f.Value.I64)
	case metadata.KindFloat:
		filterVal = f.Value.F64
	default:
		return // Non-numeric filter value
	}

	// Adaptive execution: use precomputed bitmaps for low-cardinality fields
	if field.bitmapIndex != nil {
		ni.evaluateWithBitmapInto(f.Operator, filterVal, field, dst)
		return
	}

	// High cardinality: use column scan with binary search
	ni.evaluateWithColumnInto(f.Key, f.Operator, filterVal, dst)
}

// evaluateWithBitmapInto uses precomputed bitmap index for O(1) equality
// and O(1) range queries using prefix bitmaps on low-cardinality fields.
// Uses sortedUniqueValues for deterministic, cache-friendly iteration.
// Results are OR'd into the destination bitmap (zero-alloc).
//
// Performance characteristics:
//   - OpEqual: O(1) bitmap lookup
//   - OpLessEqual/OpLessThan: O(1) using prefixBitmaps
//   - OpGreaterEqual/OpGreaterThan: O(1) using prefixBitmaps (all - prefix)
//   - OpNotEqual: O(1) using prefixBitmaps (all - equal)
func (ni *NumericIndex) evaluateWithBitmapInto(
	op metadata.Operator,
	filterVal float64,
	field *numericField,
	result *LocalBitmap,
) {
	n := len(field.sortedUniqueValues)
	if n == 0 {
		return
	}

	// Binary search to find position of filterVal
	idx := sort.SearchFloat64s(field.sortedUniqueValues, filterVal)
	hasExact := idx < n && field.sortedUniqueValues[idx] == filterVal

	switch op {
	case metadata.OpEqual:
		// O(1) bitmap lookup
		if hasExact {
			result.Or(field.bitmapIndex[filterVal])
		}

	case metadata.OpLessEqual:
		// O(1): Use prefix bitmap directly
		if hasExact {
			// Include the matching value
			result.Or(field.prefixBitmaps[idx])
		} else if idx > 0 {
			// Value not found, use prefix up to idx-1
			result.Or(field.prefixBitmaps[idx-1])
		}
		// else: all values > filterVal, return empty

	case metadata.OpLessThan:
		// O(1): Use prefix bitmap for values < filterVal
		if idx > 0 {
			result.Or(field.prefixBitmaps[idx-1])
		}
		// else: all values >= filterVal, return empty

	case metadata.OpGreaterEqual:
		// O(1): All - prefix[idx-1] (if idx > 0)
		// Equivalent to: values >= filterVal
		if idx == 0 {
			// All values >= filterVal
			result.Or(field.prefixBitmaps[n-1])
		} else if idx < n {
			// Clone the "all" bitmap and subtract prefix
			result.Or(field.prefixBitmaps[n-1])
			// Subtract prefix[idx-1] to get suffix[idx..]
			// Note: We need AndNot which roaring supports
			prefixToRemove := field.prefixBitmaps[idx-1]
			result.AndNot(prefixToRemove)
		}
		// else: idx >= n means all values < filterVal, return empty

	case metadata.OpGreaterThan:
		// O(1): All values > filterVal
		// If filterVal exists: remove prefix[idx] (values <= filterVal)
		// If filterVal doesn't exist: remove prefix[idx-1] (values < filterVal)
		var removeIdx int
		if hasExact {
			removeIdx = idx
		} else {
			removeIdx = idx - 1
		}

		// Start with all values
		result.Or(field.prefixBitmaps[n-1])

		// Remove the prefix (values <= filterVal or values < filterVal)
		if removeIdx >= 0 && removeIdx < n {
			result.AndNot(field.prefixBitmaps[removeIdx])
		}

	case metadata.OpNotEqual:
		// O(1): All - equal
		result.Or(field.prefixBitmaps[n-1])
		if hasExact {
			// Remove the matching value's bitmap
			result.AndNot(field.bitmapIndex[filterVal])
		}
	}
}

// evaluateWithColumnInto uses binary search on columnar data for high-cardinality fields.
// Results are OR'd into the destination bitmap (zero-alloc).
func (ni *NumericIndex) evaluateWithColumnInto(
	fieldKey string,
	op metadata.Operator,
	filterVal float64,
	result *LocalBitmap,
) {
	switch op {
	case metadata.OpEqual:
		// For equality, find the exact range
		ni.QueryRange(fieldKey, filterVal, filterVal, true, true, result)

	case metadata.OpLessThan:
		ni.QueryRange(fieldKey, math.Inf(-1), filterVal, true, false, result)

	case metadata.OpLessEqual:
		ni.QueryRange(fieldKey, math.Inf(-1), filterVal, true, true, result)

	case metadata.OpGreaterThan:
		ni.QueryRange(fieldKey, filterVal, math.Inf(1), false, true, result)

	case metadata.OpGreaterEqual:
		ni.QueryRange(fieldKey, filterVal, math.Inf(1), true, true, result)

	case metadata.OpNotEqual:
		// All values except the specified one
		// Get all, then remove matches
		ni.QueryRange(fieldKey, math.Inf(-1), math.Inf(1), true, true, result)
		// Use scratch buffer to find and remove matching rowIDs
		field := ni.fields[fieldKey]
		if field != nil {
			// Binary search for matching range
			lo := sort.SearchFloat64s(field.values, filterVal)
			hi := lo
			for hi < len(field.values) && field.values[hi] == filterVal {
				hi++
			}
			// Remove matching rowIDs
			for i := lo; i < hi; i++ {
				result.Remove(field.rowIDs[i])
			}
		}
	}
}

// Cardinality returns the total number of (value, rowID) pairs across all fields.
func (ni *NumericIndex) Cardinality() int {
	total := 0
	for _, field := range ni.fields {
		total += len(field.values)
	}
	return total
}

// FieldCount returns the number of indexed fields.
func (ni *NumericIndex) FieldCount() int {
	return len(ni.fields)
}

// WriteTo writes the numeric index to an io.Writer in a compact binary format.
// Uses buffered writing for better I/O performance.
func (ni *NumericIndex) WriteTo(w io.Writer) (int64, error) {
	// Wrap in buffered writer if not already buffered
	bw, ok := w.(*bufio.Writer)
	if !ok {
		bw = bufio.NewWriterSize(w, 64*1024) // 64KB buffer
	}

	var written int64
	buf := make([]byte, binary.MaxVarintLen64)

	// Write field count
	n := binary.PutUvarint(buf, uint64(len(ni.fields)))
	nw, err := bw.Write(buf[:n])
	written += int64(nw)
	if err != nil {
		return written, err
	}

	// Use chunked buffer for batch encoding (reduces syscall overhead)
	const chunkSize = 8 * 1024 // 8KB chunks
	chunk := make([]byte, chunkSize)
	offset := 0

	// Helper to flush chunk when nearly full
	flushChunk := func() error {
		if offset > 0 {
			nw, err := bw.Write(chunk[:offset])
			written += int64(nw)
			if err != nil {
				return err
			}
			offset = 0
		}
		return nil
	}

	// Helper to ensure space in chunk
	ensureSpace := func(needed int) error {
		if offset+needed > chunkSize {
			return flushChunk()
		}
		return nil
	}

	for fieldKey, field := range ni.fields {
		// Write field key length + data
		if err := ensureSpace(binary.MaxVarintLen64 + len(fieldKey)); err != nil {
			return written, err
		}
		n = binary.PutUvarint(chunk[offset:], uint64(len(fieldKey)))
		offset += n
		copy(chunk[offset:], fieldKey)
		offset += len(fieldKey)

		// Write entry count
		if err := ensureSpace(binary.MaxVarintLen64); err != nil {
			return written, err
		}
		n = binary.PutUvarint(chunk[offset:], uint64(len(field.values)))
		offset += n

		// Write values (delta-encoded for better compression)
		var prevBits uint64
		for _, v := range field.values {
			if err := ensureSpace(binary.MaxVarintLen64); err != nil {
				return written, err
			}
			bits := math.Float64bits(v)
			delta := bits - prevBits
			n = binary.PutUvarint(chunk[offset:], delta)
			offset += n
			prevBits = bits
		}

		// Write rowIDs (delta-encoded since they tend to be sequential)
		var prevRowID uint32
		for _, rid := range field.rowIDs {
			if err := ensureSpace(binary.MaxVarintLen64); err != nil {
				return written, err
			}
			delta := rid - prevRowID
			n = binary.PutUvarint(chunk[offset:], uint64(delta))
			offset += n
			prevRowID = rid
		}
	}

	// Final flush
	if err := flushChunk(); err != nil {
		return written, err
	}

	// Flush buffered writer if we created it
	if !ok {
		if err := bw.Flush(); err != nil {
			return written, err
		}
	}

	return written, nil
}

// ReadFrom reads the numeric index from an io.Reader.
func (ni *NumericIndex) ReadFrom(r io.Reader) (int64, error) {
	var read int64

	// Read field count
	fieldCount, err := readUvarintFromReader(r)
	if err != nil {
		return read, err
	}

	ni.fields = make(map[string]*numericField, fieldCount)

	for range fieldCount {
		// Read field key length
		keyLen, err := readUvarintFromReader(r)
		if err != nil {
			return read, err
		}
		keyBytes := make([]byte, keyLen)
		if _, err := io.ReadFull(r, keyBytes); err != nil {
			return read, err
		}
		fieldKey := string(keyBytes)

		// Read entry count
		entryCount, err := readUvarintFromReader(r)
		if err != nil {
			return read, err
		}

		field := &numericField{
			values: make([]float64, entryCount),
			rowIDs: make([]uint32, entryCount),
			sorted: true, // Data from disk is sorted
		}

		// Read values (delta-encoded)
		var prevBits uint64
		for i := range entryCount {
			delta, err := readUvarintFromReader(r)
			if err != nil {
				return read, err
			}
			bits := prevBits + delta
			prevBits = bits
			field.values[i] = math.Float64frombits(bits)
		}

		// Read rowIDs (delta-encoded)
		var prevRowID uint32
		for i := range entryCount {
			delta, err := readUvarintFromReader(r)
			if err != nil {
				return read, err
			}
			rid := prevRowID + uint32(delta)
			prevRowID = rid
			field.rowIDs[i] = rid
		}

		ni.fields[fieldKey] = field
	}

	return read, nil
}

// readUvarintFromReader reads a uvarint from an io.Reader.
func readUvarintFromReader(r io.Reader) (uint64, error) {
	if br, ok := r.(io.ByteReader); ok {
		return binary.ReadUvarint(br)
	}
	// Fallback: read byte by byte
	var x uint64
	var s uint
	for i := 0; ; i++ {
		var b [1]byte
		if _, err := r.Read(b[:]); err != nil {
			return 0, err
		}
		if b[0] < 0x80 {
			if i > 9 || (i == 9 && b[0] > 1) {
				return 0, io.ErrUnexpectedEOF // overflow
			}
			return x | uint64(b[0])<<s, nil
		}
		x |= uint64(b[0]&0x7f) << s
		s += 7
	}
}
