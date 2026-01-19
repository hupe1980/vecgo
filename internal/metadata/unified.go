package imetadata

import (
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"slices"
	"strconv"
	"sync"
	"unique"

	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/model"
)

// DocumentProvider is a function that retrieves a document by ID.
// The context should be used for cancellation and timeouts during I/O.
type DocumentProvider func(ctx context.Context, id model.RowID) (metadata.Document, bool)

// UnifiedIndex combines metadata storage with inverted indexing using Bitmaps.
// This provides efficient hybrid vector + metadata search with minimal memory overhead.
//
// Architecture:
//   - Primary storage: map[model.RowID]InternedDocument (metadata by RowID, interned keys)
//   - Inverted index: map[key]map[valueKey]*LocalBitmap (efficient posting lists)
//   - Numeric index: sorted (value, rowID) pairs for O(log n) range queries
//
// Benefits:
//   - Memory efficient (Bitmap compression + String Interning)
//   - Fast filter compilation (Bitmap AND/OR operations)
//   - O(log n) numeric range queries (vs O(cardinality) without NumericIndex)
//   - Simple API (single unified type)
type UnifiedIndex struct {
	mu sync.RWMutex

	// Primary metadata storage (id -> metadata document)
	documents map[model.RowID]metadata.InternedDocument

	// Inverted index for fast filtering
	// Structure: field -> valueKey -> bitmap of IDs
	// Bitmaps are compressed and support fast set operations
	inverted map[unique.Handle[string]]map[unique.Handle[string]]*LocalBitmap

	// Numeric index for fast range queries (< > <= >=)
	// Uses binary search on sorted (value, rowID) pairs
	numeric *NumericIndex

	// provider is an optional fallback for document retrieval
	provider DocumentProvider
}

// NewUnifiedIndex creates a new unified metadata index.
func NewUnifiedIndex() *UnifiedIndex {
	return &UnifiedIndex{
		documents: make(map[model.RowID]metadata.InternedDocument),
		inverted:  make(map[unique.Handle[string]]map[unique.Handle[string]]*LocalBitmap),
		numeric:   NewNumericIndex(),
	}
}

// internDocument converts a public Document to an InternedDocument.
func internDocument(doc metadata.Document) metadata.InternedDocument {
	iDoc := make(metadata.InternedDocument, len(doc))
	for k, v := range doc {
		iDoc[unique.Make(k)] = v
	}
	return iDoc
}

// Set stores metadata for an ID and updates the inverted index.
// This replaces any existing metadata for the ID.
func (ui *UnifiedIndex) Set(id model.RowID, doc metadata.Document) {
	if doc == nil {
		return
	}

	iDoc := internDocument(doc)

	ui.mu.Lock()
	defer ui.mu.Unlock()

	// Remove old document from inverted index
	if oldDoc, exists := ui.documents[id]; exists {
		ui.removeFromIndexLocked(id, oldDoc)
	}

	// Store new document
	ui.documents[id] = iDoc

	// Add to inverted index
	ui.addToIndexLocked(id, iDoc)
}

// AddInvertedIndex adds a document to the inverted index without storing the document itself.
// This is useful for building an index for immutable segments where documents are stored separately.
// Note: This does not support updates/deletes correctly as the old document is not known.
func (ui *UnifiedIndex) AddInvertedIndex(id model.RowID, doc metadata.Document) {
	if doc == nil {
		return
	}

	iDoc := internDocument(doc)

	ui.mu.Lock()
	defer ui.mu.Unlock()

	// Add to inverted index
	ui.addToIndexLocked(id, iDoc)
}

// SetDocumentProvider sets the document provider for fallback retrieval.
func (ui *UnifiedIndex) SetDocumentProvider(provider DocumentProvider) {
	ui.mu.Lock()
	defer ui.mu.Unlock()
	ui.provider = provider
}

// filterCost represents the estimated cost and selectivity of a filter.
type filterCost struct {
	filter      metadata.Filter
	selectivity float64 // Lower = more selective (0.0 = matches nothing, 1.0 = matches all)
	cost        int     // Estimated execution cost (lower = cheaper)
}

// cmpFilterCost compares two filterCosts by selectivity then cost.
// Package-level function to avoid closure allocation in SortFunc.
func cmpFilterCost(a, b filterCost) int {
	if a.selectivity != b.selectivity {
		if a.selectivity < b.selectivity {
			return -1
		}
		return 1
	}
	return a.cost - b.cost
}

// Cost model constants for predicate ordering.
// These costs reflect empirical benchmark results (range predicates are ~6× slower than equality).
//
// Ordering principle: Equality → Boolean short-circuit → Range (last)
// This minimizes total work by executing cheap, selective predicates first.
const (
	costEqual         = 1   // O(1) bitmap lookup - fastest
	costInSingle      = 1   // Same as equality
	costInMultiple    = 5   // Bitmap unions - still fast
	costRangeLowCard  = 50  // Low cardinality range - bitmap OR
	costRangeHighCard = 100 // High cardinality range - binary search + materialization
	costNotEqual      = 200 // Scans almost everything - always last
	costFallback      = 500 // Unknown operator - worst case
)

// estimateFilterCost estimates the selectivity and cost of a filter.
// This enables cost-based query planning: evaluate most selective filters first.
//
// Key insight from benchmarks:
//   - Range predicates are ~6× slower than equality (793µs vs 126µs at 50k vectors)
//   - Even with good selectivity estimates, range predicates should be evaluated last
//   - Equality filters should always go first (O(1) bitmap lookup)
//
// Ordering priority (lower cost = evaluated first):
//  1. OpEqual (cost=1) - O(1) bitmap lookup
//  2. OpIn (cost=1-5) - Single or multiple bitmap lookups
//  3. Range operators (cost=50-100) - Requires bitmap materialization
//  4. OpNotEqual (cost=200) - Matches most rows, always last
func (ui *UnifiedIndex) estimateFilterCost(f metadata.Filter) filterCost {
	fc := filterCost{filter: f, selectivity: 1.0, cost: costFallback}

	switch f.Operator {
	case metadata.OpEqual:
		// Equality is typically very selective
		// Check if value exists in index
		if b := ui.getBitmapLocked(f.Key, f.Value); b != nil {
			cardinality := b.Cardinality()
			total := ui.totalRowsLocked()
			if total > 0 {
				fc.selectivity = float64(cardinality) / float64(total)
			}
			fc.cost = costEqual // O(1) lookup - always fast
		} else {
			fc.selectivity = 0.0 // No matches
			fc.cost = costEqual
		}

	case metadata.OpIn:
		// IN is sum of individual equalities
		arr, ok := f.Value.AsArray()
		if ok {
			var totalCards uint64
			for _, v := range arr {
				if b := ui.getBitmapLocked(f.Key, v); b != nil {
					totalCards += b.Cardinality()
				}
			}
			total := ui.totalRowsLocked()
			if total > 0 {
				fc.selectivity = float64(totalCards) / float64(total)
				if fc.selectivity > 1.0 {
					fc.selectivity = 1.0 // Cap at 100%
				}
			}
			// Single element IN is as fast as equality
			if len(arr) <= 1 {
				fc.cost = costInSingle
			} else {
				fc.cost = costInMultiple
			}
		}

	case metadata.OpLessThan, metadata.OpLessEqual, metadata.OpGreaterThan, metadata.OpGreaterEqual:
		// Range queries: estimate based on value position in range
		// CRITICAL: These are ~6× slower than equality in practice (benchmarked)
		if ui.numeric.HasField(f.Key) {
			minVal, maxVal, card := ui.numeric.GetStats(f.Key)
			if card > 0 && maxVal > minVal {
				var filterVal float64
				switch f.Value.Kind {
				case metadata.KindInt:
					filterVal = float64(f.Value.I64)
				case metadata.KindFloat:
					filterVal = f.Value.F64
				}

				// Estimate position in range [0, 1]
				position := (filterVal - minVal) / (maxVal - minVal)
				position = max(0, min(1, position))

				switch f.Operator {
				case metadata.OpLessThan, metadata.OpLessEqual:
					fc.selectivity = position
				case metadata.OpGreaterThan, metadata.OpGreaterEqual:
					fc.selectivity = 1.0 - position
				}
			}
			// Cost depends on cardinality - but both are expensive compared to equality
			if card <= LowCardinalityThreshold {
				fc.cost = costRangeLowCard // Low cardinality: bitmap union
			} else {
				fc.cost = costRangeHighCard // High cardinality: binary search + materialization
			}
		} else {
			fc.cost = costFallback // No numeric index - full scan
		}

	case metadata.OpNotEqual:
		// NotEqual is typically not selective (matches most rows)
		// Always evaluate last - it provides minimal filtering benefit
		fc.selectivity = 0.9 // Conservative estimate
		fc.cost = costNotEqual
	}

	return fc
}

// totalRowsLocked returns the total number of rows (requires read lock held).
func (ui *UnifiedIndex) totalRowsLocked() uint64 {
	return uint64(len(ui.documents))
}

// EvaluateFilter evaluates a filter set against the inverted index and returns a bitmap.
// This method supports all operators, including numeric comparisons like OpLessThan.
//
// Query Planning:
//   - Estimates selectivity for each filter
//   - Evaluates most selective filters first (cost-based optimization)
//   - Short-circuits on empty intermediate results
//
// Lazy Execution:
//   - First filter uses zero-copy reference when possible (immutable OpEqual lookups)
//   - Cloning only occurs when mutation is needed (AND operations)
//   - This reduces N-1 bitmap allocations to 0-1 for typical multi-filter queries
//
// For numeric comparisons (< > <= >= !=):
//   - Uses NumericIndex with O(log n + matches) binary search if field has numeric entries
//   - Falls back to O(cardinality) scan only for non-numeric or missing fields
//
// NOTE: The returned bitmap is from a pool. Caller should call PutPooledBitmap when done.
func (ui *UnifiedIndex) EvaluateFilter(fs *metadata.FilterSet) *LocalBitmap {
	if fs == nil || len(fs.Filters) == 0 {
		return nil // All matches
	}

	ui.mu.RLock()
	defer ui.mu.RUnlock()

	filters := fs.Filters

	// Cost-based query planning: sort filters by estimated selectivity (most selective first)
	// This reduces intermediate result sizes and enables early termination
	if len(filters) > 1 {
		costs := make([]filterCost, len(filters))
		for i, f := range filters {
			costs[i] = ui.estimateFilterCost(f)
		}

		// Sort by selectivity (ascending) - most selective first
		// For equal selectivity, prefer lower cost
		slices.SortFunc(costs, cmpFilterCost)

		// Reorder filters
		reordered := make([]metadata.Filter, len(filters))
		for i, fc := range costs {
			reordered[i] = fc.filter
		}
		filters = reordered
	}

	var result *LocalBitmap
	resultIsImmutable := false // Track if result is a reference to an immutable bitmap

	for _, f := range filters {
		var current *LocalBitmap
		currentIsImmutable := false

		switch f.Operator {
		case metadata.OpEqual:
			b := ui.getBitmapLocked(f.Key, f.Value)
			if b == nil {
				return GetPooledBitmap() // Empty bitmap - no match for this filter
			}
			// Zero-copy: reference the immutable bitmap directly
			current = b
			currentIsImmutable = true

		case metadata.OpIn:
			arr, ok := f.Value.AsArray()
			if !ok || len(arr) == 0 {
				return GetPooledBitmap() // Empty - no valid array
			}

			// Single element: treat as OpEqual (zero-copy)
			if len(arr) == 1 {
				b := ui.getBitmapLocked(f.Key, arr[0])
				if b == nil {
					return GetPooledBitmap()
				}
				current = b
				currentIsImmutable = true
			} else {
				// Multiple elements: must materialize union
				current = GetPooledBitmap()
				for _, v := range arr {
					if b := ui.getBitmapLocked(f.Key, v); b != nil {
						current.Or(b)
					}
				}
			}

		case metadata.OpLessThan, metadata.OpLessEqual, metadata.OpGreaterThan, metadata.OpGreaterEqual, metadata.OpNotEqual:
			// Dual-index cost-based dispatch:
			// - Low cardinality (≤512 distinct values): BitmapIndex is faster (OR pre-materialized bitmaps)
			// - High cardinality (>512 distinct values): ColumnIndex is faster (binary search on sorted values)
			const lowCardinalityThreshold = 512

			distinctValues := ui.getFieldCardinality(f.Key)

			// Use EvaluateFilterInto to avoid allocating a new bitmap
			// Get a pooled bitmap but use the Into variant for zero-alloc evaluation
			current = GetPooledBitmap()
			if distinctValues > lowCardinalityThreshold && ui.numeric.HasField(f.Key) {
				// High cardinality: ColumnIndex with O(log n) binary search
				ui.numeric.EvaluateFilterInto(f, current)
			} else {
				// Low cardinality: BitmapIndex with fast bitmap unions
				ui.evaluateNumericFilterScanInto(f, current)
			}

		default:
			// Unsupported operator - fall back to empty (will fail filter)
			return GetPooledBitmap()
		}

		if result == nil {
			// First filter: use directly (may be immutable reference)
			result = current
			resultIsImmutable = currentIsImmutable
		} else {
			// Subsequent filters: AND with existing result

			// If result is immutable, we must clone before mutating
			if resultIsImmutable {
				cloned := GetPooledBitmap()
				cloned.Or(result)
				result = cloned
				resultIsImmutable = false
			}

			result.And(current)

			// Return non-immutable bitmaps to pool
			if !currentIsImmutable {
				PutPooledBitmap(current)
			}
		}

		// Early termination: if result is empty, no point continuing
		if result.IsEmpty() {
			// If result is immutable reference to empty bitmap, return pooled empty
			if resultIsImmutable {
				return GetPooledBitmap()
			}
			return result
		}
	}

	// If final result is still an immutable reference, clone it
	// (caller expects ownership of the returned bitmap)
	if resultIsImmutable && result != nil {
		cloned := GetPooledBitmap()
		cloned.Or(result)
		return cloned
	}

	return result
}

// EvaluateFilterResult evaluates a filter set using the dual-mode FilterResult.
// This is the zero-allocation version that avoids roaring bitmaps for low cardinality.
//
// Design:
//   - Low cardinality: returns FilterRows mode (SIMD-friendly []uint32)
//   - High cardinality: returns FilterBitmap mode (roaring.Bitmap)
//   - No allocations in hot path (uses QueryScratch)
//
// Returns:
//   - FilterResult with mode indicating the representation
//   - Caller does NOT need to return anything to pool (memory is query-scoped)
//
// Thread-safety: Thread-safe (holds RLock during evaluation)
func (ui *UnifiedIndex) EvaluateFilterResult(fs *metadata.FilterSet, qs *QueryScratch) FilterResult {
	if fs == nil || len(fs.Filters) == 0 {
		return AllResult() // No filter = match all
	}

	ui.mu.RLock()
	defer ui.mu.RUnlock()

	filters := fs.Filters

	// Cost-based query planning: sort filters by estimated selectivity (most selective first)
	// For small filter counts (typical: 1-3), avoid allocation by using stack array
	if len(filters) > 1 {
		// Stack-allocate for small counts (covers 99% of real queries)
		var costsBuf [8]filterCost
		costs := costsBuf[:0]
		if len(filters) > 8 {
			costs = make([]filterCost, 0, len(filters))
		}

		for _, f := range filters {
			costs = append(costs, ui.estimateFilterCost(f))
		}

		// Sort by selectivity (ascending) - most selective first
		slices.SortFunc(costs, cmpFilterCost)

		// Reorder filters in-place via sorted costs (no allocation)
		// costs already contains the filter references
		filters = make([]metadata.Filter, len(costs))
		for i, fc := range costs {
			filters[i] = fc.filter
		}
	}

	var result FilterResult

	for i, f := range filters {
		current := ui.evaluateSingleFilterResult(f, qs)

		if i == 0 {
			result = current
		} else {
			// AND with previous result
			// Note: For subsequent filters, we need a second scratch buffer
			// to avoid overwriting the result during AND operation
			result = ui.filterResultAndWithTemp(result, current, qs)
		}

		// Early termination
		if result.IsEmpty() {
			return EmptyResult()
		}
	}

	return result
}

// evaluateSingleFilterResult evaluates a single filter and returns a FilterResult.
// Uses zero-alloc patterns where possible.
func (ui *UnifiedIndex) evaluateSingleFilterResult(f metadata.Filter, qs *QueryScratch) FilterResult {
	switch f.Operator {
	case metadata.OpEqual:
		b := ui.getBitmapLocked(f.Key, f.Value)
		if b == nil {
			return EmptyResult()
		}
		// For small bitmaps, extract to rows mode for SIMD-friendly iteration
		const rowsThreshold = 1024
		if b.Cardinality() <= rowsThreshold {
			out := qs.TmpRowIDs[:0]
			out = b.ToArrayInto(out)
			qs.TmpRowIDs = out
			return RowsResult(out)
		}
		// Large bitmap: convert from storage (roaring) to execution (QueryBitmap)
		// This is the storage → execution boundary conversion
		qs.Tmp2.Clear()
		qs.Tmp2.PopulateFromRoaring(b.rb)
		return QueryBitmapResult(qs.Tmp2)

	case metadata.OpIn:
		arr, ok := f.Value.AsArray()
		if !ok || len(arr) == 0 {
			return EmptyResult()
		}

		// Single element: treat as OpEqual
		if len(arr) == 1 {
			return ui.evaluateSingleFilterResult(metadata.Filter{
				Key:      f.Key,
				Operator: metadata.OpEqual,
				Value:    arr[0],
			}, qs)
		}

		// Multiple elements: collect all matching IDs
		out := qs.TmpRowIDs[:0]
		for _, v := range arr {
			if b := ui.getBitmapLocked(f.Key, v); b != nil {
				out = b.ToArrayInto(out) // Append to existing
			}
		}
		// Sort for binary search support
		sortUint32(out)
		qs.TmpRowIDs = out
		return RowsResult(out)

	case metadata.OpLessThan, metadata.OpLessEqual, metadata.OpGreaterThan, metadata.OpGreaterEqual, metadata.OpNotEqual:
		const lowCardinalityThreshold = 512

		distinctValues := ui.getFieldCardinality(f.Key)

		// Use TmpStorage (roaring) for storage-layer evaluation, then convert
		qs.TmpStorage.rb.Clear()
		if distinctValues > lowCardinalityThreshold && ui.numeric.HasField(f.Key) {
			ui.numeric.EvaluateFilterInto(f, qs.TmpStorage)
		} else {
			ui.evaluateNumericFilterScanInto(f, qs.TmpStorage)
		}

		// Convert to appropriate mode
		const rowsThreshold = 1024
		if qs.TmpStorage.Cardinality() <= rowsThreshold {
			out := qs.TmpRowIDs[:0]
			out = qs.TmpStorage.ToArrayInto(out)
			qs.TmpRowIDs = out
			return RowsResult(out)
		}

		// Large result: convert from storage (roaring) to execution (QueryBitmap)
		// Use ToArrayInto + AddMany to avoid ToArray() allocation in PopulateFromRoaring
		out := qs.TmpRowIDs[:0]
		out = qs.TmpStorage.ToArrayInto(out)
		qs.TmpRowIDs = out

		qs.Tmp2.Clear()
		qs.Tmp2.AddMany(out)
		return QueryBitmapResult(qs.Tmp2)

	default:
		return EmptyResult()
	}
}

// filterResultAndWithTemp performs AND using a temporary buffer to avoid aliasing.
// This is needed when the result and current both use TmpRowIDs.
func (ui *UnifiedIndex) filterResultAndWithTemp(result, current FilterResult, qs *QueryScratch) FilterResult {
	// If one is empty, return empty
	if result.IsEmpty() || current.IsEmpty() {
		return EmptyResult()
	}

	// If both are rows mode and use the same buffer, we need to be careful
	if result.mode == FilterRows && current.mode == FilterRows {
		// Two-pointer intersection can work in-place if result is the destination
		return ui.andRowsRowsInPlace(result.rows, current.rows, qs)
	}

	// For other combinations, use the standard AND
	return FilterResultAnd(result, current, qs)
}

// andRowsRowsInPlace performs two-pointer intersection, reusing TmpRowIDs.
func (ui *UnifiedIndex) andRowsRowsInPlace(a, b []uint32, qs *QueryScratch) FilterResult {
	// Since both may be pointing to TmpRowIDs, we do in-place overwrite
	// This works because we only write to positions we've already read past
	out := qs.TmpRowIDs[:0]

	i, j := 0, 0
	for i < len(a) && j < len(b) {
		if a[i] == b[j] {
			out = append(out, a[i])
			i++
			j++
		} else if a[i] < b[j] {
			i++
		} else {
			j++
		}
	}

	qs.TmpRowIDs = out
	return RowsResult(out)
}

// evaluateNumericFilterScanInto handles comparison operators by scanning the inverted index.
// This is the SLOW PATH: O(cardinality) where cardinality is the number of unique values.
//
// This is only used as a fallback when NumericIndex doesn't have the field.
// The fast path (NumericIndex.EvaluateFilter) uses binary search for O(log n + matches).
//
// When this path is used:
//   - Field was never added to NumericIndex (non-numeric original value)
//   - Field added before NumericIndex was introduced (legacy data)
func (ui *UnifiedIndex) evaluateNumericFilterScanInto(f metadata.Filter, dst *LocalBitmap) {
	valueMap, ok := ui.inverted[unique.Make(f.Key)]
	if !ok {
		return // Field doesn't exist, no matches
	}

	for valueKey, bitmap := range valueMap {
		// Parse the stored value to check against filter
		// The valueKey is the serialized form of the value
		storedValue := parseValueKey(valueKey.Value())

		if matchesComparison(storedValue, f.Value, f.Operator) {
			dst.Or(bitmap)
		}
	}
}

// parseValueKey parses a value key back to a numeric value for comparison.
func parseValueKey(key string) metadata.Value {
	// Value.Key() format: "i:123" for int, "f:hex_bits" for float, "s:text" for string
	if len(key) < 2 {
		return metadata.String(key)
	}

	switch key[0] {
	case 'i':
		if key[1] == ':' {
			i, err := strconv.ParseInt(key[2:], 10, 64)
			if err == nil {
				return metadata.Int(i)
			}
		}
	case 'f':
		if key[1] == ':' {
			// Float is stored as hex bits
			bits, err := strconv.ParseUint(key[2:], 16, 64)
			if err == nil {
				return metadata.Float(math.Float64frombits(bits))
			}
		}
	}

	// Default to string
	if len(key) > 2 && key[1] == ':' {
		return metadata.String(key[2:])
	}
	return metadata.String(key)
}

// matchesComparison checks if stored value matches filter value with given operator.
func matchesComparison(stored, filter metadata.Value, op metadata.Operator) bool {
	// Get numeric values for comparison
	var storedNum, filterNum float64

	switch stored.Kind {
	case metadata.KindInt:
		storedNum = float64(stored.I64)
	case metadata.KindFloat:
		storedNum = stored.F64
	default:
		// Non-numeric comparison - only NotEqual makes sense for strings
		if op == metadata.OpNotEqual {
			return stored.Key() != filter.Key()
		}
		return false
	}

	switch filter.Kind {
	case metadata.KindInt:
		filterNum = float64(filter.I64)
	case metadata.KindFloat:
		filterNum = filter.F64
	default:
		return false
	}

	switch op {
	case metadata.OpLessThan:
		return storedNum < filterNum
	case metadata.OpLessEqual:
		return storedNum <= filterNum
	case metadata.OpGreaterThan:
		return storedNum > filterNum
	case metadata.OpGreaterEqual:
		return storedNum >= filterNum
	case metadata.OpNotEqual:
		return storedNum != filterNum
	default:
		return false
	}
}

// Get retrieves metadata for an ID.
// Returns nil if the ID doesn't exist.
// The context is used for provider lookups on fallback paths.
func (ui *UnifiedIndex) Get(ctx context.Context, id model.RowID) (metadata.Document, bool) {
	ui.mu.RLock()
	defer ui.mu.RUnlock()

	iDoc, ok := ui.documents[id]
	if ok {
		// Convert to public format
		doc := make(metadata.Document, len(iDoc))
		for k, v := range iDoc {
			doc[k.Value()] = v
		}
		return doc, true
	}

	if ui.provider != nil {
		return ui.provider(ctx, id)
	}

	return nil, false
}

// Delete removes metadata for an ID and updates the inverted index.
func (ui *UnifiedIndex) Delete(id model.RowID) {
	ui.mu.Lock()
	defer ui.mu.Unlock()

	// Remove from inverted index
	if doc, exists := ui.documents[id]; exists {
		ui.removeFromIndexLocked(id, doc)
	}

	// Remove from primary storage
	delete(ui.documents, id)
}

// Len returns the number of documents in the index.
func (ui *UnifiedIndex) Len() int {
	ui.mu.RLock()
	defer ui.mu.RUnlock()

	return len(ui.documents)
}

// ToMap returns a copy of all documents as a map.
// This is useful for serialization and snapshot creation.
func (ui *UnifiedIndex) ToMap() map[model.RowID]metadata.Document {
	ui.mu.RLock()
	defer ui.mu.RUnlock()

	result := make(map[model.RowID]metadata.Document, len(ui.documents))
	for id, iDoc := range ui.documents {
		doc := make(metadata.Document, len(iDoc))
		for k, v := range iDoc {
			doc[k.Value()] = v
		}
		result[id] = doc
	}
	return result
}

// addToIndexLocked adds a document to the inverted index and numeric index.
// Caller must hold ui.mu.Lock().
func (ui *UnifiedIndex) addToIndexLocked(id model.RowID, doc metadata.InternedDocument) {
	for key, value := range doc {
		keyStr := key.Value()

		// Get or create value map for this field
		valueMap, ok := ui.inverted[key]
		if !ok {
			valueMap = make(map[unique.Handle[string]]*LocalBitmap)
			ui.inverted[key] = valueMap
		}

		// Get or create bitmap for this value
		valueKey := unique.Make(value.Key())
		bitmap, ok := valueMap[valueKey]
		if !ok {
			bitmap = NewLocalBitmap()
			valueMap[valueKey] = bitmap
		}

		// Add ID to bitmap
		bitmap.Add(uint32(id))

		// Add to numeric index for fast range queries
		ui.numeric.Add(keyStr, value, id)
	}
}

// removeFromIndexLocked removes a document from the inverted index and numeric index.
// Caller must hold ui.mu.Lock().
func (ui *UnifiedIndex) removeFromIndexLocked(id model.RowID, doc metadata.InternedDocument) {
	for key, value := range doc {
		keyStr := key.Value()

		valueMap, ok := ui.inverted[key]
		if !ok {
			continue
		}

		valueKey := unique.Make(value.Key())
		bitmap, ok := valueMap[valueKey]
		if !ok {
			continue
		}

		// Remove ID from bitmap
		bitmap.Remove(uint32(id))

		// Clean up empty bitmaps
		if bitmap.IsEmpty() {
			delete(valueMap, valueKey)
			if len(valueMap) == 0 {
				delete(ui.inverted, key)
			}
		}

		// Remove from numeric index
		ui.numeric.Remove(keyStr, value, id)
	}
}

// RLock locks the index for reading.
func (ui *UnifiedIndex) RLock() {
	ui.mu.RLock()
}

// RUnlock unlocks the index for reading.
func (ui *UnifiedIndex) RUnlock() {
	ui.mu.RUnlock()
}

// SealNumericIndex seals the numeric index for sorted binary search.
// This must be called after bulk loading with AddInvertedIndex is complete.
// After sealing, numeric range queries use O(log n) binary search instead of O(n) scan.
func (ui *UnifiedIndex) SealNumericIndex() {
	ui.mu.Lock()
	defer ui.mu.Unlock()
	ui.numeric.Seal()
}

// CreateStreamingFilter creates a filter function that checks the index without allocating intermediate bitmaps.
// The caller MUST hold the read lock (RLock) while using the returned function.
// The context is used for provider lookups on fallback paths.
func (ui *UnifiedIndex) CreateStreamingFilter(ctx context.Context, fs *metadata.FilterSet) func(model.RowID) bool {
	if fs == nil || len(fs.Filters) == 0 {
		return func(model.RowID) bool { return true }
	}

	// Pre-resolve bitmaps for supported operators
	checks := make([]func(model.RowID) bool, 0, len(fs.Filters))

	for _, filter := range fs.Filters {
		checks = append(checks, ui.createFilterCheck(ctx, filter))
	}

	return func(id model.RowID) bool {
		for _, check := range checks {
			if !check(id) {
				return false
			}
		}
		return true
	}
}

func (ui *UnifiedIndex) createFilterCheck(ctx context.Context, filter metadata.Filter) func(model.RowID) bool {
	switch filter.Operator {
	case metadata.OpEqual:
		return ui.createEqualCheck(filter)
	case metadata.OpIn:
		return ui.createInCheck(filter)
	case metadata.OpLessThan, metadata.OpLessEqual, metadata.OpGreaterThan, metadata.OpGreaterEqual:
		return ui.createRangeCheck(filter)
	default:
		return ui.createFallbackCheck(ctx, filter)
	}
}

// createRangeCheck creates a check function for range operators using NumericIndex.
// This is O(1) per rowID using pre-captured field data when the index is sealed.
// Falls back to document scanning if the index is not sealed or field not found.
func (ui *UnifiedIndex) createRangeCheck(filter metadata.Filter) func(model.RowID) bool {
	// Try to convert filter value to float64
	var filterVal float64
	switch filter.Value.Kind {
	case metadata.KindInt:
		filterVal = float64(filter.Value.I64)
	case metadata.KindFloat:
		filterVal = filter.Value.F64
	default:
		// Non-numeric value, can't use numeric index
		return func(model.RowID) bool { return false }
	}

	// Get matcher from NumericIndex (O(1) per row)
	// This only works if the index has been sealed (rowIDToIndex populated)
	matcher := ui.numeric.GetFieldMatcher(filter.Key, filter.Operator, filterVal)
	if matcher != nil {
		return func(id model.RowID) bool {
			return matcher(uint32(id))
		}
	}

	// Fallback: Field not in sealed numeric index - use document lookup
	// This happens for in-memory memtable or unsealed segments
	documents := ui.documents
	op := filter.Operator
	key := unique.Make(filter.Key)
	return func(id model.RowID) bool {
		doc, ok := documents[id]
		if !ok {
			return false
		}
		val, exists := doc[key]
		if !exists {
			return false
		}
		// Convert to float64 for comparison
		var docVal float64
		switch val.Kind {
		case metadata.KindInt:
			docVal = float64(val.I64)
		case metadata.KindFloat:
			docVal = val.F64
		default:
			return false
		}
		// Use numeric comparison (same as NumericIndex)
		switch op {
		case metadata.OpLessThan:
			return docVal < filterVal
		case metadata.OpLessEqual:
			return docVal <= filterVal
		case metadata.OpGreaterThan:
			return docVal > filterVal
		case metadata.OpGreaterEqual:
			return docVal >= filterVal
		default:
			return false
		}
	}
}

func (ui *UnifiedIndex) createEqualCheck(filter metadata.Filter) func(model.RowID) bool {
	// Fast path: check bitmap directly
	bitmap := ui.getBitmapLocked(filter.Key, filter.Value)
	if bitmap == nil {
		return func(model.RowID) bool { return false }
	}
	return func(id model.RowID) bool {
		return bitmap.Contains(uint32(id))
	}
}

func (ui *UnifiedIndex) createInCheck(filter metadata.Filter) func(model.RowID) bool {
	// Check if ID is in ANY of the bitmaps
	arr, ok := filter.Value.AsArray()
	if !ok {
		return func(model.RowID) bool { return false }
	}

	var bitmaps []*LocalBitmap
	for _, v := range arr {
		if b := ui.getBitmapLocked(filter.Key, v); b != nil {
			bitmaps = append(bitmaps, b)
		}
	}

	if len(bitmaps) == 0 {
		return func(model.RowID) bool { return false }
	}

	return func(id model.RowID) bool {
		for _, b := range bitmaps {
			if b.Contains(uint32(id)) {
				return true
			}
		}
		return false
	}
}

// GetFilterMatcher returns a pooled FilterMatcher for zero-allocation filter evaluation.
// This is the preferred API for hot paths - uses interface dispatch instead of closures.
//
// Returns:
//   - AlwaysTrueMatcher if fs is nil or empty
//   - AlwaysFalseMatcher if any filter is impossible
//   - CompositeMatcher for multiple filters (AND semantics)
//   - Single matcher for single-filter cases
//
// Caller MUST call Release() on the returned matcher when done.
// The caller MUST hold the read lock (RLock) while using the matcher.
func (ui *UnifiedIndex) GetFilterMatcher(fs *metadata.FilterSet, scratch *MatcherScratch) FilterMatcher {
	if fs == nil || len(fs.Filters) == 0 {
		return GetAlwaysTrueMatcher()
	}

	// Reset scratch
	scratch.Matchers = scratch.Matchers[:0]
	scratch.Bitmaps = scratch.Bitmaps[:0]

	for _, filter := range fs.Filters {
		matcher := ui.createSingleMatcher(filter, scratch)
		// Check for AlwaysFalse - short circuit
		if _, isFalse := matcher.(*AlwaysFalseMatcher); isFalse {
			for _, m := range scratch.Matchers {
				m.Release()
			}
			scratch.Matchers = scratch.Matchers[:0]
			return GetAlwaysFalseMatcher()
		}
		// Skip AlwaysTrue matchers (no-op)
		if _, isTrue := matcher.(*AlwaysTrueMatcher); isTrue {
			continue
		}
		scratch.Matchers = append(scratch.Matchers, matcher)
	}

	// If all filters were AlwaysTrue, return AlwaysTrue
	if len(scratch.Matchers) == 0 {
		return GetAlwaysTrueMatcher()
	}

	// Single filter optimization - return directly without composite wrapper
	if len(scratch.Matchers) == 1 {
		return scratch.Matchers[0]
	}

	// Multiple filters - create composite matcher
	return GetCompositeMatcher(scratch.Matchers)
}

// createSingleMatcher creates a FilterMatcher for a single filter.
// Returns nil if the filter is impossible.
func (ui *UnifiedIndex) createSingleMatcher(filter metadata.Filter, scratch *MatcherScratch) FilterMatcher {
	switch filter.Operator {
	case metadata.OpEqual:
		bitmap := ui.getBitmapLocked(filter.Key, filter.Value)
		if bitmap == nil {
			return GetAlwaysFalseMatcher()
		}
		return GetBitmapMatcher(bitmap)

	case metadata.OpIn:
		arr, ok := filter.Value.AsArray()
		if !ok {
			return GetAlwaysFalseMatcher()
		}
		// Collect bitmaps into scratch
		scratch.Bitmaps = scratch.Bitmaps[:0]
		for _, v := range arr {
			if b := ui.getBitmapLocked(filter.Key, v); b != nil {
				scratch.Bitmaps = append(scratch.Bitmaps, b)
			}
		}
		if len(scratch.Bitmaps) == 0 {
			return GetAlwaysFalseMatcher()
		}
		if len(scratch.Bitmaps) == 1 {
			return GetBitmapMatcher(scratch.Bitmaps[0])
		}
		return GetMultiBitmapMatcher(scratch.Bitmaps)

	case metadata.OpNotEqual:
		// NotEqual is tricky - need to check if NOT in bitmap
		// For now, fall back to closure-based (rare in practice)
		bitmap := ui.getBitmapLocked(filter.Key, filter.Value)
		if bitmap == nil {
			return GetAlwaysTrueMatcher()
		}
		return GetNotBitmapMatcher(bitmap)

	case metadata.OpLessThan, metadata.OpLessEqual, metadata.OpGreaterThan, metadata.OpGreaterEqual:
		if !ui.numeric.HasField(filter.Key) {
			return GetAlwaysFalseMatcher()
		}

		var filterVal float64
		switch filter.Value.Kind {
		case metadata.KindInt:
			filterVal = float64(filter.Value.I64)
		case metadata.KindFloat:
			filterVal = filter.Value.F64
		default:
			return GetAlwaysFalseMatcher()
		}

		matcher := ui.numeric.GetNumericMatcher(filter.Key, filter.Operator, filterVal)
		if matcher == nil {
			return GetAlwaysFalseMatcher()
		}
		return matcher

	default:
		return GetAlwaysFalseMatcher()
	}
}

// FilterCursor returns a push-based cursor for filtered iteration.
// This is the zero-allocation hot path that:
//   - Avoids Roaring bitmap OR operations (no dst.Or(bitmap) calls)
//   - Evaluates filters lazily during iteration
//   - Supports early termination
//
// For equality/IN filters: O(1) bitmap Contains() check per row
// For range filters: O(1) numeric comparison per row
//
// The caller MUST hold the read lock (RLock) while using the cursor.
func (ui *UnifiedIndex) FilterCursor(fs *metadata.FilterSet, rowCount uint32) FilterCursor {
	if fs == nil || len(fs.Filters) == 0 {
		return NewAllCursor(rowCount)
	}

	// Fast path: single equality filter with small bitmap
	// Check this FIRST before building any closures to avoid unnecessary allocations
	if len(fs.Filters) == 1 && fs.Filters[0].Operator == metadata.OpEqual {
		if b := ui.getBitmapLocked(fs.Filters[0].Key, fs.Filters[0].Value); b != nil {
			// For small bitmaps: extract to rows for SIMD-friendly iteration
			// For larger bitmaps: use bitmap iterator directly (avoids allocation)
			if b.Cardinality() <= 4096 {
				rows := make([]uint32, 0, b.Cardinality())
				rows = b.ToArrayInto(rows)
				return NewRowsCursor(rows)
			}
			// Larger bitmap: iterate directly without slice extraction
			return NewBitmapCursor(b)
		}
		return GetEmptyCursor()
	}

	// Build filter checks (same as CreateStreamingFilter but returns cursor)
	checks := make([]func(uint32) bool, 0, len(fs.Filters))
	estimate := int(rowCount)

	for _, filter := range fs.Filters {
		check, selectivity := ui.createFilterCheckWithSelectivity(filter)
		checks = append(checks, check)
		estimate = int(float64(estimate) * selectivity)
		if estimate < 1 {
			estimate = 1
		}
	}

	return &unifiedFilterCursor{
		checks:   checks,
		rowCount: rowCount,
		estimate: estimate,
	}
}

// createFilterCheckWithSelectivity returns a filter check function and estimated selectivity.
func (ui *UnifiedIndex) createFilterCheckWithSelectivity(filter metadata.Filter) (func(uint32) bool, float64) {
	switch filter.Operator {
	case metadata.OpEqual:
		bitmap := ui.getBitmapLocked(filter.Key, filter.Value)
		if bitmap == nil {
			return func(uint32) bool { return false }, 0.0
		}
		selectivity := 0.1 // Default estimate
		if total := uint64(len(ui.documents)); total > 0 {
			selectivity = float64(bitmap.Cardinality()) / float64(total)
		}
		return func(id uint32) bool { return bitmap.Contains(id) }, selectivity

	case metadata.OpIn:
		arr, ok := filter.Value.AsArray()
		if !ok {
			return func(uint32) bool { return false }, 0.0
		}
		var bitmaps []*LocalBitmap
		var totalCard uint64
		for _, v := range arr {
			if b := ui.getBitmapLocked(filter.Key, v); b != nil {
				bitmaps = append(bitmaps, b)
				totalCard += b.Cardinality()
			}
		}
		if len(bitmaps) == 0 {
			return func(uint32) bool { return false }, 0.0
		}
		selectivity := 0.2 // Default estimate
		if total := uint64(len(ui.documents)); total > 0 {
			selectivity = float64(totalCard) / float64(total)
			if selectivity > 1.0 {
				selectivity = 1.0
			}
		}
		return func(id uint32) bool {
			for _, b := range bitmaps {
				if b.Contains(id) {
					return true
				}
			}
			return false
		}, selectivity

	case metadata.OpNotEqual:
		bitmap := ui.getBitmapLocked(filter.Key, filter.Value)
		if bitmap == nil {
			// Field/value doesn't exist, everything matches NotEqual
			return func(uint32) bool { return true }, 1.0
		}
		return func(id uint32) bool { return !bitmap.Contains(id) }, 0.9

	case metadata.OpLessThan, metadata.OpLessEqual, metadata.OpGreaterThan, metadata.OpGreaterEqual:
		// For range filters, use NumericIndex directly
		// This avoids Roaring bitmap OR operations
		if !ui.numeric.HasField(filter.Key) {
			// No numeric index for this field - fall back to provider if available
			if ui.provider == nil {
				return func(uint32) bool { return false }, 0.0
			}
			// Use provider-based evaluation (slower but works without numeric index)
			return ui.createProviderBasedRangeCheck(filter)
		}

		var filterVal float64
		switch filter.Value.Kind {
		case metadata.KindInt:
			filterVal = float64(filter.Value.I64)
		case metadata.KindFloat:
			filterVal = filter.Value.F64
		default:
			return func(uint32) bool { return false }, 0.0
		}

		// Use GetFieldMatcher for zero-allocation per-row check
		// This captures the field reference once, eliminating per-rowID string map lookups
		selectivity := ui.numeric.EstimateSelectivity(filter.Key, filter.Operator, filterVal)
		matcher := ui.numeric.GetFieldMatcher(filter.Key, filter.Operator, filterVal)
		if matcher == nil {
			return func(uint32) bool { return false }, 0.0
		}
		return matcher, selectivity

	default:
		return func(uint32) bool { return false }, 0.0
	}
}

// createProviderBasedRangeCheck creates a range check using the document provider.
// This is slower than MatchRowID but works when numeric index is unavailable.
func (ui *UnifiedIndex) createProviderBasedRangeCheck(filter metadata.Filter) (func(uint32) bool, float64) {
	if ui.provider == nil {
		return func(uint32) bool { return false }, 0.0
	}

	var filterVal float64
	switch filter.Value.Kind {
	case metadata.KindInt:
		filterVal = float64(filter.Value.I64)
	case metadata.KindFloat:
		filterVal = filter.Value.F64
	default:
		return func(uint32) bool { return false }, 0.0
	}

	return func(id uint32) bool {
		doc, ok := ui.provider(context.Background(), model.RowID(id))
		if !ok || doc == nil {
			return false
		}
		val, exists := doc[filter.Key]
		if !exists {
			return false
		}
		var numVal float64
		switch val.Kind {
		case metadata.KindInt:
			numVal = float64(val.I64)
		case metadata.KindFloat:
			numVal = val.F64
		default:
			return false
		}
		switch filter.Operator {
		case metadata.OpLessThan:
			return numVal < filterVal
		case metadata.OpLessEqual:
			return numVal <= filterVal
		case metadata.OpGreaterThan:
			return numVal > filterVal
		case metadata.OpGreaterEqual:
			return numVal >= filterVal
		default:
			return false
		}
	}, 0.5
}

// unifiedFilterCursor iterates over row IDs, applying filter checks lazily.
type unifiedFilterCursor struct {
	checks   []func(uint32) bool
	rowCount uint32
	estimate int
}

func (c *unifiedFilterCursor) ForEach(fn func(rowID uint32) bool) {
	for id := uint32(0); id < c.rowCount; id++ {
		// Apply all filter checks (AND logic, short-circuit on first false)
		match := true
		for _, check := range c.checks {
			if !check(id) {
				match = false
				break
			}
		}
		if match {
			if !fn(id) {
				return
			}
		}
	}
}

func (c *unifiedFilterCursor) EstimateCardinality() int { return c.estimate }
func (c *unifiedFilterCursor) IsEmpty() bool            { return c.rowCount == 0 }
func (c *unifiedFilterCursor) IsAll() bool              { return len(c.checks) == 0 }

// -----------------------------------------------------------------------------
// MatcherCursor - Zero-allocation cursor using FilterMatcher interface
// -----------------------------------------------------------------------------

// MatcherCursor iterates over row IDs using a FilterMatcher for zero-allocation evaluation.
// This is the preferred cursor for hot paths - eliminates closure allocations.
type MatcherCursor struct {
	matcher  FilterMatcher
	rowCount uint32
	estimate int
}

var matcherCursorPool = sync.Pool{
	New: func() any { return &MatcherCursor{} },
}

// NewMatcherCursor creates a cursor from a FilterMatcher.
// Caller MUST call Release() when done.
func NewMatcherCursor(matcher FilterMatcher, rowCount uint32, estimate int) *MatcherCursor {
	c := matcherCursorPool.Get().(*MatcherCursor)
	c.matcher = matcher
	c.rowCount = rowCount
	c.estimate = estimate
	return c
}

func (c *MatcherCursor) ForEach(fn func(rowID uint32) bool) {
	for id := uint32(0); id < c.rowCount; id++ {
		if c.matcher.Matches(id) {
			if !fn(id) {
				return
			}
		}
	}
}

func (c *MatcherCursor) EstimateCardinality() int { return c.estimate }
func (c *MatcherCursor) IsEmpty() bool            { return c.rowCount == 0 }
func (c *MatcherCursor) IsAll() bool {
	_, isAll := c.matcher.(*AlwaysTrueMatcher)
	return isAll
}

// Release returns the cursor and its matcher to pools.
func (c *MatcherCursor) Release() {
	if c.matcher != nil {
		c.matcher.Release()
		c.matcher = nil
	}
	c.rowCount = 0
	c.estimate = 0
	matcherCursorPool.Put(c)
}

// FilterCursorWithMatcher returns a zero-allocation cursor using FilterMatcher.
// This is the preferred API for hot paths.
//
// Returns:
//   - cursor: The filter cursor (MUST call cursor.Release() when done)
//   - estimate: Estimated cardinality
//
// The caller MUST hold the read lock (RLock) while using the cursor.
func (ui *UnifiedIndex) FilterCursorWithMatcher(fs *metadata.FilterSet, rowCount uint32, scratch *MatcherScratch) *MatcherCursor {
	if fs == nil || len(fs.Filters) == 0 {
		return NewMatcherCursor(GetAlwaysTrueMatcher(), rowCount, int(rowCount))
	}

	// Get matcher (handles all filter types)
	matcher := ui.GetFilterMatcher(fs, scratch)

	// Estimate cardinality based on matcher type
	estimate := int(rowCount)
	if _, isFalse := matcher.(*AlwaysFalseMatcher); isFalse {
		estimate = 0
	} else if _, isTrue := matcher.(*AlwaysTrueMatcher); !isTrue {
		// For real filters, estimate based on filter count
		// Each filter typically reduces by ~0.3x (assumption)
		for range fs.Filters {
			estimate = int(float64(estimate) * 0.3)
			if estimate < 1 {
				estimate = 1
			}
		}
	}

	return NewMatcherCursor(matcher, rowCount, estimate)
}

// Query evaluates a filter set and returns a bitmap of matching documents.
// Returns nil if the filter set matches all documents (empty filter).
// Returns error if the filter contains unsupported operators (e.g. range queries).
func (ui *UnifiedIndex) Query(fs *metadata.FilterSet) (*LocalBitmap, error) {
	ui.mu.RLock()
	defer ui.mu.RUnlock()

	if fs == nil || len(fs.Filters) == 0 {
		return nil, nil
	}

	var result *LocalBitmap

	for _, f := range fs.Filters {
		var current *LocalBitmap
		var err error

		switch f.Operator {
		case metadata.OpEqual:
			b := ui.getBitmapLocked(f.Key, f.Value)
			if b != nil {
				current = b.Clone()
			} else {
				// No match for this filter -> Empty result
				return NewLocalBitmap(), nil
			}
		case metadata.OpIn:
			current, err = ui.queryInLocked(f)
			if err != nil {
				return nil, err
			}
		default:
			return nil, fmt.Errorf("operator %s not supported in bitmap index", f.Operator)
		}

		if result == nil {
			result = current
		} else {
			result.And(current)
		}

		if result.IsEmpty() {
			return result, nil
		}
	}

	return result, nil
}

func (ui *UnifiedIndex) queryInLocked(f metadata.Filter) (*LocalBitmap, error) {
	arr, ok := f.Value.AsArray()
	if !ok {
		// If value is not an array, treat as Equal? Or error?
		// Logic suggests OpIn expects Array.
		return NewLocalBitmap(), nil
	}

	result := NewLocalBitmap()
	for _, v := range arr {
		if b := ui.getBitmapLocked(f.Key, v); b != nil {
			result.Or(b)
		}
	}
	return result, nil
}

func (ui *UnifiedIndex) createFallbackCheck(ctx context.Context, filter metadata.Filter) func(model.RowID) bool {
	// Fallback for unsupported operators: check document
	// This is slow but necessary
	fs := metadata.NewFilterSet(filter)
	return func(id model.RowID) bool {
		doc, ok := ui.documents[id]
		if ok {
			return fs.MatchesInterned(doc)
		}
		if ui.provider != nil {
			d, ok := ui.provider(ctx, id)
			if ok {
				return filter.Matches(d)
			}
		}
		return false
	}
}

// CompileFilter compiles a FilterSet into a fast bitmap-based filter.
// Returns a bitmap of matching IDs, or nil if compilation fails.
//
// Supported operators:
//   - OpEqual: field == value
//   - OpIn: field IN (value1, value2, ...)
//   - OpNotEqual, OpGreaterThan, etc.: Falls back to scanning
func (ui *UnifiedIndex) CompileFilter(fs *metadata.FilterSet) *LocalBitmap {
	dst := NewLocalBitmap()
	if ui.CompileFilterTo(fs, dst) {
		return dst
	}
	return nil
}

// CompileFilterTo compiles a FilterSet into the provided destination bitmap.
// Returns true if compilation succeeded (all operators supported), false otherwise.
// The destination bitmap is cleared before use.
func (ui *UnifiedIndex) CompileFilterTo(fs *metadata.FilterSet, dst *LocalBitmap) bool {
	if fs == nil || len(fs.Filters) == 0 {
		dst.Clear()
		return true
	}

	ui.mu.RLock()
	defer ui.mu.RUnlock()

	dst.Clear()
	first := true

	for _, filter := range fs.Filters {
		if !ui.applyFilterToBitmap(filter, dst, first) {
			return false
		}

		if !first && dst.IsEmpty() {
			return true
		}
		first = false
	}

	return true
}

func (ui *UnifiedIndex) applyFilterToBitmap(filter metadata.Filter, dst *LocalBitmap, first bool) bool {
	switch filter.Operator {
	case metadata.OpEqual:
		return ui.applyEqualFilter(filter, dst, first)
	case metadata.OpIn:
		return ui.applyInFilter(filter, dst, first)
	default:
		return false
	}
}

func (ui *UnifiedIndex) applyEqualFilter(filter metadata.Filter, dst *LocalBitmap, first bool) bool {
	bitmap := ui.getBitmapLocked(filter.Key, filter.Value)
	if bitmap == nil {
		dst.Clear()
		return true // No matches for this filter -> AND implies 0 matches
	}

	if first {
		dst.Or(bitmap) // Copy first bitmap
	} else {
		dst.And(bitmap)
	}
	return true
}

func (ui *UnifiedIndex) applyInFilter(filter metadata.Filter, dst *LocalBitmap, first bool) bool {
	arr, ok := filter.Value.AsArray()
	if !ok {
		return false
	}

	if first {
		for _, v := range arr {
			if b := ui.getBitmapLocked(filter.Key, v); b != nil {
				dst.Or(b)
			}
		}
	} else {
		// We need to intersect dst with (Union of values)
		// dst = dst AND (v1 OR v2 OR ...)
		scratch := GetPooledBitmap() // Use pooled bitmap
		for _, v := range arr {
			if b := ui.getBitmapLocked(filter.Key, v); b != nil {
				scratch.Or(b)
			}
		}
		dst.And(scratch)
		PutPooledBitmap(scratch) // Return to pool
	}
	return true
}

// getBitmapLocked retrieves the bitmap for a specific field=value combination.
// Returns nil if no matches exist. Caller must hold ui.mu.RLock().
func (ui *UnifiedIndex) getBitmapLocked(key string, value metadata.Value) *LocalBitmap {
	valueMap, ok := ui.inverted[unique.Make(key)]
	if !ok {
		return nil
	}

	bitmap, ok := valueMap[unique.Make(value.Key())]
	if !ok {
		return nil
	}

	return bitmap
}

// getFieldCardinality returns the number of distinct values for a field.
// This is O(1) - just the map length. Caller must hold ui.mu.RLock().
func (ui *UnifiedIndex) getFieldCardinality(key string) int {
	valueMap, ok := ui.inverted[unique.Make(key)]
	if !ok {
		return 0
	}
	return len(valueMap)
}

// ScanFilter evaluates a FilterSet by scanning all documents.
// This is slower than CompileFilter but supports all operators.
// Use this as a fallback when CompileFilter returns nil.
func (ui *UnifiedIndex) ScanFilter(fs *metadata.FilterSet) []model.RowID {
	if fs == nil {
		return nil
	}

	ui.mu.RLock()
	defer ui.mu.RUnlock()

	result := make([]model.RowID, 0, len(ui.documents))

	for id, doc := range ui.documents {
		if fs.MatchesInterned(doc) {
			result = append(result, id)
		}
	}

	return result
}

// CreateFilterFunc creates a filter function from a FilterSet.
// This is used by hybrid search to efficiently test membership.
//
// Returns:
//   - Fast path: If compilation succeeds, returns bitmap-based O(1) lookup
//   - Slow path: Falls back to scanning + evaluating each document (locks per call)
func (ui *UnifiedIndex) CreateFilterFunc(fs *metadata.FilterSet) func(model.RowID) bool {
	if fs == nil || len(fs.Filters) == 0 {
		return nil
	}

	// Try fast path: compile to bitmap
	bitmap := ui.CompileFilter(fs)
	if bitmap != nil {
		// Fast bitmap-based lookup (O(1) average case)
		return func(id model.RowID) bool {
			return bitmap.Contains(uint32(id))
		}
	}

	// Slow path: evaluate filter for each ID
	// Lock per call to avoid holding lock across function lifetime
	return func(id model.RowID) bool {
		ui.mu.RLock()
		doc, ok := ui.documents[id]
		ui.mu.RUnlock()
		if !ok {
			return false
		}
		return fs.MatchesInterned(doc)
	}
}

// Stats returns statistics about the unified index.
type Stats struct {
	DocumentCount    int    // Total documents
	FieldCount       int    // Number of indexed fields
	BitmapCount      int    // Total number of bitmaps
	TotalCardinality uint64 // Sum of all bitmap cardinalities
	MemoryBytes      uint64 // Estimated memory usage
}

// GetStats returns statistics about the index.
func (ui *UnifiedIndex) GetStats() Stats {
	ui.mu.RLock()
	defer ui.mu.RUnlock()

	stats := Stats{
		DocumentCount: len(ui.documents),
		FieldCount:    len(ui.inverted),
	}

	for _, valueMap := range ui.inverted {
		for _, bitmap := range valueMap {
			stats.BitmapCount++
			stats.TotalCardinality += bitmap.Cardinality()
			stats.MemoryBytes += bitmap.GetSizeInBytes()
		}
	}

	return stats
}

// WriteInvertedIndex writes the inverted index to the writer.
func (ui *UnifiedIndex) WriteInvertedIndex(w io.Writer) error {
	ui.mu.RLock()
	defer ui.mu.RUnlock()

	// Write field count
	buf := make([]byte, binary.MaxVarintLen64)
	n := binary.PutUvarint(buf, uint64(len(ui.inverted)))
	if _, err := w.Write(buf[:n]); err != nil {
		return err
	}

	for fieldKey, valueMap := range ui.inverted {
		// Write field key
		keyStr := fieldKey.Value()
		n = binary.PutUvarint(buf, uint64(len(keyStr)))
		if _, err := w.Write(buf[:n]); err != nil {
			return err
		}
		if _, err := io.WriteString(w, keyStr); err != nil {
			return err
		}

		// Write value count
		n = binary.PutUvarint(buf, uint64(len(valueMap)))
		if _, err := w.Write(buf[:n]); err != nil {
			return err
		}

		for valueKey, bitmap := range valueMap {
			// Write value key
			valStr := valueKey.Value()
			n = binary.PutUvarint(buf, uint64(len(valStr)))
			if _, err := w.Write(buf[:n]); err != nil {
				return err
			}
			if _, err := io.WriteString(w, valStr); err != nil {
				return err
			}

			// Write bitmap
			if _, err := bitmap.WriteTo(w); err != nil {
				return err
			}
		}
	}
	return nil
}

// ReadInvertedIndex reads the inverted index from the reader.
func (ui *UnifiedIndex) ReadInvertedIndex(r io.Reader) error {
	ui.mu.Lock()
	defer ui.mu.Unlock()

	// Read field count
	count, err := readUvarint(r)
	if err != nil {
		return err
	}

	for range count {
		// Read field key
		keyLen, err := readUvarint(r)
		if err != nil {
			return err
		}
		keyBytes := make([]byte, keyLen)
		if _, err := io.ReadFull(r, keyBytes); err != nil {
			return err
		}
		fieldKey := unique.Make(string(keyBytes))

		// Read value count
		valCount, err := readUvarint(r)
		if err != nil {
			return err
		}

		valueMap := make(map[unique.Handle[string]]*LocalBitmap, valCount)
		ui.inverted[fieldKey] = valueMap

		for range valCount {
			// Read value key
			valLen, err := readUvarint(r)
			if err != nil {
				return err
			}
			valBytes := make([]byte, valLen)
			if _, err := io.ReadFull(r, valBytes); err != nil {
				return err
			}
			valueKey := unique.Make(string(valBytes))

			// Read bitmap
			bitmap := NewLocalBitmap()
			if _, err := bitmap.ReadFrom(r); err != nil {
				return err
			}
			valueMap[valueKey] = bitmap
		}
	}
	return nil
}

func readUvarint(r io.Reader) (uint64, error) {
	// Use binary.ReadUvarint if r implements ByteReader, otherwise wrap it
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
			if i > 9 || i == 9 && b[0] > 1 {
				return 0, io.ErrUnexpectedEOF // overflow
			}
			return x | uint64(b[0])<<s, nil
		}
		x |= uint64(b[0]&0x7f) << s
		s += 7
	}
}
