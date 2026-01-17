package imetadata

import (
	"io"
	"iter"
	"slices"
	"sync"

	"github.com/RoaringBitmap/roaring/v2"
	"github.com/hupe1980/vecgo/internal/bitmap"
	"github.com/hupe1980/vecgo/model"
)

// LocalBitmap implements a 32-bit Roaring Bitmap.
// It wraps the official roaring implementation.
// Used for internal row filtering (RowID).
type LocalBitmap struct {
	rb *roaring.Bitmap
}

// bitmapPool is a sync.Pool for reusing LocalBitmap instances.
// This reduces allocations in filtered search hot paths.
var bitmapPool = sync.Pool{
	New: func() any {
		return &LocalBitmap{
			rb: roaring.New(),
		}
	},
}

// NewLocalBitmap creates a new empty local bitmap.
// The returned bitmap is owned by the caller and should NOT be returned to the pool.
// Use this for long-lived bitmaps (e.g., inverted index storage, API boundaries).
// For temporary bitmaps in hot paths, use GetPooledBitmap instead.
func NewLocalBitmap() *LocalBitmap {
	return &LocalBitmap{
		rb: roaring.New(),
	}
}

// GetPooledBitmap gets a bitmap from the pool for temporary use.
// IMPORTANT: Caller MUST call PutPooledBitmap when done to avoid pool exhaustion.
// The bitmap is cleared before being returned.
// Use this for temporary bitmaps in hot paths (e.g., filter evaluation).
// For owned bitmaps that outlive a function call, use NewLocalBitmap instead.
func GetPooledBitmap() *LocalBitmap {
	b := bitmapPool.Get().(*LocalBitmap)
	b.rb.Clear()
	return b
}

// PutPooledBitmap returns a pooled bitmap to the pool.
// Only call this for bitmaps obtained via GetPooledBitmap.
// Passing nil is safe and will be ignored.
// WARNING: Do not use the bitmap after calling this function.
func PutPooledBitmap(b *LocalBitmap) {
	if b == nil {
		return
	}
	// No clear needed here - GetPooledBitmap clears on retrieval
	bitmapPool.Put(b)
}

// Add adds a RowID to the bitmap.
func (b *LocalBitmap) Add(id uint32) {
	b.rb.Add(id)
}

// Remove removes a RowID from the bitmap.
func (b *LocalBitmap) Remove(id uint32) {
	b.rb.Remove(id)
}

// Contains checks if a RowID is in the bitmap.
func (b *LocalBitmap) Contains(id uint32) bool {
	return b.rb.Contains(id)
}

// ForEach iterates over the bitmap.
func (b *LocalBitmap) ForEach(fn func(id uint32) bool) {
	it := b.rb.Iterator()
	for it.HasNext() {
		if !fn(it.Next()) {
			break
		}
	}
}

// ToArray returns all elements in the bitmap as a slice.
// This allocates a new slice. For zero-alloc extraction, use ToArrayInto.
func (b *LocalBitmap) ToArray() []uint32 {
	return b.rb.ToArray()
}

// ToArrayInto copies all elements into the provided buffer, returning the populated slice.
// If dst has insufficient capacity, a new slice is allocated and returned.
// This is the zero-alloc path for extracting rowIDs in hot loops.
//
// Example:
//
//	rowIDs := b.ToArrayInto(scratch[:0])
func (b *LocalBitmap) ToArrayInto(dst []uint32) []uint32 {
	card := int(b.rb.GetCardinality())
	if cap(dst) < card {
		dst = make([]uint32, card)
	} else {
		dst = dst[:card]
	}
	// Use roaring's ManyIterator for bulk extraction (faster than Iterator)
	it := b.rb.ManyIterator()
	written := 0
	for {
		n := it.NextMany(dst[written:])
		if n == 0 {
			break
		}
		written += n
	}
	return dst[:written]
}

// IsEmpty returns true if the bitmap is empty.
func (b *LocalBitmap) IsEmpty() bool {
	return b.rb.IsEmpty()
}

// Cardinality returns the number of elements in the bitmap.
func (b *LocalBitmap) Cardinality() uint64 {
	return b.rb.GetCardinality()
}

// Clone returns a deep copy of the bitmap.
// The returned bitmap is owned (not pooled) and allocates a new roaring.Bitmap.
// For hot paths where you want to avoid allocation, use CloneTo with a pooled bitmap.
func (b *LocalBitmap) Clone() *LocalBitmap {
	return &LocalBitmap{
		rb: b.rb.Clone(),
	}
}

// CloneTo copies this bitmap's contents into the destination bitmap.
// The destination is cleared first, then filled via Or operation.
// This avoids roaring's internal Clone allocation when used with pooled bitmaps.
// Example:
//
//	dst := GetPooledBitmap()
//	src.CloneTo(dst)
//	// use dst...
//	PutPooledBitmap(dst)
func (b *LocalBitmap) CloneTo(dst *LocalBitmap) {
	dst.rb.Clear()
	dst.rb.Or(b.rb)
}

// Iterator returns an iterator over the bitmap.
func (b *LocalBitmap) Iterator() iter.Seq[model.RowID] {
	return func(yield func(model.RowID) bool) {
		it := b.rb.Iterator()
		for it.HasNext() {
			if !yield(model.RowID(it.Next())) {
				return
			}
		}
	}
}

// And computes the intersection of two bitmaps.
func (b *LocalBitmap) And(other *LocalBitmap) {
	b.rb.And(other.rb)
}

// AndNot computes the difference: b = b AND NOT other.
// This removes all elements in other from b.
func (b *LocalBitmap) AndNot(other *LocalBitmap) {
	b.rb.AndNot(other.rb)
}

// Or computes the union of two bitmaps.
func (b *LocalBitmap) Or(other *LocalBitmap) {
	b.rb.Or(other.rb)
}

// OrMany computes the union of this bitmap with multiple other bitmaps.
// Uses roaring.FastOr for optimal performance when combining many bitmaps.
// This is significantly faster than calling Or() in a loop for 3+ bitmaps.
func (b *LocalBitmap) OrMany(others []*LocalBitmap) {
	if len(others) == 0 {
		return
	}
	if len(others) == 1 {
		b.rb.Or(others[0].rb)
		return
	}

	// Collect all bitmaps including self
	bitmaps := make([]*roaring.Bitmap, 0, len(others)+1)
	bitmaps = append(bitmaps, b.rb)
	for _, other := range others {
		bitmaps = append(bitmaps, other.rb)
	}

	// FastOr is optimized for combining many bitmaps at once
	result := roaring.FastOr(bitmaps...)
	b.rb = result
}

// FastOrMany creates a new bitmap that is the union of all provided bitmaps.
// Returns nil if the input slice is empty.
// This is a static helper that doesn't modify any input bitmaps.
func FastOrMany(bitmaps []*LocalBitmap) *LocalBitmap {
	if len(bitmaps) == 0 {
		return nil
	}
	if len(bitmaps) == 1 {
		return bitmaps[0].Clone()
	}

	rbs := make([]*roaring.Bitmap, len(bitmaps))
	for i, b := range bitmaps {
		rbs[i] = b.rb
	}

	return &LocalBitmap{rb: roaring.FastOr(rbs...)}
}

// AddMany adds multiple RowIDs to the bitmap in batch.
// This is more efficient than calling Add repeatedly.
func (b *LocalBitmap) AddMany(ids []uint32) {
	b.rb.AddMany(ids)
}

// Clear removes all elements from the bitmap.
func (b *LocalBitmap) Clear() {
	b.rb.Clear()
}

// RunOptimize attempts to further compress the bitmap by converting containers
// to run-length encoding where beneficial. This should be called after the
// bitmap is fully built and before it's used in queries.
//
// This is especially effective for:
//   - Bitmaps with consecutive ranges (e.g., prefix cumulative bitmaps)
//   - Bitmaps built from sorted data
//   - Dense regions followed by sparse regions
//
// Trade-off: Slightly slower to build, but faster queries and smaller memory.
func (b *LocalBitmap) RunOptimize() {
	b.rb.RunOptimize()
}

// GetSizeInBytes returns the size of the bitmap in bytes.
func (b *LocalBitmap) GetSizeInBytes() uint64 {
	return b.rb.GetSizeInBytes()
}

// WriteTo writes the bitmap to an io.Writer.
func (b *LocalBitmap) WriteTo(w io.Writer) (int64, error) {
	return b.rb.WriteTo(w)
}

// ReadFrom reads the bitmap from an io.Reader.
func (b *LocalBitmap) ReadFrom(r io.Reader) (int64, error) {
	return b.rb.ReadFrom(r)
}

// DefaultUniverseSize is the default size for query-time bitmaps.
// 1M covers typical segment sizes; larger segments trigger reallocation.
const DefaultUniverseSize = 1 << 20

// QueryScratch provides reusable scratch space for query-local bitmap operations.
// This reduces allocations by reusing the same bitmaps across filter stages
// within a single query, rather than getting/putting from the pool per stage.
//
// Architecture:
//   - Tmp1/Tmp2: QueryBitmap (SIMD, zero-alloc) for execution-time operations
//   - TmpStorage: LocalBitmap (roaring) for storage-layer bridge operations
//
// The separation ensures Roaring never appears in hot execution paths while
// still supporting storage-layer operations that output to LocalBitmap.
//
// Usage:
//
//	qs := GetQueryScratch()
//	defer PutQueryScratch(qs)
//	// use qs.Tmp1, qs.Tmp2 for SIMD operations
//	// use qs.TmpStorage for storage-layer bridge
type QueryScratch struct {
	// Tmp1 is a SIMD-friendly scratch bitmap for intermediate results.
	// This is the execution-time bitmap - NOT roaring.
	Tmp1 *bitmap.QueryBitmap
	// Tmp2 is a second SIMD-friendly scratch bitmap for complex operations.
	Tmp2 *bitmap.QueryBitmap
	// TmpStorage is a roaring-based bitmap for storage-layer operations.
	// Used as a bridge between storage (LocalBitmap) and execution (QueryBitmap).
	// Example: NumericIndex.EvaluateFilterInto outputs here, then convert to QueryBitmap.
	TmpStorage *LocalBitmap
	// TmpRowIDs is a reusable buffer for collecting rowIDs
	TmpRowIDs []uint32
	// TmpIndices is a reusable buffer for SIMD index operations (int32 for SIMD compatibility)
	TmpIndices []int32
}

// queryScratchPool provides reusable QueryScratch instances.
var queryScratchPool = sync.Pool{
	New: func() any {
		return &QueryScratch{
			Tmp1:       bitmap.New(DefaultUniverseSize),
			Tmp2:       bitmap.New(DefaultUniverseSize),
			TmpStorage: &LocalBitmap{rb: roaring.New()},
			TmpRowIDs:  make([]uint32, 0, 1024),
			TmpIndices: make([]int32, 0, 1024),
		}
	},
}

// GetQueryScratch gets a QueryScratch from the pool.
// All scratch buffers are cleared before being returned.
// Caller MUST call PutQueryScratch when done.
func GetQueryScratch() *QueryScratch {
	qs := queryScratchPool.Get().(*QueryScratch)
	qs.Tmp1.Clear()
	qs.Tmp2.Clear()
	qs.TmpStorage.rb.Clear()
	qs.TmpRowIDs = qs.TmpRowIDs[:0]
	qs.TmpIndices = qs.TmpIndices[:0]
	return qs
}

// PutQueryScratch returns a QueryScratch to the pool.
// Passing nil is safe and will be ignored.
func PutQueryScratch(qs *QueryScratch) {
	if qs == nil {
		return
	}
	// Clear before returning (fast for QueryBitmap - only touches active blocks)
	qs.Tmp1.Clear()
	qs.Tmp2.Clear()
	qs.TmpStorage.rb.Clear()
	qs.TmpRowIDs = qs.TmpRowIDs[:0]
	qs.TmpIndices = qs.TmpIndices[:0]
	queryScratchPool.Put(qs)
}

// ==============================================================================
// FilterResult: Dual-mode filter result (rows vs bitmap)
// ==============================================================================
//
// FilterResult uses QueryBitmap (SIMD, zero-alloc) for execution-time operations.
// This is the same pattern used in DuckDB, ClickHouse, and Vespa but with
// SIMD acceleration.
//
// Ownership Model:
//   - FilterResult does not allocate
//   - It borrows memory from QueryScratch
//   - Lifetime = query
//   - No pooling, no GC churn

// FilterMode indicates the representation mode of a FilterResult.
type FilterMode uint8

const (
	// FilterNone indicates empty result (no matches).
	FilterNone FilterMode = iota
	// FilterAll indicates no filtering - all rows match.
	// This is semantically equivalent to a nil bitmap in the old API.
	FilterAll
	// FilterRows indicates result is stored as a sorted []uint32.
	FilterRows
	// FilterBitmap indicates result is stored as a QueryBitmap (SIMD, zero-alloc).
	FilterBitmap
	// FilterRange indicates result is a contiguous range [start, end).
	// Extremely efficient for temporal/sequential data.
	FilterRange
)

// FilterResult is a dual-mode filter result that avoids allocations.
// For low cardinality, it stores rows as []uint32 (SIMD-friendly).
// For high cardinality, it uses QueryBitmap (SIMD-accelerated, zero-alloc).
// For contiguous ranges, it stores [start, end) bounds (zero storage overhead).
//
// IMPORTANT: FilterResult now uses QueryBitmap instead of roaring.
// This removes Roaring from all query hot paths.
//
// Critical: FilterResult does NOT own its memory.
// The rows slice and bitmap pointer are borrowed from QueryScratch.
// Lifetime is query-scoped.
type FilterResult struct {
	mode       FilterMode
	rows       []uint32            // borrowed from QueryScratch.TmpRowIDs
	qbm        *bitmap.QueryBitmap // borrowed from QueryScratch.Tmp1 (SIMD, zero-alloc)
	rangeStart uint32              // for FilterRange: inclusive start
	rangeEnd   uint32              // for FilterRange: exclusive end
}

// RowsResult creates a FilterResult from a []uint32 slice.
// The slice is NOT copied - caller must ensure it outlives the FilterResult.
// Zero allocations.
func RowsResult(rows []uint32) FilterResult {
	if len(rows) == 0 {
		return FilterResult{mode: FilterNone}
	}
	return FilterResult{
		mode: FilterRows,
		rows: rows,
	}
}

// QueryBitmapResult creates a FilterResult from a QueryBitmap.
// The bitmap is NOT copied - caller must ensure it outlives the FilterResult.
// Zero allocations.
func QueryBitmapResult(qb *bitmap.QueryBitmap) FilterResult {
	if qb == nil || qb.IsEmpty() {
		return FilterResult{mode: FilterNone}
	}
	return FilterResult{
		mode: FilterBitmap,
		qbm:  qb,
	}
}

// RangeResult creates a FilterResult from a contiguous range [start, end).
// This is extremely efficient for sequential/temporal data where many
// consecutive rows match. Zero storage overhead beyond two integers.
// Returns FilterNone if start >= end.
func RangeResult(start, end uint32) FilterResult {
	if start >= end {
		return FilterResult{mode: FilterNone}
	}
	return FilterResult{
		mode:       FilterRange,
		rangeStart: start,
		rangeEnd:   end,
	}
}

// AdaptiveResult creates a FilterResult choosing the optimal representation
// based on cardinality and data characteristics. This is the recommended
// constructor when the optimal representation is not known in advance.
//
// Thresholds:
//   - Empty → FilterNone
//   - 1 element → FilterRows (most cache-friendly)
//   - ≤1024 elements → FilterRows (fits in L1 cache for iteration)
//   - Contiguous range → FilterRange (zero storage overhead)
//   - >1024 elements → FilterBitmap (SIMD-accelerated ops)
//
// The segmentSize parameter enables range detection and FilterAll optimization.
func AdaptiveResult(rows []uint32, segmentSize uint32) FilterResult {
	n := len(rows)

	// Empty
	if n == 0 {
		return EmptyResult()
	}

	// All rows match (common for unfiltered queries)
	if segmentSize > 0 && uint32(n) == segmentSize {
		return AllResult()
	}

	// Small cardinality: always use rows (cache-friendly)
	const rowsThreshold = 1024
	if n <= rowsThreshold {
		return RowsResult(rows)
	}

	// Check if contiguous range (common for temporal data)
	if isContiguousRange(rows) {
		return RangeResult(rows[0], rows[n-1]+1)
	}

	// Large cardinality: would need bitmap but caller must provide QueryBitmap
	// Fall back to rows (caller can upgrade to bitmap if needed)
	return RowsResult(rows)
}

// isContiguousRange checks if rows form a contiguous sequence.
// O(1) check using first, last, and count.
func isContiguousRange(rows []uint32) bool {
	if len(rows) < 2 {
		return true
	}
	expected := rows[len(rows)-1] - rows[0] + 1
	return expected == uint32(len(rows))
}

// EmptyResult creates an empty FilterResult.
func EmptyResult() FilterResult {
	return FilterResult{mode: FilterNone}
}

// AllResult creates a FilterResult that matches all rows.
// This indicates no filtering should be applied (equivalent to nil bitmap in old API).
func AllResult() FilterResult {
	return FilterResult{mode: FilterAll}
}

// IsRange returns true if the result is stored as a range.
func (fr FilterResult) IsRange() bool {
	return fr.mode == FilterRange
}

// Range returns the range bounds (only valid for FilterRange mode).
// Returns (0, 0) for other modes.
func (fr FilterResult) Range() (start, end uint32) {
	if fr.mode == FilterRange {
		return fr.rangeStart, fr.rangeEnd
	}
	return 0, 0
}

// Mode returns the current mode of the FilterResult.
func (fr FilterResult) Mode() FilterMode {
	return fr.mode
}

// IsEmpty returns true if the result contains no matches.
func (fr FilterResult) IsEmpty() bool {
	return fr.mode == FilterNone
}

// IsAll returns true if the result matches all rows (no filtering).
// This is semantically equivalent to a nil bitmap in the old API.
func (fr FilterResult) IsAll() bool {
	return fr.mode == FilterAll
}

// Cardinality returns the number of matching row IDs.
func (fr FilterResult) Cardinality() int {
	switch fr.mode {
	case FilterRows:
		return len(fr.rows)
	case FilterBitmap:
		return fr.qbm.Cardinality()
	case FilterRange:
		return int(fr.rangeEnd - fr.rangeStart)
	default:
		return 0
	}
}

// CardinalityUint64 returns the cardinality as uint64 (for segment.Bitmap interface).
func (fr FilterResult) CardinalityUint64() uint64 {
	switch fr.mode {
	case FilterRows:
		return uint64(len(fr.rows))
	case FilterBitmap:
		return uint64(fr.qbm.Cardinality())
	case FilterRange:
		return uint64(fr.rangeEnd - fr.rangeStart)
	default:
		return 0
	}
}

// Rows returns the underlying rows slice (only valid for FilterRows mode).
// Returns nil for other modes.
func (fr FilterResult) Rows() []uint32 {
	if fr.mode == FilterRows {
		return fr.rows
	}
	return nil
}

// QueryBitmap returns the underlying QueryBitmap (only valid for FilterBitmap mode).
// Returns nil for other modes.
func (fr FilterResult) QueryBitmap() *bitmap.QueryBitmap {
	if fr.mode == FilterBitmap {
		return fr.qbm
	}
	return nil
}

// Clone creates an independent copy of the FilterResult.
// For FilterRows mode, this copies the underlying slice to prevent
// aliasing issues when the original slice is reused.
// For FilterBitmap mode, this clones the QueryBitmap.
// For FilterRange mode, returns the same value (no data to copy).
// For FilterNone/FilterAll, returns the same value (no data to copy).
func (fr FilterResult) Clone() FilterResult {
	switch fr.mode {
	case FilterRows:
		if len(fr.rows) == 0 {
			return FilterResult{mode: FilterNone}
		}
		copied := make([]uint32, len(fr.rows))
		copy(copied, fr.rows)
		return FilterResult{mode: FilterRows, rows: copied}
	case FilterBitmap:
		if fr.qbm == nil {
			return FilterResult{mode: FilterNone}
		}
		return FilterResult{mode: FilterBitmap, qbm: fr.qbm.Clone()}
	case FilterRange:
		return fr // Range is a value type, no aliasing issues
	default:
		return fr // FilterNone and FilterAll have no data
	}
}

// CloneInto copies the FilterResult into the provided buffer, returning a new FilterResult
// that owns its data. This avoids allocations when the caller provides a buffer with
// sufficient capacity. The returned FilterResult owns the slice (no aliasing with dst).
//
// For FilterRows mode: appends rows to dst and returns a FilterResult pointing to the new slice.
// For FilterBitmap mode: extracts to rows mode (QueryBitmap is zero-alloc, no Clone needed).
// For FilterNone/FilterAll: returns the same value (no data to copy).
//
// Example:
//
//	buf := make([]uint32, 0, 1024)
//	for _, fr := range filterResults {
//	    cloned, buf = fr.CloneInto(buf)
//	    collected = append(collected, cloned)
//	}
func (fr FilterResult) CloneInto(dst []uint32) (FilterResult, []uint32) {
	switch fr.mode {
	case FilterRows:
		if len(fr.rows) == 0 {
			return FilterResult{mode: FilterNone}, dst
		}
		// Mark start position in dst
		startIdx := len(dst)
		// Append rows to dst
		dst = append(dst, fr.rows...)
		// Return FilterResult pointing to the newly appended slice
		return FilterResult{mode: FilterRows, rows: dst[startIdx:]}, dst
	case FilterBitmap:
		// QueryBitmap: always extract to rows (zero-alloc iteration)
		if fr.qbm == nil || fr.qbm.IsEmpty() {
			return FilterResult{mode: FilterNone}, dst
		}
		card := fr.qbm.Cardinality()
		startIdx := len(dst)
		// Ensure capacity
		if cap(dst)-len(dst) < card {
			newCap := len(dst) + card
			newDst := make([]uint32, len(dst), newCap)
			copy(newDst, dst)
			dst = newDst
		}
		// Use ToSlice for zero-alloc extraction
		dst = fr.qbm.ToSlice(dst)
		return FilterResult{mode: FilterRows, rows: dst[startIdx:]}, dst
	case FilterRange:
		// Range: expand to rows if small enough, otherwise keep as range
		count := int(fr.rangeEnd - fr.rangeStart)
		if count <= 0 {
			return FilterResult{mode: FilterNone}, dst
		}
		// For large ranges, keep as range (no copy needed)
		if count > 8192 {
			return fr, dst
		}
		// For small ranges, expand to rows
		startIdx := len(dst)
		if cap(dst)-len(dst) < count {
			newCap := len(dst) + count
			newDst := make([]uint32, len(dst), newCap)
			copy(newDst, dst)
			dst = newDst
		}
		for i := fr.rangeStart; i < fr.rangeEnd; i++ {
			dst = append(dst, i)
		}
		return FilterResult{mode: FilterRows, rows: dst[startIdx:]}, dst
	default:
		return fr, dst // FilterNone and FilterAll have no data
	}
}

// ForEach iterates over all row IDs in the result.
// The callback should return true to continue, false to stop.
// Zero allocations.
func (fr FilterResult) ForEach(fn func(uint32) bool) {
	switch fr.mode {
	case FilterRows:
		for _, id := range fr.rows {
			if !fn(id) {
				return
			}
		}
	case FilterBitmap:
		// QueryBitmap ForEach: zero-alloc, SIMD-friendly
		fr.qbm.ForEach(fn)
	case FilterRange:
		// Range iteration: extremely cache-friendly sequential access
		for i := fr.rangeStart; i < fr.rangeEnd; i++ {
			if !fn(i) {
				return
			}
		}
	}
}

// ToArray extracts all row IDs into a slice.
// For FilterRows, returns the slice directly (no copy).
// For FilterBitmap, uses the provided scratch buffer.
// For FilterRange, expands range into scratch buffer.
func (fr FilterResult) ToArray(scratch []uint32) []uint32 {
	switch fr.mode {
	case FilterRows:
		return fr.rows
	case FilterBitmap:
		// QueryBitmap ToSlice: zero-alloc with scratch
		return fr.qbm.ToSlice(scratch)
	case FilterRange:
		// Expand range to scratch
		count := int(fr.rangeEnd - fr.rangeStart)
		if count <= 0 {
			return scratch[:0]
		}
		if cap(scratch) < count {
			scratch = make([]uint32, count)
		} else {
			scratch = scratch[:count]
		}
		for i := 0; i < count; i++ {
			scratch[i] = fr.rangeStart + uint32(i)
		}
		return scratch
	default:
		return scratch[:0]
	}
}

// ToArrayInto is an alias for ToArray to satisfy segment.Bitmap interface.
// Copies all elements into dst, returning the populated slice.
// If dst has insufficient capacity, a new slice may be allocated.
func (fr FilterResult) ToArrayInto(dst []uint32) []uint32 {
	return fr.ToArray(dst)
}

// Contains checks if the given row ID is in the result.
func (fr FilterResult) Contains(id uint32) bool {
	switch fr.mode {
	case FilterRows:
		// Binary search on sorted rows
		_, found := slices.BinarySearch(fr.rows, id)
		return found
	case FilterBitmap:
		return fr.qbm.Contains(id)
	case FilterRange:
		// O(1) range check
		return id >= fr.rangeStart && id < fr.rangeEnd
	default:
		return false
	}
}

// FilterResultBitmap wraps FilterResult to implement segment.Bitmap interface.
// This allows FilterResult to be used anywhere segment.Bitmap is expected.
// Zero allocations - the wrapper is a value type.
type FilterResultBitmap struct {
	fr FilterResult
}

// AsBitmap returns a FilterResultBitmap wrapper that implements segment.Bitmap.
// Zero allocations.
func (fr FilterResult) AsBitmap() FilterResultBitmap {
	return FilterResultBitmap{fr: fr}
}

// Contains reports whether id is present in the set.
func (frb FilterResultBitmap) Contains(id uint32) bool {
	return frb.fr.Contains(id)
}

// Cardinality returns the number of elements in the set.
func (frb FilterResultBitmap) Cardinality() uint64 {
	return frb.fr.CardinalityUint64()
}

// ForEach calls fn for each id in the set.
func (frb FilterResultBitmap) ForEach(fn func(id uint32) bool) {
	frb.fr.ForEach(fn)
}

// ToArrayInto copies all elements into dst, returning the populated slice.
func (frb FilterResultBitmap) ToArrayInto(dst []uint32) []uint32 {
	return frb.fr.ToArrayInto(dst)
}

// ==============================================================================
// FilterResult AND/OR operations (SIMD-accelerated via QueryBitmap)
// ==============================================================================

// FilterResultAnd performs AND operation on two FilterResults.
// Design rules:
//   - Rows ∩ Rows → Rows (two-pointer merge, zero-alloc)
//   - Rows ∩ Bitmap → Rows (probe QueryBitmap, zero-alloc)
//   - Rows ∩ Range → Rows (filter by range bounds, zero-alloc)
//   - Range ∩ Range → Range (intersect bounds, O(1))
//   - Bitmap ∩ Bitmap → SIMD AND, return rows if small
//   - Range ∩ Bitmap → materialize range intersection
//
// Uses QueryScratch for scratch space (zero allocations in steady state).
func FilterResultAnd(a, b FilterResult, qs *QueryScratch) FilterResult {
	// Fast exits
	if a.mode == FilterNone || b.mode == FilterNone {
		return EmptyResult()
	}

	// Handle Range mode specially (very efficient)
	if a.mode == FilterRange && b.mode == FilterRange {
		return andRangeRange(a, b)
	}
	if a.mode == FilterRange {
		return andRangeOther(a, b, qs)
	}
	if b.mode == FilterRange {
		return andRangeOther(b, a, qs)
	}

	// Normalize: rows first (smaller side first for efficiency)
	if a.mode == FilterBitmap && b.mode == FilterRows {
		a, b = b, a
	}

	switch {
	case a.mode == FilterRows && b.mode == FilterRows:
		return andRowsRows(a.rows, b.rows, qs)

	case a.mode == FilterRows && b.mode == FilterBitmap:
		return andRowsQueryBitmap(a.rows, b.qbm, qs)

	case a.mode == FilterBitmap && b.mode == FilterBitmap:
		return andQueryBitmaps(a.qbm, b.qbm, qs)
	}

	return EmptyResult() // unreachable
}

// andRangeRange intersects two ranges - O(1) operation.
func andRangeRange(a, b FilterResult) FilterResult {
	// Intersect: max(start1, start2), min(end1, end2)
	start := a.rangeStart
	if b.rangeStart > start {
		start = b.rangeStart
	}
	end := a.rangeEnd
	if b.rangeEnd < end {
		end = b.rangeEnd
	}
	if start >= end {
		return EmptyResult()
	}
	return RangeResult(start, end)
}

// andRangeOther intersects a range with rows or bitmap.
func andRangeOther(rangeResult, other FilterResult, qs *QueryScratch) FilterResult {
	switch other.mode {
	case FilterRows:
		// Filter rows by range bounds
		out := qs.TmpRowIDs[:0]
		for _, id := range other.rows {
			if id >= rangeResult.rangeStart && id < rangeResult.rangeEnd {
				out = append(out, id)
			}
		}
		qs.TmpRowIDs = out
		return RowsResult(out)
	case FilterBitmap:
		// Materialize range and intersect with bitmap
		// For large ranges, iterate bitmap and filter by range
		out := qs.TmpRowIDs[:0]
		other.qbm.ForEach(func(id uint32) bool {
			if id >= rangeResult.rangeStart && id < rangeResult.rangeEnd {
				out = append(out, id)
			}
			return true
		})
		qs.TmpRowIDs = out
		return RowsResult(out)
	default:
		return EmptyResult()
	}
}

// andRowsRows performs two-pointer intersection of sorted row slices.
// Zero allocations - uses QueryScratch.TmpRowIDs.
func andRowsRows(a, b []uint32, qs *QueryScratch) FilterResult {
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

// andRowsQueryBitmap probes QueryBitmap for each row ID.
// Zero allocations - uses QueryScratch.TmpRowIDs.
func andRowsQueryBitmap(rows []uint32, qb *bitmap.QueryBitmap, qs *QueryScratch) FilterResult {
	out := qs.TmpRowIDs[:0]

	for _, id := range rows {
		if qb.Contains(id) {
			out = append(out, id)
		}
	}

	qs.TmpRowIDs = out
	return RowsResult(out)
}

// andQueryBitmaps intersects two QueryBitmaps using SIMD.
// If result is small (<1024), returns rows mode.
// Otherwise, returns QueryBitmap mode (SIMD AND already done).
func andQueryBitmaps(a, b *bitmap.QueryBitmap, qs *QueryScratch) FilterResult {
	// Copy 'a' into Tmp1, then SIMD AND with 'b'
	qs.Tmp1.Clear()
	qs.Tmp1.CopyFrom(a)
	qs.Tmp1.And(b)

	// If result is small, extract to rows (more cache-friendly for iteration)
	const rowsThreshold = 1024
	card := qs.Tmp1.Cardinality()
	if card < rowsThreshold {
		out := qs.Tmp1.ToSlice(qs.TmpRowIDs[:0])
		qs.TmpRowIDs = out
		return RowsResult(out)
	}

	return QueryBitmapResult(qs.Tmp1)
}

// FilterResultOr performs OR operation on two FilterResults.
// Design rules:
//   - Rows ∪ Rows → Rows if small, else QueryBitmap (SIMD)
//   - Rows ∪ Bitmap → QueryBitmap (SIMD)
//   - Bitmap ∪ Bitmap → QueryBitmap (SIMD)
//
// Uses QueryScratch for scratch space.
func FilterResultOr(a, b FilterResult, qs *QueryScratch) FilterResult {
	if a.mode == FilterNone {
		return b
	}
	if b.mode == FilterNone {
		return a
	}

	// Rows | Rows: stay in rows mode if result is small
	const rowsThreshold = 1024
	if a.mode == FilterRows && b.mode == FilterRows {
		if len(a.rows)+len(b.rows) <= rowsThreshold {
			return orRowsRows(a.rows, b.rows, qs)
		}
	}

	// Fallback to QueryBitmap for large unions (SIMD OR)
	qs.Tmp1.Clear()
	materializeIntoQueryBitmap(a, qs.Tmp1)
	materializeIntoQueryBitmap(b, qs.Tmp1)
	return QueryBitmapResult(qs.Tmp1)
}

// orRowsRows performs sorted union of row slices.
// Uses QueryScratch.TmpRowIDs.
func orRowsRows(a, b []uint32, qs *QueryScratch) FilterResult {
	out := qs.TmpRowIDs[:0]

	i, j := 0, 0
	for i < len(a) && j < len(b) {
		if a[i] == b[j] {
			out = append(out, a[i])
			i++
			j++
		} else if a[i] < b[j] {
			out = append(out, a[i])
			i++
		} else {
			out = append(out, b[j])
			j++
		}
	}
	// Append remaining
	out = append(out, a[i:]...)
	out = append(out, b[j:]...)

	qs.TmpRowIDs = out
	return RowsResult(out)
}

// materializeIntoQueryBitmap adds FilterResult contents to a QueryBitmap.
func materializeIntoQueryBitmap(fr FilterResult, qb *bitmap.QueryBitmap) {
	switch fr.mode {
	case FilterRows:
		qb.AddMany(fr.rows)
	case FilterBitmap:
		qb.Or(fr.qbm)
	}
}

// BitmapBuilder provides zero-allocation bitmap construction.
// Instead of building bitmaps by mutation (Add calls), it collects rowIDs
// into a preallocated slice, then builds the bitmap once via AddMany.
//
// This eliminates:
//   - arrayContainer growth allocations
//   - container cloning during OR operations
//   - Iterator allocations
//
// Usage:
//
//	bb := NewBitmapBuilder()
//	bb.Reset(estimatedHits)
//	for _, id := range matches {
//	    bb.Append(id)
//	}
//	bitmap := bb.Finalize() // or bb.RowIDs() for direct slice access
type BitmapBuilder struct {
	// rowIDs is a preallocated buffer for collecting matching IDs
	rowIDs []uint32
	// bitmap is reused across queries (never pooled!)
	bitmap *roaring.Bitmap
	// hitsHint is the estimated number of matches
	hitsHint int
}

// NewBitmapBuilder creates a new BitmapBuilder with default capacity.
func NewBitmapBuilder() *BitmapBuilder {
	bm := roaring.New()
	bm.RunOptimize() // good default
	return &BitmapBuilder{
		rowIDs: make([]uint32, 0, 1024),
		bitmap: bm,
	}
}

// Reset prepares the builder for a new query.
// Call this before appending IDs. Zero allocations.
func (bb *BitmapBuilder) Reset(hitsHint int) {
	bb.bitmap.Clear()
	bb.rowIDs = bb.rowIDs[:0]
	bb.hitsHint = hitsHint
}

// Append adds a rowID to the builder.
// This is pure slice append, extremely cache-friendly.
func (bb *BitmapBuilder) Append(id uint32) {
	bb.rowIDs = append(bb.rowIDs, id)
}

// AppendMany adds multiple rowIDs to the builder.
func (bb *BitmapBuilder) AppendMany(ids []uint32) {
	bb.rowIDs = append(bb.rowIDs, ids...)
}

// Len returns the number of rowIDs collected.
func (bb *BitmapBuilder) Len() int {
	return len(bb.rowIDs)
}

// RowIDs returns the collected rowIDs directly.
// For low cardinality filters, use this to skip bitmap entirely.
// The returned slice is only valid until the next Reset call.
func (bb *BitmapBuilder) RowIDs() []uint32 {
	return bb.rowIDs
}

// Finalize builds the bitmap from collected rowIDs.
// RowIDs are sorted (required for optimal AddMany) and added in one call.
// Returns the reusable bitmap (do NOT pool it).
func (bb *BitmapBuilder) Finalize() *LocalBitmap {
	if len(bb.rowIDs) == 0 {
		return &LocalBitmap{rb: bb.bitmap}
	}

	// Sort for optimal AddMany (creates containers in order)
	sortUint32(bb.rowIDs)

	bb.bitmap.AddMany(bb.rowIDs)
	return &LocalBitmap{rb: bb.bitmap}
}

// FinalizeInto builds the bitmap into the provided LocalBitmap.
// This allows using a pooled bitmap while still benefiting from AddMany.
func (bb *BitmapBuilder) FinalizeInto(dst *LocalBitmap) {
	dst.rb.Clear()
	if len(bb.rowIDs) == 0 {
		return
	}

	// Sort for optimal AddMany
	sortUint32(bb.rowIDs)

	dst.rb.AddMany(bb.rowIDs)
}

// sortUint32 sorts a slice of uint32 in ascending order.
// Uses insertion sort for small slices, otherwise uses radix-like approach.
func sortUint32(s []uint32) {
	n := len(s)
	if n <= 16 {
		// Insertion sort for small slices
		for i := 1; i < n; i++ {
			key := s[i]
			j := i - 1
			for j >= 0 && s[j] > key {
				s[j+1] = s[j]
				j--
			}
			s[j+1] = key
		}
		return
	}

	// Use standard library sort for larger slices
	slices.Sort(s)
}
