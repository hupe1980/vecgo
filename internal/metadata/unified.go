package imetadata

import (
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"math"
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
//
// Benefits:
//   - Memory efficient (Bitmap compression + String Interning)
//   - Fast filter compilation (Bitmap AND/OR operations)
//   - Simple API (single unified type)
type UnifiedIndex struct {
	mu sync.RWMutex

	// Primary metadata storage (id -> metadata document)
	documents map[model.RowID]metadata.InternedDocument

	// Inverted index for fast filtering
	// Structure: field -> valueKey -> bitmap of IDs
	// Bitmaps are compressed and support fast set operations
	inverted map[unique.Handle[string]]map[unique.Handle[string]]*LocalBitmap

	// provider is an optional fallback for document retrieval
	provider DocumentProvider
}

// NewUnifiedIndex creates a new unified metadata index.
func NewUnifiedIndex() *UnifiedIndex {
	return &UnifiedIndex{
		documents: make(map[model.RowID]metadata.InternedDocument),
		inverted:  make(map[unique.Handle[string]]map[unique.Handle[string]]*LocalBitmap),
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

// EvaluateFilter evaluates a filter set against the inverted index and returns a bitmap.
// This method supports all operators, including numeric comparisons like OpLessThan.
// For operators not directly supported by the inverted index (e.g. OpLessThan),
// it falls back to scanning all unique values in the index.
func (ui *UnifiedIndex) EvaluateFilter(fs *metadata.FilterSet) *LocalBitmap {
	if fs == nil || len(fs.Filters) == 0 {
		return nil // All matches
	}

	ui.mu.RLock()
	defer ui.mu.RUnlock()

	var result *LocalBitmap

	for _, f := range fs.Filters {
		var current *LocalBitmap

		switch f.Operator {
		case metadata.OpEqual:
			b := ui.getBitmapLocked(f.Key, f.Value)
			if b != nil {
				current = b.Clone()
			} else {
				return NewLocalBitmap() // No match for this filter
			}

		case metadata.OpIn:
			current = NewLocalBitmap()
			arr, ok := f.Value.AsArray()
			if ok {
				for _, v := range arr {
					if b := ui.getBitmapLocked(f.Key, v); b != nil {
						current.Or(b)
					}
				}
			}

		case metadata.OpLessThan, metadata.OpLessEqual, metadata.OpGreaterThan, metadata.OpGreaterEqual, metadata.OpNotEqual:
			// For numeric comparisons, scan all values in the inverted index
			current = ui.evaluateNumericFilter(f)

		default:
			// Unsupported operator - fall back to empty (will fail filter)
			return NewLocalBitmap()
		}

		if result == nil {
			result = current
		} else {
			result.And(current)
		}

		if result.IsEmpty() {
			return result
		}
	}

	return result
}

// evaluateNumericFilter handles comparison operators by scanning the inverted index.
func (ui *UnifiedIndex) evaluateNumericFilter(f metadata.Filter) *LocalBitmap {
	result := NewLocalBitmap()

	valueMap, ok := ui.inverted[unique.Make(f.Key)]
	if !ok {
		return result // Field doesn't exist, no matches
	}

	for valueKey, bitmap := range valueMap {
		// Parse the stored value to check against filter
		// The valueKey is the serialized form of the value
		storedValue := parseValueKey(valueKey.Value())

		if matchesComparison(storedValue, f.Value, f.Operator) {
			result.Or(bitmap)
		}
	}

	return result
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

// addToIndexLocked adds a document to the inverted index.
// Caller must hold ui.mu.Lock().
func (ui *UnifiedIndex) addToIndexLocked(id model.RowID, doc metadata.InternedDocument) {
	for key, value := range doc {
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
	}
}

// removeFromIndexLocked removes a document from the inverted index.
// Caller must hold ui.mu.Lock().
func (ui *UnifiedIndex) removeFromIndexLocked(id model.RowID, doc metadata.InternedDocument) {
	for key, value := range doc {
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
	default:
		return ui.createFallbackCheck(ctx, filter)
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
		scratch := NewLocalBitmap()
		for _, v := range arr {
			if b := ui.getBitmapLocked(filter.Key, v); b != nil {
				scratch.Or(b)
			}
		}
		dst.And(scratch)
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
//   - Slow path: Falls back to scanning + evaluating each document
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
	ui.mu.RLock()
	defer ui.mu.RUnlock()

	return func(id model.RowID) bool {
		doc, ok := ui.documents[id]
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
