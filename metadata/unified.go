// Package metadata provides a unified metadata storage and indexing system.
// This package combines metadata storage with inverted indexing using Bitmaps
// for efficient hybrid vector + metadata search.
package metadata

import (
	"sync"
	"unique"

	"github.com/hupe1980/vecgo/core"
)

// UnifiedIndex combines metadata storage with inverted indexing using Bitmaps.
// This provides efficient hybrid vector + metadata search with minimal memory overhead.
//
// Architecture:
//   - Primary storage: map[core.LocalID]InternedDocument (metadata by ID, interned keys)
//   - Inverted index: map[key]map[valueKey]*LocalBitmap (efficient posting lists)
//
// Benefits:
//   - Memory efficient (Bitmap compression + String Interning)
//   - Fast filter compilation (Bitmap AND/OR operations)
//   - Simple API (single unified type)
type UnifiedIndex struct {
	mu sync.RWMutex

	// Primary metadata storage (id -> metadata document)
	documents map[core.LocalID]InternedDocument

	// Inverted index for fast filtering
	// Structure: field -> valueKey -> bitmap of IDs
	// Bitmaps are compressed and support fast set operations
	inverted map[unique.Handle[string]]map[unique.Handle[string]]*LocalBitmap
}

// NewUnifiedIndex creates a new unified metadata index.
func NewUnifiedIndex() *UnifiedIndex {
	return &UnifiedIndex{
		documents: make(map[core.LocalID]InternedDocument),
		inverted:  make(map[unique.Handle[string]]map[unique.Handle[string]]*LocalBitmap),
	}
}

// Set stores metadata for an ID and updates the inverted index.
// This replaces any existing metadata for the ID.
func (ui *UnifiedIndex) Set(id core.LocalID, doc Document) {
	if doc == nil {
		return
	}

	// Convert to interned format
	iDoc := make(InternedDocument, len(doc))
	for k, v := range doc {
		iDoc[unique.Make(k)] = v
	}

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

// Get retrieves metadata for an ID.
// Returns nil if the ID doesn't exist.
func (ui *UnifiedIndex) Get(id core.LocalID) (Document, bool) {
	ui.mu.RLock()
	defer ui.mu.RUnlock()

	iDoc, ok := ui.documents[id]
	if !ok {
		return nil, false
	}

	// Convert to public format
	doc := make(Document, len(iDoc))
	for k, v := range iDoc {
		doc[k.Value()] = v
	}
	return doc, true
}

// Delete removes metadata for an ID and updates the inverted index.
func (ui *UnifiedIndex) Delete(id core.LocalID) {
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
func (ui *UnifiedIndex) ToMap() map[core.LocalID]Document {
	ui.mu.RLock()
	defer ui.mu.RUnlock()

	result := make(map[core.LocalID]Document, len(ui.documents))
	for id, iDoc := range ui.documents {
		doc := make(Document, len(iDoc))
		for k, v := range iDoc {
			doc[k.Value()] = v
		}
		result[id] = doc
	}
	return result
}

// addToIndexLocked adds a document to the inverted index.
// Caller must hold ui.mu.Lock().
func (ui *UnifiedIndex) addToIndexLocked(id core.LocalID, doc InternedDocument) {
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
		bitmap.Add(id)
	}
}

// removeFromIndexLocked removes a document from the inverted index.
// Caller must hold ui.mu.Lock().
func (ui *UnifiedIndex) removeFromIndexLocked(id core.LocalID, doc InternedDocument) {
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
		bitmap.Remove(id)

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
func (ui *UnifiedIndex) CreateStreamingFilter(fs *FilterSet) func(core.LocalID) bool {
	if fs == nil || len(fs.Filters) == 0 {
		return func(core.LocalID) bool { return true }
	}

	// Pre-resolve bitmaps for supported operators
	checks := make([]func(core.LocalID) bool, 0, len(fs.Filters))

	for _, filter := range fs.Filters {
		checks = append(checks, ui.createFilterCheck(filter))
	}

	return func(id core.LocalID) bool {
		for _, check := range checks {
			if !check(id) {
				return false
			}
		}
		return true
	}
}

func (ui *UnifiedIndex) createFilterCheck(filter Filter) func(core.LocalID) bool {
	switch filter.Operator {
	case OpEqual:
		return ui.createEqualCheck(filter)
	case OpIn:
		return ui.createInCheck(filter)
	default:
		return ui.createFallbackCheck(filter)
	}
}

func (ui *UnifiedIndex) createEqualCheck(filter Filter) func(core.LocalID) bool {
	// Fast path: check bitmap directly
	bitmap := ui.getBitmapLocked(filter.Key, filter.Value)
	if bitmap == nil {
		return func(core.LocalID) bool { return false }
	}
	return func(id core.LocalID) bool {
		return bitmap.Contains(id)
	}
}

func (ui *UnifiedIndex) createInCheck(filter Filter) func(core.LocalID) bool {
	// Check if ID is in ANY of the bitmaps
	arr, ok := filter.Value.AsArray()
	if !ok {
		return func(core.LocalID) bool { return false }
	}

	var bitmaps []*LocalBitmap
	for _, v := range arr {
		if b := ui.getBitmapLocked(filter.Key, v); b != nil {
			bitmaps = append(bitmaps, b)
		}
	}

	if len(bitmaps) == 0 {
		return func(core.LocalID) bool { return false }
	}

	return func(id core.LocalID) bool {
		for _, b := range bitmaps {
			if b.Contains(id) {
				return true
			}
		}
		return false
	}
}

func (ui *UnifiedIndex) createFallbackCheck(filter Filter) func(core.LocalID) bool {
	// Fallback for unsupported operators: check document
	// This is slow but necessary
	return func(id core.LocalID) bool {
		doc, ok := ui.documents[id]
		if !ok {
			return false
		}
		return filter.MatchesInterned(doc)
	}
}

// CompileFilter compiles a FilterSet into a fast bitmap-based filter.
// Returns a bitmap of matching IDs, or nil if compilation fails.
//
// Supported operators:
//   - OpEqual: field == value
//   - OpIn: field IN (value1, value2, ...)
//   - OpNotEqual, OpGreaterThan, etc.: Falls back to scanning
func (ui *UnifiedIndex) CompileFilter(fs *FilterSet) *LocalBitmap {
	dst := NewLocalBitmap()
	if ui.CompileFilterTo(fs, dst) {
		return dst
	}
	return nil
}

// CompileFilterTo compiles a FilterSet into the provided destination bitmap.
// Returns true if compilation succeeded (all operators supported), false otherwise.
// The destination bitmap is cleared before use.
func (ui *UnifiedIndex) CompileFilterTo(fs *FilterSet, dst *LocalBitmap) bool {
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

func (ui *UnifiedIndex) applyFilterToBitmap(filter Filter, dst *LocalBitmap, first bool) bool {
	switch filter.Operator {
	case OpEqual:
		return ui.applyEqualFilter(filter, dst, first)
	case OpIn:
		return ui.applyInFilter(filter, dst, first)
	default:
		return false
	}
}

func (ui *UnifiedIndex) applyEqualFilter(filter Filter, dst *LocalBitmap, first bool) bool {
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

func (ui *UnifiedIndex) applyInFilter(filter Filter, dst *LocalBitmap, first bool) bool {
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
func (ui *UnifiedIndex) getBitmapLocked(key string, value Value) *LocalBitmap {
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
func (ui *UnifiedIndex) ScanFilter(fs *FilterSet) []core.LocalID {
	if fs == nil {
		return nil
	}

	ui.mu.RLock()
	defer ui.mu.RUnlock()

	result := make([]core.LocalID, 0, len(ui.documents))

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
func (ui *UnifiedIndex) CreateFilterFunc(fs *FilterSet) func(core.LocalID) bool {
	if fs == nil || len(fs.Filters) == 0 {
		return nil
	}

	// Try fast path: compile to bitmap
	bitmap := ui.CompileFilter(fs)
	if bitmap != nil {
		// Fast bitmap-based lookup (O(1) average case)
		return func(id core.LocalID) bool {
			return bitmap.Contains(id)
		}
	}

	// Slow path: evaluate filter for each ID
	ui.mu.RLock()
	defer ui.mu.RUnlock()

	return func(id core.LocalID) bool {
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
