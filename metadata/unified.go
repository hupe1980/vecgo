// Package metadata provides a unified metadata storage and indexing system.
// This package combines metadata storage with inverted indexing using Bitmaps
// for efficient hybrid vector + metadata search.
package metadata

import (
	"sync"
	"unique"
)

// UnifiedIndex combines metadata storage with inverted indexing using Bitmaps.
// This provides efficient hybrid vector + metadata search with minimal memory overhead.
//
// Architecture:
//   - Primary storage: map[uint64]InternedDocument (metadata by ID, interned keys)
//   - Inverted index: map[key]map[valueKey]*Bitmap (efficient posting lists)
//
// Benefits:
//   - Memory efficient (Bitmap compression + String Interning)
//   - Fast filter compilation (Bitmap AND/OR operations)
//   - Simple API (single unified type)
type UnifiedIndex struct {
	mu sync.RWMutex

	// Primary metadata storage (id -> metadata document)
	documents map[uint64]InternedDocument

	// Inverted index for fast filtering
	// Structure: field -> valueKey -> bitmap of IDs
	// Bitmaps are compressed and support fast set operations
	inverted map[unique.Handle[string]]map[unique.Handle[string]]*Bitmap
}

// NewUnifiedIndex creates a new unified metadata index.
func NewUnifiedIndex() *UnifiedIndex {
	return &UnifiedIndex{
		documents: make(map[uint64]InternedDocument),
		inverted:  make(map[unique.Handle[string]]map[unique.Handle[string]]*Bitmap),
	}
}

// Set stores metadata for an ID and updates the inverted index.
// This replaces any existing metadata for the ID.
func (ui *UnifiedIndex) Set(id uint64, doc Document) {
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
func (ui *UnifiedIndex) Get(id uint64) (Document, bool) {
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
func (ui *UnifiedIndex) Delete(id uint64) {
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
func (ui *UnifiedIndex) ToMap() map[uint64]Document {
	ui.mu.RLock()
	defer ui.mu.RUnlock()

	result := make(map[uint64]Document, len(ui.documents))
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
func (ui *UnifiedIndex) addToIndexLocked(id uint64, doc InternedDocument) {
	for key, value := range doc {
		// Get or create value map for this field
		valueMap, ok := ui.inverted[key]
		if !ok {
			valueMap = make(map[unique.Handle[string]]*Bitmap)
			ui.inverted[key] = valueMap
		}

		// Get or create bitmap for this value
		valueKey := unique.Make(value.Key())
		bitmap, ok := valueMap[valueKey]
		if !ok {
			bitmap = NewBitmap()
			valueMap[valueKey] = bitmap
		}

		// Add ID to bitmap
		bitmap.Add(id)
	}
}

// removeFromIndexLocked removes a document from the inverted index.
// Caller must hold ui.mu.Lock().
func (ui *UnifiedIndex) removeFromIndexLocked(id uint64, doc InternedDocument) {
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

// CompileFilter compiles a FilterSet into a fast bitmap-based filter.
// Returns a bitmap of matching IDs, or nil if compilation fails.
//
// Supported operators:
//   - OpEqual: field == value
//   - OpIn: field IN (value1, value2, ...)
//   - OpNotEqual, OpGreaterThan, etc.: Falls back to scanning
func (ui *UnifiedIndex) CompileFilter(fs *FilterSet) *Bitmap {
	if fs == nil || len(fs.Filters) == 0 {
		return nil
	}

	ui.mu.RLock()
	defer ui.mu.RUnlock()

	var result *Bitmap

	for _, filter := range fs.Filters {
		var filterBitmap *Bitmap

		switch filter.Operator {
		case OpEqual:
			// Get bitmap for this exact key=value
			filterBitmap = ui.getBitmapLocked(filter.Key, filter.Value)

		case OpIn:
			// Union of all matching values
			arr, ok := filter.Value.AsArray()
			if !ok {
				// Can't compile OpIn with non-array value
				return nil
			}

			filterBitmap = NewBitmap()
			for _, v := range arr {
				if bitmap := ui.getBitmapLocked(filter.Key, v); bitmap != nil {
					filterBitmap = filterBitmap.Or(bitmap)
				}
			}

		default:
			// Can't compile other operators (GreaterThan, LessThan, etc.)
			// Caller should fall back to scanning + evaluating
			return nil
		}

		// Intersect with previous results (AND operation)
		if result == nil {
			if filterBitmap != nil {
				result = filterBitmap
			} else {
				// First filter has no matches - return empty
				return NewBitmap()
			}
		} else if filterBitmap != nil {
			result = result.And(filterBitmap)
		} else {
			// Empty result - no matches possible
			return NewBitmap()
		}

		// Early termination if result is empty
		if result.IsEmpty() {
			return result
		}
	}

	// If no filters were provided, return empty bitmap
	if result == nil {
		return NewBitmap()
	}

	return result
}

// getBitmapLocked retrieves the bitmap for a specific field=value combination.
// Returns nil if no matches exist. Caller must hold ui.mu.RLock().
func (ui *UnifiedIndex) getBitmapLocked(key string, value Value) *Bitmap {
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
func (ui *UnifiedIndex) ScanFilter(fs *FilterSet) []uint64 {
	if fs == nil {
		return nil
	}

	ui.mu.RLock()
	defer ui.mu.RUnlock()

	result := make([]uint64, 0, len(ui.documents))

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
func (ui *UnifiedIndex) CreateFilterFunc(fs *FilterSet) func(uint64) bool {
	if fs == nil || len(fs.Filters) == 0 {
		return nil
	}

	// Try fast path: compile to bitmap
	bitmap := ui.CompileFilter(fs)
	if bitmap != nil {
		// Fast bitmap-based lookup (O(1) average case)
		return func(id uint64) bool {
			return bitmap.Contains(id)
		}
	}

	// Slow path: evaluate filter for each ID
	ui.mu.RLock()
	defer ui.mu.RUnlock()

	return func(id uint64) bool {
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
