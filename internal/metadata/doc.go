// Package imetadata provides internal metadata indexing for efficient filtering.
//
// This package implements the inverted index used for metadata filtering.
// It is separate from the public metadata package to avoid import cycles.
//
// # Architecture
//
// UnifiedIndex combines primary storage with an inverted index:
//
//	Primary:  map[RowID]InternedDocument      - O(1) document retrieval
//	Inverted: map[field]map[value]*Bitmap     - O(1) filter compilation
//
// # Filter Compilation
//
// Filters are compiled into bitmap operations:
//
//	OpEqual:  field == value  → single bitmap lookup
//	OpIn:     field IN [...]  → union of bitmaps (OR)
//	Multiple: AND all filter bitmaps
//
// Unsupported operators (OpLessThan, etc.) fall back to document scanning.
//
// # Memory Efficiency
//
// The package uses several techniques to minimize memory:
//
//   - String interning via unique.Handle[string] (Go 1.23+)
//   - Roaring Bitmaps for compressed posting lists
//   - Value.Key() deduplication in inverted index
//
// # Thread Safety
//
// UnifiedIndex is fully thread-safe. All public methods acquire appropriate
// locks (RLock for reads, Lock for writes). For streaming filters, callers
// must hold RLock manually via RLock()/RUnlock() methods.
//
// # Serialization
//
// The inverted index supports binary serialization via WriteInvertedIndex
// and ReadInvertedIndex for persistence in disk segments.
package imetadata
