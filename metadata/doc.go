// Package metadata provides efficient metadata storage and filtering for Vecgo.
//
// This package defines the typed metadata model (Value, Document, Filter) and
// comparison logic. The inverted index implementation is in internal/metadata.
//
// # Metadata Types
//
// Metadata values can be:
//
//   - String: metadata.String("tech")
//   - Int: metadata.Int(2024)
//   - Float: metadata.Float(3.14)
//   - Bool: metadata.Bool(true)
//   - Slice: metadata.Array([]Value{...})
//
// Example:
//
//	meta := metadata.Document{
//	    "category": metadata.String("tech"),
//	    "year": metadata.Int(2024),
//	    "published": metadata.Bool(true),
//	}
//
// # Filter Operations
//
// Build filters using the Filter struct:
//
//   - OpEqual: Equality check
//   - OpNotEqual: Inequality check
//   - OpGreaterThan, OpGreaterEqual: Numeric comparisons
//   - OpLessThan, OpLessEqual: Numeric comparisons
//   - OpIn: Value in set (array)
//   - OpContains: String substring match
//
// Multiple filters can be combined with FilterSet (AND logic):
//
//	filter := metadata.NewFilterSet(
//	    metadata.Filter{Key: "category", Operator: metadata.OpEqual, Value: metadata.String("tech")},
//	    metadata.Filter{Key: "year", Operator: metadata.OpGreaterEqual, Value: metadata.Int(2023)},
//	)
//
// # Performance Features
//
//   - String interning: unique.Handle[string] reduces memory for repeated strings
//   - Binary encoding: Compact serialization for persistence
//   - Cached interned keys: FilterSet caches interned keys for repeated matching
//   - Short-circuit evaluation: AND logic stops at first non-match
//
// # Usage with Search
//
// Apply metadata filters during search:
//
//	results, err := db.Search(query).
//	    KNN(10).
//	    Filter(filter).
//	    Execute(ctx)
//
// The engine compiles filters to bitmap operations via internal/metadata
// for efficient set-based filtering on large datasets.
package metadata
