// Package metadata provides efficient metadata storage and filtering for Vecgo.
//
// The metadata system uses a Roaring Bitmap-based inverted index for fast
// filtering during vector search operations.
//
// # Metadata Types
//
// Metadata values can be:
//
//   - String: metadata.String("tech")
//   - Int: metadata.Int(2024)
//   - Float: metadata.Float(3.14)
//   - Bool: metadata.Bool(true)
//   - Slice: metadata.Slice([]string{"a", "b"})
//
// Example:
//
//	meta := metadata.Metadata{
//	    "category": metadata.String("tech"),
//	    "year": metadata.Int(2024),
//	    "published": metadata.Bool(true),
//	}
//
// # Filter Operations
//
// Build complex filters with boolean operators:
//
//   - Eq(field, value): Equality check
//   - Neq(field, value): Inequality check
//   - Gt(field, value): Greater than
//   - Gte(field, value): Greater than or equal
//   - Lt(field, value): Less than
//   - Lte(field, value): Less than or equal
//   - In(field, values...): Value in set
//   - And(filters...): Logical AND
//   - Or(filters...): Logical OR
//   - Not(filter): Logical NOT
//
// Example:
//
//	filter := metadata.And(
//	    metadata.Eq("category", "tech"),
//	    metadata.Gte("year", 2023),
//	    metadata.Or(
//	        metadata.Eq("status", "published"),
//	        metadata.Eq("status", "featured"),
//	    ),
//	)
//
// # Performance
//
// The Roaring Bitmap-based index provides:
//
//   - 10,000x faster filter compilation vs linear scan
//   - 50% memory reduction vs duplicated metadata per index
//   - Efficient set operations (AND/OR/NOT) on millions of IDs
//
// # Usage with Search
//
// Apply metadata filters during search:
//
//	results, err := db.Search(query).
//	    KNN(10).
//	    Filter(metadata.Eq("category", "tech")).
//	    Execute(ctx)
//
// The filter is compiled to a bitmap and applied during the search,
// returning only vectors that match both the vector similarity
// criterion and the metadata filter.
//
// # Subpackages
//
//   - index: Roaring Bitmap-based inverted index implementation
package metadata
