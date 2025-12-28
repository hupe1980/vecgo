// Package vecgo provides functionalities for an embedded vector store database.
//
// This file implements a fluent search API for querying Vecgo instances.
package vecgo

import (
	"context"
	"iter"

	"github.com/hupe1980/vecgo/metadata"
)

// Search creates a new fluent search builder for the given query vector.
//
// Example:
//
//	results, err := db.Search(query).
//	    KNN(10).
//	    EF(100).
//	    Execute(ctx)
//
//	// Or with streaming:
//	for result, err := range db.Search(query).KNN(100).Stream(ctx) {
//	    if err != nil { break }
//	    if result.Distance > threshold { break }
//	    process(result)
//	}
func (vg *Vecgo[T]) Search(query []float32) *SearchBuilder[T] {
	return &SearchBuilder[T]{
		vg:    vg,
		query: query,
		k:     10, // Default k
		ef:    0,  // Use index default
	}
}

// SearchBuilder is a fluent builder for constructing search queries.
type SearchBuilder[T any] struct {
	vg    *Vecgo[T]
	query []float32
	k     int
	ef    int

	// Filters
	filterFunc      func(id uint32) bool
	metadataFilters *metadata.FilterSet

	// Options
	hybrid bool
}

// KNN sets the number of nearest neighbors to return.
func (sb *SearchBuilder[T]) KNN(k int) *SearchBuilder[T] {
	sb.k = k
	return sb
}

// EF sets the exploration factor for HNSW search.
// Higher values improve recall but slow down search.
// Must be >= k.
func (sb *SearchBuilder[T]) EF(ef int) *SearchBuilder[T] {
	sb.ef = ef
	return sb
}

// Filter sets a filter function for search results.
// Only vectors where filter(id) returns true are considered.
func (sb *SearchBuilder[T]) Filter(fn func(id uint32) bool) *SearchBuilder[T] {
	sb.filterFunc = fn
	return sb
}

// WhereID filters results by ID criteria.
// Convenience method for common ID-based filtering patterns.
func (sb *SearchBuilder[T]) WhereID(fn func(id uint32) bool) *SearchBuilder[T] {
	return sb.Filter(fn)
}

// WithMetadata sets metadata filters for hybrid search.
func (sb *SearchBuilder[T]) WithMetadata(filters *metadata.FilterSet) *SearchBuilder[T] {
	sb.metadataFilters = filters
	sb.hybrid = true
	return sb
}

// Execute runs the search and returns the results.
func (sb *SearchBuilder[T]) Execute(ctx context.Context) ([]SearchResult[T], error) {
	if sb.hybrid || sb.metadataFilters != nil {
		// Hybrid search doesn't support FilterFunc directly,
		// only metadata filters. If user wants ID-based filtering
		// with metadata, they should use KNN with post-filtering.
		return sb.vg.HybridSearch(ctx, sb.query, sb.k, func(o *HybridSearchOptions) {
			if sb.ef > 0 {
				o.EF = sb.ef
			}
			o.MetadataFilters = sb.metadataFilters
		})
	}

	return sb.vg.KNNSearch(ctx, sb.query, sb.k, func(o *KNNSearchOptions) {
		if sb.ef > 0 {
			o.EF = sb.ef
		}
		if sb.filterFunc != nil {
			o.FilterFunc = sb.filterFunc
		}
	})
}

// MustExecute runs the search, panicking on error.
// Use this only in tests or when you're certain the query is valid.
func (sb *SearchBuilder[T]) MustExecute(ctx context.Context) []SearchResult[T] {
	results, err := sb.Execute(ctx)
	if err != nil {
		panic(err)
	}
	return results
}

// Stream returns an iterator over search results for memory-efficient processing.
// Results are yielded in order from nearest to farthest.
// The iterator supports early termination by breaking from the loop.
//
// Example:
//
//	for result, err := range db.Search(query).KNN(100).Stream(ctx) {
//	    if err != nil { break }
//	    if result.Distance > 100.0 { break } // Early termination
//	    process(result)
//	}
func (sb *SearchBuilder[T]) Stream(ctx context.Context) iter.Seq2[SearchResult[T], error] {
	return sb.vg.KNNSearchStream(ctx, sb.query, sb.k, func(o *KNNSearchOptions) {
		if sb.ef > 0 {
			o.EF = sb.ef
		}
		if sb.filterFunc != nil {
			o.FilterFunc = sb.filterFunc
		}
	})
}

// First returns only the nearest result, or an error if none found.
func (sb *SearchBuilder[T]) First(ctx context.Context) (SearchResult[T], error) {
	sb.k = 1
	results, err := sb.Execute(ctx)
	if err != nil {
		return SearchResult[T]{}, err
	}
	if len(results) == 0 {
		return SearchResult[T]{}, ErrNotFound
	}
	return results[0], nil
}

// Count executes the search and returns the number of results.
func (sb *SearchBuilder[T]) Count(ctx context.Context) (int, error) {
	results, err := sb.Execute(ctx)
	if err != nil {
		return 0, err
	}
	return len(results), nil
}

// Exists checks if at least one result matches the search.
func (sb *SearchBuilder[T]) Exists(ctx context.Context) (bool, error) {
	sb.k = 1
	results, err := sb.Execute(ctx)
	if err != nil {
		return false, err
	}
	return len(results) > 0, nil
}
