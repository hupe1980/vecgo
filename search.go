// Package vecgo provides functionalities for an embedded vector store database.
//
// This file implements a fluent search API for querying Vecgo instances.
package vecgo

import (
	"context"
	"iter"
	"slices"

	"github.com/hupe1980/vecgo/index"
	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/searcher"
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
//
//	// Range query (all vectors within distance):
//	results, err := db.Search(query).
//	    WithinDistance(0.5).
//	    MaxResults(1000).
//	    Execute(ctx)
func (vg *Vecgo[T]) Search(query []float32) *SearchBuilder[T] {
	return &SearchBuilder[T]{
		vg:         vg,
		query:      query,
		k:          10,    // Default k
		ef:         0,     // Use index default
		maxResults: 10000, // Default max for range queries
	}
}

// SearchBuilder is a fluent builder for constructing search queries.
type SearchBuilder[T any] struct {
	vg    *Vecgo[T]
	query []float32
	k     int
	ef    int

	// Range query support
	withinDistance *float32 // nil = disabled (use KNN), value = distance threshold
	maxResults     int      // Maximum results for range queries

	// Filters
	filterFunc      func(id uint64) bool
	metadataFilters *metadata.FilterSet

	// Options
	hybrid bool

	// Zero-Alloc Context
	searcher *searcher.Searcher
	buffer   *[]SearchResult[T]
}

// WithSearcher sets the reusable Searcher context for this query.
// If provided, the search will use this context instead of allocating a new one.
// The caller is responsible for resetting and managing the Searcher's lifecycle.
//
// This enables zero-allocation searches in hot loops.
func (sb *SearchBuilder[T]) WithSearcher(s *searcher.Searcher) *SearchBuilder[T] {
	sb.searcher = s
	return sb
}

// WithBuffer sets a reusable buffer for appending results.
// If provided, results will be appended to this slice instead of allocating a new one.
func (sb *SearchBuilder[T]) WithBuffer(buf *[]SearchResult[T]) *SearchBuilder[T] {
	sb.buffer = buf
	return sb
}

// KNN sets the number of nearest neighbors to return.
// This is mutually exclusive with WithinDistance - the last one called wins.
func (sb *SearchBuilder[T]) KNN(k int) *SearchBuilder[T] {
	sb.k = k
	sb.withinDistance = nil // Disable range query mode
	return sb
}

// WithinDistance sets a distance threshold for range queries.
// Returns all vectors with distance <= threshold (up to MaxResults).
// This is mutually exclusive with KNN - the last one called wins.
//
// Use cases:
//   - Deduplication: Find all vectors within distance 0.1
//   - Clustering: Retrieve all neighbors within radius
//   - Threshold alerts: Detect if any similar vectors exist
//
// Example:
//
//	// Find all similar vectors within distance 0.5
//	results, _ := db.Search(query).
//	    WithinDistance(0.5).
//	    MaxResults(100).
//	    Execute(ctx)
func (sb *SearchBuilder[T]) WithinDistance(threshold float32) *SearchBuilder[T] {
	sb.withinDistance = &threshold
	return sb
}

// MaxResults sets the maximum number of results for range queries.
// Only applies when using WithinDistance. Default: 10000.
// For KNN queries, use KNN(k) instead.
func (sb *SearchBuilder[T]) MaxResults(n int) *SearchBuilder[T] {
	sb.maxResults = n
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
func (sb *SearchBuilder[T]) Filter(fn func(id uint64) bool) *SearchBuilder[T] {
	sb.filterFunc = fn
	return sb
}

// WhereID filters results by ID criteria.
// Convenience method for common ID-based filtering patterns.
func (sb *SearchBuilder[T]) WhereID(fn func(id uint64) bool) *SearchBuilder[T] {
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
	// Range query mode
	if sb.withinDistance != nil {
		return sb.executeRangeQuery(ctx)
	}

	// Standard KNN mode
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

	// Zero-Alloc Path: Use provided Searcher
	if sb.searcher != nil {
		if err := sb.vg.KNNSearchWithContext(ctx, sb.query, sb.k, sb.searcher, func(o *KNNSearchOptions) {
			if sb.ef > 0 {
				o.EF = sb.ef
			}
			if sb.filterFunc != nil {
				o.FilterFunc = sb.filterFunc
			}
		}); err != nil {
			return nil, err
		}

		// If buffer is provided, append results to it
		if sb.buffer != nil {
			// Extract from heap to buffer
			// Note: MaxHeap pops worst first, so we need to pop (Len-k) items first if any
			// But KNNSearchWithContext should have already kept top K.
			// We just need to extract and reverse.

			// We need to be careful not to drain the searcher if the user wants to inspect it later?
			// But the contract is we return []SearchResult.
			// If the user provided a buffer, we append to it.

			// Extract results from searcher.Candidates
			results := sb.searcher.Candidates
			startIdx := len(*sb.buffer)

			for results.Len() > 0 {
				item, _ := results.PopItem()
				*sb.buffer = append(*sb.buffer, SearchResult[T]{
					SearchResult: index.SearchResult{
						ID:       item.Node,
						Distance: item.Distance,
					},
				})
			}

			// Reverse
			endIdx := len(*sb.buffer) - 1
			for i := 0; i < (endIdx-startIdx+1)/2; i++ {
				(*sb.buffer)[startIdx+i], (*sb.buffer)[endIdx-i] = (*sb.buffer)[endIdx-i], (*sb.buffer)[startIdx+i]
			}

			return *sb.buffer, nil
		}

		// If no buffer provided, allocate one
		res := make([]SearchResult[T], 0, sb.k)
		results := sb.searcher.Candidates
		for results.Len() > 0 {
			item, _ := results.PopItem()
			res = append(res, SearchResult[T]{
				SearchResult: index.SearchResult{
					ID:       item.Node,
					Distance: item.Distance,
				},
			})
		}
		// Reverse
		slices.Reverse(res)
		return res, nil
	}

	// Standard Path (Internal Searcher)
	return sb.vg.KNNSearch(ctx, sb.query, sb.k, func(o *KNNSearchOptions) {
		if sb.ef > 0 {
			o.EF = sb.ef
		}
		if sb.filterFunc != nil {
			o.FilterFunc = sb.filterFunc
		}
	})
}

// executeRangeQuery performs a range query returning all vectors within the distance threshold.
func (sb *SearchBuilder[T]) executeRangeQuery(ctx context.Context) ([]SearchResult[T], error) {
	threshold := *sb.withinDistance

	// Use maxResults as the search k to get enough candidates
	// We'll filter by distance threshold after
	searchK := sb.maxResults

	// For range queries with filters, we need to over-fetch to account for filtered results
	if sb.filterFunc != nil {
		searchK = min(searchK*2, 100000) // Over-fetch but cap at reasonable limit
	}

	// Perform KNN search with expanded k
	results, err := sb.vg.KNNSearch(ctx, sb.query, searchK, func(o *KNNSearchOptions) {
		if sb.ef > 0 {
			o.EF = sb.ef
		}
		if sb.filterFunc != nil {
			o.FilterFunc = sb.filterFunc
		}
	})
	if err != nil {
		return nil, err
	}

	// Filter by distance threshold
	filtered := make([]SearchResult[T], 0, len(results))
	for _, r := range results {
		if r.Distance <= threshold {
			filtered = append(filtered, r)
			if len(filtered) >= sb.maxResults {
				break
			}
		}
	}

	return filtered, nil
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
// For range queries (WithinDistance), only results within the threshold are yielded.
//
// Example:
//
//	for result, err := range db.Search(query).KNN(100).Stream(ctx) {
//	    if err != nil { break }
//	    if result.Distance > 100.0 { break } // Early termination
//	    process(result)
//	}
func (sb *SearchBuilder[T]) Stream(ctx context.Context) iter.Seq2[SearchResult[T], error] {
	// For range queries, wrap the standard stream with distance filtering
	if sb.withinDistance != nil {
		threshold := *sb.withinDistance
		return func(yield func(SearchResult[T], error) bool) {
			count := 0
			for result, err := range sb.vg.KNNSearchStream(ctx, sb.query, sb.maxResults, func(o *KNNSearchOptions) {
				if sb.ef > 0 {
					o.EF = sb.ef
				}
				if sb.filterFunc != nil {
					o.FilterFunc = sb.filterFunc
				}
			}) {
				if err != nil {
					yield(SearchResult[T]{}, err)
					return
				}
				// Stop if beyond distance threshold
				if result.Distance > threshold {
					return
				}
				count++
				if count > sb.maxResults {
					return
				}
				if !yield(result, nil) {
					return
				}
			}
		}
	}

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

// Nearest is an alias for First - returns the single nearest result.
func (sb *SearchBuilder[T]) Nearest(ctx context.Context) (SearchResult[T], error) {
	return sb.First(ctx)
}

// Count executes the search and returns the number of results.
// For range queries, returns the count of vectors within the distance threshold.
func (sb *SearchBuilder[T]) Count(ctx context.Context) (int, error) {
	results, err := sb.Execute(ctx)
	if err != nil {
		return 0, err
	}
	return len(results), nil
}

// Exists checks if at least one result matches the search.
// For range queries, checks if any vector exists within the distance threshold.
func (sb *SearchBuilder[T]) Exists(ctx context.Context) (bool, error) {
	if sb.withinDistance != nil {
		// For range queries, check if any result is within threshold
		sb.maxResults = 1
	} else {
		sb.k = 1
	}
	results, err := sb.Execute(ctx)
	if err != nil {
		return false, err
	}
	return len(results) > 0, nil
}

// ExistsWithin checks if any vector exists within the specified distance.
// This is a convenience method equivalent to WithinDistance(threshold).Exists(ctx).
func (sb *SearchBuilder[T]) ExistsWithin(ctx context.Context, threshold float32) (bool, error) {
	return sb.WithinDistance(threshold).MaxResults(1).Exists(ctx)
}

// CountWithin returns the count of vectors within the specified distance.
// This is a convenience method equivalent to WithinDistance(threshold).Count(ctx).
func (sb *SearchBuilder[T]) CountWithin(ctx context.Context, threshold float32) (int, error) {
	return sb.WithinDistance(threshold).Count(ctx)
}
