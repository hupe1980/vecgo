package engine

// SearchResult represents a search result with a global ID.
type SearchResult struct {
	// ID is the global identifier of the search result.
	ID uint64

	// Distance is the distance between the query vector and the result vector.
	Distance float32
}

// SearchOptions contains parameters for KNN search.
type SearchOptions struct {
	// EFSearch is the exploration factor for HNSW search.
	EFSearch int

	// Filter function to exclude results during search.
	// The filter receives the global ID.
	Filter func(id uint64) bool
}
