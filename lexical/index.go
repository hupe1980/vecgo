package lexical

import "github.com/hupe1980/vecgo/model"

// Index is the interface for a lexical search index.
type Index interface {
	// Add adds a document to the index.
	Add(pk model.PK, text string) error
	// Delete removes a document from the index.
	Delete(pk model.PK) error
	// Search performs a keyword search and returns a list of candidates.
	Search(text string, k int) ([]model.Candidate, error)
	// Close closes the index.
	Close() error
}
