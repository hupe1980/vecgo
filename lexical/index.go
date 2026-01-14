package lexical

import (
	"context"

	"github.com/hupe1980/vecgo/model"
)

// Index is the interface for a lexical search index.
type Index interface {
	// Add adds a document to the index.
	Add(id model.ID, text string) error
	// Delete removes a document from the index.
	Delete(id model.ID) error
	// Search performs a keyword search and returns a list of candidates.
	// The context can be used for cancellation.
	Search(ctx context.Context, text string, k int) ([]model.Candidate, error)
	// Close closes the index.
	Close() error
}
