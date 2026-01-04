package lexical

import "github.com/hupe1980/vecgo/model"

// Index is the interface for a lexical search index.
type Index interface {
	// Add adds a document to the index.
	Add(pk model.PrimaryKey, text string) error
	// Delete removes a document from the index.
	Delete(pk model.PrimaryKey) error
	// Search performs a keyword search and returns a map of PK -> Score.
	Search(text string) (map[model.PrimaryKey]float32, error)
	// Close closes the index.
	Close() error
}
