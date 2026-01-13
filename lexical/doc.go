// Package lexical defines the interface for lexical (keyword) search indexes.
//
// Lexical indexes enable hybrid search by combining keyword matching with
// vector similarity using Reciprocal Rank Fusion (RRF).
//
// # Built-in Implementation
//
// The bm25 subpackage provides a BM25-based lexical index:
//
//	import "github.com/hupe1980/vecgo/lexical/bm25"
//
//	idx := bm25.New()
//	db, _ := vecgo.Open(vecgo.Local("./data"),
//	    vecgo.Create(128, vecgo.MetricL2),
//	    vecgo.WithLexicalIndex(idx, "text"),
//	)
//
// # Custom Implementations
//
// Implement the Index interface for custom lexical search:
//
//	type Index interface {
//	    Add(id model.ID, text string) error
//	    Delete(id model.ID) error
//	    Search(text string, k int) ([]model.Candidate, error)
//	    Close() error
//	}
package lexical
