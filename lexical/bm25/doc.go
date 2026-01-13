// Package bm25 provides a BM25-based lexical search index.
//
// BM25 (Best Matching 25) is a ranking function used for keyword search.
// This implementation uses an in-memory inverted index with efficient
// document-at-a-time (DAAT) scoring.
//
// # Usage
//
//	idx := bm25.New()
//	db, _ := vecgo.Open(vecgo.Local("./data"),
//	    vecgo.Create(128, vecgo.MetricL2),
//	    vecgo.WithLexicalIndex(idx, "text"),
//	)
//
//	// Hybrid search combines BM25 + vector similarity
//	results, _ := db.HybridSearch(ctx, vector, "search query", 10)
//
// # Parameters
//
// Uses standard BM25 parameters: k1=1.2, b=0.75
//
// # Thread Safety
//
// The index is safe for concurrent reads and writes.
package bm25
