// Package bm25 provides a BM25-based lexical search index.
//
// BM25 (Best Matching 25) is a ranking function used for keyword search.
// This implementation uses an in-memory inverted index with efficient
// document-at-a-time (DAAT) scoring for optimal cache locality.
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
//	results, _ := db.HybridSearch(ctx, vector, "search query", 10, 60)
//
// # Parameters
//
// Uses standard BM25 parameters: k1=1.2, b=0.75
//
// # Thread Safety
//
// The index is safe for concurrent reads and writes.
// Writes acquire an exclusive lock, reads acquire a shared lock.
//
// # Performance
//
//   - DAAT scoring: Processes one document at a time for cache efficiency
//   - Pooled allocations: Iterators and heaps are pooled to reduce GC pressure
//   - ASCII fast path: ASCII text uses optimized tokenization
//   - O(terms) delete: Each document tracks its terms for efficient removal
//
// # Context Support
//
// Search respects context cancellation, checking periodically during scoring.
package bm25
