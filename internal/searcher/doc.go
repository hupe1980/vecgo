// Package searcher provides pooled search context for zero-allocation queries.
//
// The Searcher struct owns all reusable resources needed for search:
//   - Priority queues (candidates, results)
//   - Visited sets (bitsets)
//   - Scratch vectors (decompression)
//   - IO buffers (disk reads)
//
// Searchers are managed by an engine-level pool for reuse across queries.
package searcher
