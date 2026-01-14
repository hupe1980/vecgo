// Package diskann implements a DiskANN/Vamana segment for graph-based ANN search.
//
// # Overview
//
// DiskANN segments use the Vamana graph algorithm for approximate nearest
// neighbor search with high recall at low latency. They support multiple
// quantization methods for memory-efficient storage.
//
// # Features
//
//   - Vamana graph: Navigable small-world graph with robust pruning
//   - Lazy loading: Partial mmap support for large segments
//   - PQ compression: ~32x storage reduction
//   - RaBitQ: Random-bit quantization with Hamming distance
//   - INT4 quantization: 8x compression with SIMD distance
//   - Inverted index: Metadata filtering during search
//   - LZ4/ZSTD compression: Block-level compression for cold data
//
// # File Format
//
//	┌─────────────────────────────────────────┐
//	│ Header (512 bytes)                      │
//	├─────────────────────────────────────────┤
//	│ Vectors (RowCount × Dim × 4 bytes)      │
//	├─────────────────────────────────────────┤
//	│ IDs (RowCount × 8 bytes)                │
//	├─────────────────────────────────────────┤
//	│ Metadata Offsets (RowCount+1 × 8 bytes) │
//	│ Metadata Index (inverted index)         │
//	├─────────────────────────────────────────┤
//	│ Graph (RowCount × MaxDegree × 4 bytes)  │
//	├─────────────────────────────────────────┤
//	│ PQ/RaBitQ/INT4 Params                   │
//	│ Codes (RowCount × CodeSize bytes)       │
//	└─────────────────────────────────────────┘
//
// # Search Algorithm (Greedy Beam Search)
//
//  1. Start from entrypoint (medoid or random)
//  2. Maintain min-heap of candidates to explore
//  3. For each candidate: compute distance, explore neighbors
//  4. Prune: stop when best unexplored > worst in result set
//  5. Return top-k from result heap
//
// # Search Parameters
//
//   - L (search list): Controls recall vs latency (default: k + 100)
//   - RefineFactor: Multiplier for L (opts.RefineFactor)
//   - Filter: Predicate pushdown during search
//
// # Quantization Comparison
//
//	| Type   | Compression | Distance Cost | Recall  |
//	|--------|-------------|---------------|---------|
//	| None   | 1x          | ~20ns/vec     | 100%    |
//	| PQ     | ~32x        | ~50ns/vec     | 95-99%  |
//	| RaBitQ | ~32x        | ~15ns/vec     | 92-98%  |
//	| INT4   | 8x          | ~10ns/vec     | 98-99%  |
//
// # Thread Safety
//
// DiskANN segments are immutable and safe for concurrent reads.
// The Search method supports context cancellation for long traversals.
package diskann
