// Package diskann implements a disk-resident approximate nearest neighbor index.
//
// DiskANN is based on the Vamana graph algorithm from Microsoft Research, designed
// for billion-scale vector search on commodity hardware. It enables searching vectors
// that exceed RAM capacity by storing full vectors on SSD while keeping a compressed
// navigation structure in memory.
//
// # Architecture
//
// DiskANN uses a three-tier architecture:
//
//  1. Navigation Graph (RAM): Vamana graph structure with PQ-compressed vectors
//     - ~10-20 bytes per vector (vs 3KB for 768-dim float32)
//     - Enables fast approximate traversal without disk I/O
//
//  2. PQ Codebooks (RAM): Product quantization centroids for distance approximation
//     - Shared across all vectors
//     - Used for graph traversal and candidate ranking
//
//  3. Full Vectors (SSD): Original float32 vectors stored on disk
//     - Accessed only for final reranking of top candidates
//     - SSD-optimized layout for parallel reads
//
// # Algorithm
//
// Building:
//   - Construct Vamana graph with pruned edges (similar to HNSW but optimized for SSD)
//   - Train PQ codebooks on training vectors
//   - Store compressed PQ codes in RAM, full vectors on disk
//
// Search:
//   - Beam search through Vamana graph using PQ distances (no disk I/O)
//   - Collect top candidates based on approximate distances
//   - Parallel disk reads to fetch full vectors for final candidates
//   - Rerank using exact distances
//
// # Performance Characteristics
//
//   - Capacity: 1B+ vectors on commodity hardware
//   - RAM usage: ~10-20 bytes per vector (vs 3KB for 768-dim)
//   - Query latency: 5-20ms on NVMe SSD (vs <1ms for in-memory HNSW)
//   - Throughput: 100s-1000s QPS with SSD parallelism
//
// # File Format
//
// DiskANN uses a multi-file format:
//   - index.meta: Header, configuration, PQ codebooks
//   - index.graph: Vamana graph adjacency lists
//   - index.pqcodes: PQ-compressed vectors (M bytes per vector)
//   - index.vectors: Full float32 vectors (for reranking)
//
// All files support memory-mapping for efficient access.
package diskann
