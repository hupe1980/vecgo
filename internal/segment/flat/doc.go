// Package flat implements an immutable Flat segment with brute-force search.
//
// # Overview
//
// Flat segments store vectors with optional quantization and partitioning.
// They provide exact search (or high-recall approximate with quantization)
// through linear scanning with SIMD-optimized distance computation.
//
// # Features
//
//   - Full-precision vectors: Zero-copy mmap access
//   - SQ8 quantization: 4x storage reduction, SIMD batch distance
//   - PQ quantization: ~32x compression with ADC distance
//   - K-means partitioning: IVF-style search with nprobe control
//   - Block skipping: Field statistics enable block-level pruning
//   - Inverted index: Fast metadata filtering
//
// # File Format
//
//	┌─────────────────────────────────────────┐
//	│ Header (256 bytes)                      │
//	├─────────────────────────────────────────┤
//	│ Vectors (RowCount × Dim × 4 bytes)      │
//	├─────────────────────────────────────────┤
//	│ IDs (RowCount × 8 bytes)                │
//	├─────────────────────────────────────────┤
//	│ Metadata Offsets (RowCount+1 × 4 bytes) │
//	│ Metadata Blob (variable)                │
//	├─────────────────────────────────────────┤
//	│ Quantization (SQ8: mins/maxs + codes)   │
//	│            or (PQ: codebooks + codes)   │
//	├─────────────────────────────────────────┤
//	│ Centroids (NumPartitions × Dim × 4)     │
//	│ Partition Offsets (NumPartitions+1 × 4) │
//	├─────────────────────────────────────────┤
//	│ Block Stats (BlockCount × variable)     │
//	└─────────────────────────────────────────┘
//
// # Search Algorithm
//
//  1. If partitioned: Find closest centroids via nprobe
//  2. For each block: Check block stats (skip if filter doesn't match)
//  3. For each row: Apply filter, compute distance (SQ8/PQ/exact)
//  4. Maintain top-k heap with bounded push
//
// # Performance
//
//   - SQ8 L2 batch: ~15ns per vector (SIMD)
//   - Exact L2: ~25ns per vector (256-dim, SIMD)
//   - Block skipping: Up to 10x speedup on selective filters
//
// # Thread Safety
//
// Flat segments are immutable and safe for concurrent reads.
// The Search method supports context cancellation for long scans.
package flat
