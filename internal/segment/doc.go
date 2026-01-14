// Package segment defines interfaces for immutable data segments.
//
// # Overview
//
// Segments are the fundamental unit of storage in Vecgo's LSM-tree architecture.
// Each segment is immutable once written, enabling lock-free concurrent reads
// and simplified crash recovery.
//
// # Architecture
//
//	┌─────────────────────────────────────────────────────────────┐
//	│                     Engine (LSM-Tree)                       │
//	│                                                             │
//	│  ┌─────────────┐                                           │
//	│  │  L0 Layer   │  MemTable (in-memory HNSW, 16 shards)     │
//	│  └──────┬──────┘                                           │
//	│         │ flush                                             │
//	│  ┌──────▼──────┐                                           │
//	│  │  L1 Layer   │  Flat/DiskANN segments (disk-resident)    │
//	│  └──────┬──────┘                                           │
//	│         │ compact                                           │
//	│  ┌──────▼──────┐                                           │
//	│  │  L2+ Layer  │  DiskANN segments (compressed)            │
//	│  └─────────────┘                                           │
//	└─────────────────────────────────────────────────────────────┘
//
// # Segment Types
//
//   - memtable: In-memory L0 segment with HNSW index for fast writes
//   - flat: Disk segment with exact brute-force search (optionally partitioned)
//   - diskann: Disk segment with Vamana graph (PQ/RaBitQ/INT4 compressed)
//
// # Segment Interface
//
// All segments implement the Segment interface:
//
//   - Search: Approximate/exact nearest neighbor search
//   - Rerank: Exact distance computation for candidates
//   - Fetch: Columnar data retrieval (vectors, metadata, payload)
//   - EvaluateFilter: Metadata filtering with inverted indexes
//
// # Filter Interface
//
// Filters enable efficient predicate pushdown:
//
//   - Matches: Per-row filter evaluation
//   - MatchesBatch: Vectorized batch evaluation
//   - MatchesBlock: Block-level statistics pruning
//   - AsBitmap: Bitmap representation for set operations
//
// # Thread Safety
//
// Segment instances are safe for concurrent reads. The Segment interface
// does not require any locking for read operations after initialization.
//
// # Memory Management
//
// Segments use mmap for zero-copy access to disk data when possible.
// The Advise method allows hinting kernel about access patterns for
// optimal page cache utilization.
package segment
