// Package vectorstore provides a canonical vector storage interface and a
// high-performance columnar implementation.
//
// # Architecture
//
// Columnar storage uses a Structure-of-Arrays (SOA) layout where vectors are stored
// contiguously in memory. This provides several advantages:
//
//   - Cache efficiency: Sequential vector access has optimal L1/L2 cache utilization
//   - SIMD friendliness: Contiguous data enables vectorized distance computations
//   - Zero-copy mmap: Files can be memory-mapped directly without deserialization
//   - Compaction: Deleted vectors are reclaimed via periodic compaction
//
// # File Format
//
// The columnar file format consists of:
//
//	+----------------+
//	|   FileHeader   |  64 bytes - magic, version, dimensions, counts
//	+----------------+
//	|  VectorData    |  count * dim * 4 bytes - contiguous float32 arrays
//	+----------------+
//	|  DeleteBitmap  |  (count + 7) / 8 bytes - deletion markers
//	+----------------+
//	|  VersionData   |  count * 8 bytes - vector versions (optional)
//	+----------------+
//	|   Checksum     |  4 bytes - CRC32 of entire file
//	+----------------+
//
// # Usage
//
// Create an in-memory store:
//
//	store := vectorstore.New(128, nil) // 128-dimensional vectors
//	store.Append(vec1)
//	store.Append(vec2)
//	v, ok := store.GetVector(0)
//
// Create a memory-mapped store from file:
//
//	store, closer, err := vectorstore.OpenMmap("vectors.col")
//	defer closer.Close()
//
// # Concurrency
//
// The store is safe for concurrent read access. Write operations (Append, Delete,
// Compact) require external synchronization.
//
// # Compaction
//
// Deleted vectors are marked but not immediately removed. Call Compact() to
// reclaim space and defragment the store:
//
//	store.DeleteVector(5)
//	store.DeleteVector(10)
//	newIDs := store.Compact() // Returns mapping from old ID to new ID
package vectorstore
