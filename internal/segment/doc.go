// Package segment defines interfaces for immutable data segments.
//
// Segments are the unit of storage in Vecgo's LSM-tree architecture.
// Each segment contains vectors, metadata, and an optional graph index.
//
// # Segment Types
//
//   - memtable: In-memory L0 segment (HNSW index)
//   - flat: Disk segment with exact search
//   - diskann: Disk segment with Vamana graph (PQ/RaBitQ compressed)
package segment
