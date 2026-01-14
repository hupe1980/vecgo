// Package engine implements the core vector database engine.
//
// The engine orchestrates:
//   - MemTable for hot writes (append-only, lock-free reads)
//   - Flat segments for medium-sized data
//   - DiskANN segments for large-scale ANN search
//   - Background flush and compaction loops
//   - MVCC snapshot isolation for concurrent readers
//   - Tombstone-based deletion with COW semantics
//   - WAL for durability and crash recovery
package engine
