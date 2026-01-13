// Package memtable implements the in-memory L0 segment with HNSW indexing.
//
// # Features
//
//   - 16-way sharding for write concurrency
//   - Lock-free HNSW search
//   - Arena-based allocation
//
// # Lifecycle
//
// MemTables are created for writes, then flushed to disk as Flat or DiskANN
// segments when they reach the configured size threshold.
package memtable
