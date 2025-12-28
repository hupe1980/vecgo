// Package engine provides the coordinator layer for Vecgo.
//
// The coordinator orchestrates all database operations, integrating:
//   - Index management (single-shard or multi-shard)
//   - Metadata storage and filtering
//   - Write-Ahead Log (WAL) for durability
//   - Transaction coordination
//
// # Coordinator Architecture
//
// Single-Shard Mode (default):
//   - Direct pass-through to index
//   - Single lock for all writes
//   - Simple and efficient for small-medium workloads
//
// Multi-Shard Mode (.Shards(n) in builder):
//   - Hash-based vector distribution across N shards
//   - Independent locks per shard → parallel writes
//   - Fan-out search to all shards, merge results
//   - 2.7x-3.4x write speedup on multi-core systems
//
// # Transaction Model
//
// All operations are atomic and thread-safe:
//
//   - Insert: Acquire lock → Add to index → Update metadata → Append WAL → Release
//   - Update: Lookup shard → Acquire lock → Update in-place → Release
//   - Delete: Soft delete (tombstone) → Trigger compaction if threshold exceeded
//   - Search: Read-only, concurrent with writes (lock-free in HNSW)
//
// # WAL Integration
//
// The coordinator manages WAL lifecycle:
//
//   - Append operations to active segment
//   - Group commit for batched fsync (83x faster than sync)
//   - Auto-checkpoint to prevent unbounded log growth
//   - Crash recovery via replay
//
// See package wal for durability configuration.
//
// # Subpackages
//
//   - idalloc: ID allocation for vector storage
package engine
