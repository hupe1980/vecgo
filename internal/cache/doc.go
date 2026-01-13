// Package cache provides LRU caching for block data.
//
// # Block Cache (RAM)
//
// The ShardedLRUBlockCache stores recently accessed data blocks from segments.
// It uses 64-way sharding for high concurrency (~18ns access under parallel load).
//
// Key features:
//   - Lock-free shard selection using splitmix64 hash
//   - Per-shard mutex for minimal contention
//   - Integrated with ResourceController for memory limits
//
// # Disk Cache (L2)
//
// For cloud storage backends, DiskBlockCache provides a persistent L2 cache:
//   - Async writes to avoid blocking the search path
//   - LRU eviction with configurable size limits
//   - Rebuilds index from disk on startup
package cache
