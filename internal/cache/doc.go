// Package cache provides LRU caching for block data.
//
// # Block Cache
//
// The LRU block cache stores recently accessed data blocks from segments.
// It uses 64-way sharding for high concurrency (~18ns access under load).
//
// # Disk Cache
//
// For cloud storage backends, a two-tier cache is used:
//   - L1: RAM (LRU, configurable size)
//   - L2: Disk (persistent across restarts)
package cache
