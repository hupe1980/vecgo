// Package bitset provides a lock-free segmented bitset for concurrent access.
//
// Architecture:
//   - Segmented design: 64KB segments (1024 uint64 words = 65536 bits each)
//   - Lock-free: atomic.Pointer for segment array, atomic.Uint64 for words
//   - Lazy allocation: segments allocated on first write
//
// Used internally for:
//   - Tombstone tracking (deleted record IDs)
//   - Metadata filter results (pre-computed bitmaps)
package bitset
