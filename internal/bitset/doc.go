// Package bitset provides Roaring Bitmap-based bitsets for efficient filtering.
//
// Used internally for:
//   - Tombstone tracking (deleted record IDs)
//   - Metadata filter results (pre-computed bitmaps)
//   - Visited set tracking during graph traversal
package bitset
