package cache

import (
	"context"

	"github.com/hupe1980/vecgo/model"
)

// CacheKind is used to separate key spaces and tuning.
type CacheKind uint8

const (
	CacheKindUnknown      CacheKind = iota
	CacheKindColumnBlocks           // vector/code/rowid/payload column blocks
	CacheKindPostings               // filter posting lists / bitmaps
	CacheKindGraph                  // DiskANN adjacency / node blocks
	CacheKindBlob                   // Generic blob store blocks
)

// CacheKey must be stable across processes and snapshot-safe.
// If the cached value depends on visibility, include ManifestID.
type CacheKey struct {
	Kind      CacheKind
	SegmentID model.SegmentID
	// ManifestID is optional; include it for snapshot-dependent entries (e.g. result cache).
	ManifestID uint64
	// Offset is a logical block identifier (e.g., byte offset / block index / node id).
	Offset uint64
	// Path is optional; if provided, it identifies the source (e.g. filename).
	// Used by generic blob caching when SegmentID is not known or sufficient.
	Path string
}

// BlockCache is a byte-oriented cache for immutable blocks.
// Returned slices must be treated as read-only.
type BlockCache interface {
	// Get returns a cached block. ok=false if missing.
	Get(ctx context.Context, key CacheKey) (b []byte, ok bool)
	// Set caches a block. Implementations may copy or retain; caller must treat b as immutable.
	Set(ctx context.Context, key CacheKey, b []byte)
	// Invalidate removes entries matching the predicate.
	Invalidate(predicate func(key CacheKey) bool)
	// Close releases any resources (e.g. background workers).
	Close() error
	// Stats returns cache statistics.
	Stats() (hits, misses int64)
}

// AdmissionPolicy decides whether a value should be cached.
// Start simple (e.g., “cache on second hit” or size-based).
type AdmissionPolicy interface {
	Admit(key CacheKey, sizeBytes int) bool
}
