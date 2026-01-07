package blobstore

import (
	"context"
	"io"

	"github.com/hupe1980/vecgo/cache"
)

// CachingStore wraps a BlobStore and adds block-level caching.
type CachingStore struct {
	inner     BlobStore
	cache     cache.BlockCache
	blockSize int64
}

// NewCachingStore creates a new CachingStore.
// blockSize defaults to 4KB if <= 0.
func NewCachingStore(inner BlobStore, cache cache.BlockCache, blockSize int64) *CachingStore {
	if blockSize <= 0 {
		blockSize = 4096
	}
	return &CachingStore{
		inner:     inner,
		cache:     cache,
		blockSize: blockSize,
	}
}

func (s *CachingStore) Open(ctx context.Context, name string) (Blob, error) {
	b, err := s.inner.Open(ctx, name)
	if err != nil {
		return nil, err
	}
	return &CachingBlob{
		inner:     b,
		cache:     s.cache,
		name:      name,
		blockSize: s.blockSize,
	}, nil
}

func (s *CachingStore) Create(ctx context.Context, name string) (WritableBlob, error) {
	// We don't cache writes, only reads. Invalidating cache on write might be needed
	// if we support mutable blobs, but Segments are immutable.
	// However, if we overwrite an existing blob, we should probably invalidate.
	// For now, simple pass-through.
	return s.inner.Create(ctx, name)
}

func (s *CachingStore) Delete(ctx context.Context, name string) error {
	// Invalidate cache entries for this blob
	s.cache.Invalidate(func(key cache.CacheKey) bool {
		return key.Kind == cache.CacheKindBlob && key.Path == name
	})
	return s.inner.Delete(ctx, name)
}

func (s *CachingStore) List(ctx context.Context, prefix string) ([]string, error) {
	return s.inner.List(ctx, prefix)
}

// CachingBlob wraps a Blob and uses the block cache for reads.
type CachingBlob struct {
	inner     Blob
	cache     cache.BlockCache
	name      string
	blockSize int64
}

func (b *CachingBlob) Close() error {
	return b.inner.Close()
}

func (b *CachingBlob) Size() int64 {
	return b.inner.Size()
}

func (b *CachingBlob) ReadAt(p []byte, off int64) (int, error) {
	if len(p) == 0 {
		return 0, nil
	}

	totalRead := 0

	// Align to block boundaries
	startBlock := off / b.blockSize
	endBlock := (off + int64(len(p)) - 1) / b.blockSize

	for blk := startBlock; blk <= endBlock; blk++ {
		// Calculate block range
		blkStart := blk * b.blockSize
		// Offset within the block for the *request* (not the block itself)
		// For the first block, we might start in the middle.
		// For subsequent blocks, we start at 0.

		// Map block coordinates to output buffer coordinates
		// Data from this block that intersects with [off, off+len(p))

		// Intersection:
		// [blkStart, blkStart+b.blockSize)
		// [off, off+len(p))

		intersectStart := max(blkStart, off)
		intersectEnd := min(blkStart+b.blockSize, off+int64(len(p)))

		if intersectEnd <= intersectStart {
			continue // Should not happen given logic loop
		}

		copySize := int(intersectEnd - intersectStart)

		// Where in 'p' does this go?
		dstOffset := intersectStart - off

		// Get block data (either from cache or read from inner)
		blockData, err := b.dofetchBlock(blk)
		if err != nil {
			// If EOF on a block read, it might be partial?
			// But specific block logic handles boundaries.
			return totalRead, err
		}

		// Where in 'blockData' do we read from?
		srcOffset := intersectStart - blkStart

		// Safety check
		if srcOffset+int64(copySize) > int64(len(blockData)) {
			// This might happen if file size is not a multiple of block size
			// and we are at the last block.
			copySize = len(blockData) - int(srcOffset)
		}

		if copySize > 0 {
			n := copy(p[dstOffset:dstOffset+int64(copySize)], blockData[srcOffset:])
			totalRead += n
		}
	}

	return totalRead, nil
}

func (b *CachingBlob) dofetchBlock(blkIdx int64) ([]byte, error) {
	key := cache.CacheKey{
		Kind:   cache.CacheKindBlob,
		Path:   b.name,
		Offset: uint64(blkIdx),
	}

	// 1. Try Cache
	if data, ok := b.cache.Get(context.Background(), key); ok {
		return data, nil
	}

	// 2. Read from Inner
	// Allocate full block
	buf := make([]byte, b.blockSize)
	offset := blkIdx * b.blockSize

	// ReadAt might return fewer bytes if EOF is reached
	n, err := b.inner.ReadAt(buf, offset)
	if err != nil && err != io.EOF {
		return nil, err
	}

	validData := buf[:n]

	// 3. Cache it (only if we got data)
	if n > 0 {
		// Make a copy to ensure ownership?
		// ReadAt usually copies into buf.
		// We cache 'validData'.
		// Note: We might want to shrink capacity if n < blockSize?
		// For simplicity, just cache validData.
		b.cache.Set(context.Background(), key, validData)
	}

	return validData, nil
}

// ReadRange optimizes for larger reads by bypassing cache or pre-warming?
// For now, simpler implementation: utilize the generic ReadRange which usually calls ReadAt.
// OR, we can just use the default implementation or delegate to ReadAt loop.
func (b *CachingBlob) ReadRange(off, len int64) (io.ReadCloser, error) {
	// TODO: For very large ranges, we might bypass cache to avoid thrashing?
	// For now, just use SectionReader which calls ReadAt.
	return io.NopCloser(io.NewSectionReader(b, off, len)), nil
}

func max(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}

func min(a, b int64) int64 {
	if a < b {
		return a
	}
	return b
}
