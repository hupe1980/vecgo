package cache

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/hupe1980/vecgo/model"
)

// DiskCacheConfig holds configuration for the disk cache.
type DiskCacheConfig struct {
	// RootDir is the directory where cache files are stored.
	RootDir string
	// MaxSizeBytes is the maximum size of the cache in bytes.
	MaxSizeBytes int64
}

// DiskBlockCache implements BlockCache backed by the local filesystem.
// It maintains an in-memory LRU index of the files on disk.
type DiskBlockCache struct {
	mu          sync.Mutex
	rootDir     string
	maxSize     int64
	currentSize int64

	// Index
	items   map[CacheKey]*lruEntry
	lruHead *lruEntry
	lruTail *lruEntry
	wg      sync.WaitGroup
}

type lruEntry struct {
	key        CacheKey
	size       int64
	filePath   string
	next, prev *lruEntry
}

// NewDiskBlockCache creates a new disk-backed block cache.
// It scans the directory to rebuild the index on startup (async).
func NewDiskBlockCache(config DiskCacheConfig) (*DiskBlockCache, error) {
	if err := os.MkdirAll(config.RootDir, 0755); err != nil {
		return nil, err
	}

	c := &DiskBlockCache{
		rootDir: config.RootDir,
		maxSize: config.MaxSizeBytes,
		items:   make(map[CacheKey]*lruEntry),
	}

	// Synchronous scan for now to ensure consistency,
	// or we can make it async but then Get() might miss existing files until scan done.
	// For "Best in class", we want fast startup.
	// Let's reload state in background, but Get() checks disk if misses memory?
	// No, if we don't know it's there, we might overwrite or double usage.
	// Let's fast scan.
	c.scanExistingFiles()

	return c, nil
}

func (c *DiskBlockCache) scanExistingFiles() {
	// Walk the directory
	// Expect structure: root/<Path>/<encoded_key>.blk
	// Or simplistic: just flatten everything if Path isn't directory-friendly?
	// But BlobStore uses actual file paths.

	_ = filepath.Walk(c.rootDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return nil // Skip errors
		}
		if info.IsDir() {
			return nil
		}

		// Parse filepath to Key
		key, ok := c.parsePathToKey(path)
		if !ok {
			return nil
		}

		c.addToLRU(key, path, info.Size())
		return nil
	})
}

// encodeKeyToRelPath creates a relative path string from a key.
// Format: <Path>/<Kind>-<SegmentID>-<Offset>.blk
// The <Path> part is preserved as directory structure.
func (c *DiskBlockCache) encodeKeyToRelPath(key CacheKey) string {
	fileName := fmt.Sprintf("%d-%d-%d.blk", key.Kind, key.SegmentID, key.Offset)
	if key.Path != "" {
		return filepath.Join(key.Path, fileName)
	}
	return filepath.Join("_misc", fileName)
}

func (c *DiskBlockCache) parsePathToKey(absPath string) (CacheKey, bool) {
	relPath, err := filepath.Rel(c.rootDir, absPath)
	if err != nil {
		return CacheKey{}, false
	}

	dir, file := filepath.Split(relPath)
	// dir might be "path/to/file/"
	// file is "kind-seg-off.blk"

	var k CacheKey
	var kind int
	var segID model.SegmentID
	var off uint64

	n, err := fmt.Sscanf(file, "%d-%d-%d.blk", &kind, &segID, &off)
	if err != nil || n != 3 {
		return CacheKey{}, false
	}

	k.Kind = CacheKind(kind)
	k.SegmentID = segID
	k.Offset = off

	// Reconstruct Path from dir
	if dir != "" {
		// remove trailing slash provided by Split
		dir = strings.TrimSuffix(dir, string(filepath.Separator))
		if dir == "_misc" {
			k.Path = ""
		} else {
			k.Path = dir
		}
	}

	return k, true
}

func (c *DiskBlockCache) Get(ctx context.Context, key CacheKey) ([]byte, bool) {
	c.mu.Lock()
	ent, ok := c.items[key]
	if ok {
		c.moveToFront(ent)
	}
	c.mu.Unlock()

	if !ok {
		return nil, false
	}

	data, err := os.ReadFile(ent.filePath)
	if err != nil {
		// File missing? Remove from index
		c.mu.Lock()
		c.removeEntry(ent)
		c.mu.Unlock()
		return nil, false
	}
	return data, true
}

func (c *DiskBlockCache) Set(ctx context.Context, key CacheKey, b []byte) {
	c.mu.Lock()
	defer c.mu.Unlock()

	// 1. Check if exists
	if ent, ok := c.items[key]; ok {
		c.moveToFront(ent)
		// Assuming immutable blocks, we don't rewrite.
		// But if size changed (corruption fix?), we might.
		// For now, ignore overwrite.
		return
	}

	// 2. Prepare write
	size := int64(len(b))
	relPath := c.encodeKeyToRelPath(key)
	absPath := filepath.Join(c.rootDir, relPath)

	// Create dir if needed
	if err := os.MkdirAll(filepath.Dir(absPath), 0755); err != nil {
		return
	}

	// 3. Evict if needed (Reserve space)
	for c.currentSize+size > c.maxSize {
		if c.lruTail == nil {
			// Cache full but empty? Only happens if single item > max size
			break
		}
		c.evictOne()
	}

	// 4. Write to disk
	// To avoid blocking the search path on disk I/O, we could make this async.
	// But simple goroutines might pile up.
	// We'll use a semaphore or worker queue if this becomes a bottleneck.
	// For "Best-in-Class" production, let's just do it in a goroutine but verify no race on index.
	//
	// Issue: if we return early, Get() might see it in index but file not ready.
	// Solution: Update Index *after* write.
	// But then we might have parallel writes for same key.
	//
	// Let's stick to synchronous write for correctness for now, unless profiled.
	// ACTUALLY: The user asked for "fastest" solution. Blocking 4MB write to disk takes 10-50ms.
	// This adds 10-50ms to the S3 fetch (which is 50-100ms).
	// It's significant.
	//
	// Proposed Async Flow:
	// 1. Add to separate "inflight" map (or use index with "writing" state).
	// 2. Launch goroutine to write.
	// 3. On complete, update state.
	// 4. Get() checks index. If "writing", wait or return data from memory if we kept it?
	//
	// Simpler: Just allow Set to return.
	// The caller (CachingStore) holds 'b'.
	// We can update index immediately pointing to path, but file might be partial.
	//
	// Better: Don't update index until write complete.
	// Parallel Gets will miss cache and hit S3 again. That's acceptable for a "warm up" phase.
	c.wg.Add(1)
	go func() {
		defer c.wg.Done()
		// Create dir if needed
		if err := os.MkdirAll(filepath.Dir(absPath), 0755); err != nil {
			return
		}

		tmpFile, err := os.CreateTemp(filepath.Dir(absPath), "tmp-blk-*")
		if err != nil {
			return
		}
		tmpName := tmpFile.Name()

		defer func() {
			if _, err := os.Stat(tmpName); err == nil {
				_ = os.Remove(tmpName)
			}
		}()

		if _, err := tmpFile.Write(b); err != nil {
			_ = tmpFile.Close() // Intentionally ignore: cleanup path
			return
		}
		if err := tmpFile.Close(); err != nil {
			return
		}

		if err := os.Rename(tmpName, absPath); err != nil {
			return
		}

		// 5. Update Index (Need lock)
		c.mu.Lock()
		defer c.mu.Unlock()

		// Recheck capacity in case other writes happened
		for c.currentSize+size > c.maxSize {
			if c.lruTail == nil {
				break
			}
			c.evictOne()
		}

		c.addToLRU(key, absPath, size)
	}()
}

func (c *DiskBlockCache) Invalidate(predicate func(key CacheKey) bool) {
	c.mu.Lock()
	defer c.mu.Unlock()

	var toRemove []*lruEntry
	for k, ent := range c.items {
		if predicate(k) {
			toRemove = append(toRemove, ent)
		}
	}

	for _, ent := range toRemove {
		_ = os.Remove(ent.filePath) // Delete file
		c.removeEntry(ent)          // Update index/size
	}

	// Optimization: If predicate matches a Path, we could RemoveAll the directory.
	// But predicate is a function.
}

// Close waits for all background writes to complete.
func (c *DiskBlockCache) Close() error {
	c.wg.Wait()
	return nil
}

func (c *DiskBlockCache) Stats() (hits, misses int64) {
	// Not implemented for disk cache yet
	return 0, 0
}

// Internal LRU helpers (must hold lock)

func (c *DiskBlockCache) addToLRU(key CacheKey, path string, size int64) {
	ent := &lruEntry{
		key:      key,
		filePath: path,
		size:     size,
	}
	c.items[key] = ent
	c.currentSize += size

	// Push Front
	if c.lruHead == nil {
		c.lruHead = ent
		c.lruTail = ent
	} else {
		ent.next = c.lruHead
		c.lruHead.prev = ent
		c.lruHead = ent
	}
}

func (c *DiskBlockCache) moveToFront(ent *lruEntry) {
	if c.lruHead == ent {
		return
	}

	// Detach
	if ent.prev != nil {
		ent.prev.next = ent.next
	}
	if ent.next != nil {
		ent.next.prev = ent.prev
	}
	if c.lruTail == ent {
		c.lruTail = ent.prev
	}

	// Attach Front
	ent.next = c.lruHead
	ent.prev = nil
	if c.lruHead != nil {
		c.lruHead.prev = ent
	}
	c.lruHead = ent
	if c.lruTail == nil {
		c.lruTail = ent
	}
}

func (c *DiskBlockCache) removeEntry(ent *lruEntry) {
	if ent.prev != nil {
		ent.prev.next = ent.next
	} else {
		c.lruHead = ent.next
	}

	if ent.next != nil {
		ent.next.prev = ent.prev
	} else {
		c.lruTail = ent.prev
	}

	delete(c.items, ent.key)
	c.currentSize -= ent.size
}

func (c *DiskBlockCache) evictOne() {
	if c.lruTail == nil {
		return
	}
	// Delete file
	_ = os.Remove(c.lruTail.filePath) // ignore error
	c.removeEntry(c.lruTail)
}
