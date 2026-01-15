package engine

import (
	"math"
	"sync"
	"sync/atomic"

	imetadata "github.com/hupe1980/vecgo/internal/metadata"
	"github.com/hupe1980/vecgo/internal/segment"
)

const (
	chunkBits = 12             // 4096 items per chunk
	chunkSize = 1 << chunkBits // 4096
	chunkMask = chunkSize - 1

	// Number of sharded mutexes for write operations.
	// 64 shards = ~64KB total mutex overhead, provides excellent write parallelism.
	numShards = 64
	shardMask = numShards - 1
)

// tombstoneChunk holds a page of deletion LSNs.
type tombstoneChunk struct {
	lsns [chunkSize]uint64
}

// tombstoneState holds the directory of chunks for VersionedTombstones.
// The directory itself is immutable once created; chunks are copy-on-write.
type tombstoneState struct {
	chunks []*tombstoneChunk
	minLSN uint64
}

// VersionedTombstones tracks deletion LSNs using a sharded-mutex + COW architecture.
//
// Architecture:
//   - Reads are lock-free via atomic.Pointer (O(1) wait-free)
//   - Writes use sharded mutexes (by rowID) for zero-allocation updates
//   - Chunk data is copy-on-write for snapshot isolation
//
// This design provides:
//   - O(1) reads without locks
//   - O(1) writes with minimal contention (64 shards)
//   - Zero allocations on successful writes to existing chunks
//   - Memory efficiency via COW (only modified chunks are copied)
type VersionedTombstones struct {
	state atomic.Pointer[tombstoneState]

	// Sharded mutexes for write operations.
	// Shard selection: rowID % numShards
	// This eliminates CAS retry loops and their associated allocations.
	shards [numShards]sync.Mutex

	// Mutex for directory growth (rare operation).
	growMu sync.Mutex
}

func NewVersionedTombstones(initialCapacity int) *VersionedTombstones {
	vt := &VersionedTombstones{}
	// Initialize with enough chunks to cover initialCapacity
	numChunks := (initialCapacity + chunkSize - 1) / chunkSize
	if numChunks == 0 {
		numChunks = 1 // Minimum one chunk slot (can be nil)
	}

	vt.state.Store(&tombstoneState{
		chunks: make([]*tombstoneChunk, numChunks),
		minLSN: math.MaxUint64,
	})
	return vt
}

// MarkDeleted marks a row as deleted at the given LSN.
// Uses sharded mutexes for zero-allocation updates with minimal contention.
// Reads remain lock-free via atomic state pointer.
func (vt *VersionedTombstones) MarkDeleted(rowID uint32, lsn uint64) {
	chunkIdx := int(rowID) >> chunkBits
	offset := int(rowID) & chunkMask
	shardIdx := rowID & shardMask

	// Fast path: check if already deleted (lock-free read)
	curr := vt.state.Load()
	if chunkIdx < len(curr.chunks) {
		if chunk := curr.chunks[chunkIdx]; chunk != nil {
			if currentLSN := chunk.lsns[offset]; currentLSN != 0 && currentLSN <= lsn {
				return // Already deleted at an earlier (or same) LSN
			}
		}
	}

	// Acquire shard lock for this rowID
	vt.shards[shardIdx].Lock()
	defer vt.shards[shardIdx].Unlock()

	// Re-check under lock (another goroutine may have updated)
	curr = vt.state.Load()
	if chunkIdx < len(curr.chunks) {
		if chunk := curr.chunks[chunkIdx]; chunk != nil {
			if currentLSN := chunk.lsns[offset]; currentLSN != 0 && currentLSN <= lsn {
				return
			}
		}
	}

	// Check if directory needs to grow (rare)
	if chunkIdx >= len(curr.chunks) {
		vt.growDirectory(chunkIdx)
		curr = vt.state.Load()
	}

	// Prepare new chunk (COW) or create new
	var newChunk *tombstoneChunk
	if curr.chunks[chunkIdx] != nil {
		// Copy existing chunk
		copied := *curr.chunks[chunkIdx]
		newChunk = &copied
	} else {
		newChunk = &tombstoneChunk{}
	}

	// Update LSN
	newChunk.lsns[offset] = lsn

	// Create new directory with updated chunk pointer
	newChunks := make([]*tombstoneChunk, len(curr.chunks))
	copy(newChunks, curr.chunks)
	newChunks[chunkIdx] = newChunk

	// Update minLSN
	minLSN := curr.minLSN
	if lsn < minLSN {
		minLSN = lsn
	}

	// Publish new state atomically
	vt.state.Store(&tombstoneState{
		chunks: newChunks,
		minLSN: minLSN,
	})
}

// growDirectory expands the chunk directory to accommodate chunkIdx.
// Called under shard lock; acquires growMu for directory expansion.
func (vt *VersionedTombstones) growDirectory(chunkIdx int) {
	vt.growMu.Lock()
	defer vt.growMu.Unlock()

	curr := vt.state.Load()
	if chunkIdx < len(curr.chunks) {
		return // Another goroutine already grew it
	}

	// Double growth strategy
	newNumChunks := chunkIdx + 1
	if newNumChunks < 2*len(curr.chunks) {
		newNumChunks = 2 * len(curr.chunks)
	}

	newChunks := make([]*tombstoneChunk, newNumChunks)
	copy(newChunks, curr.chunks)

	vt.state.Store(&tombstoneState{
		chunks: newChunks,
		minLSN: curr.minLSN,
	})
}

// IsDeleted returns true if the row is deleted at the snapshot LSN.
// This operation is wait-free.
func (vt *VersionedTombstones) IsDeleted(rowID uint32, snapshotLSN uint64) bool {
	state := vt.state.Load()
	chunkIdx := int(rowID) >> chunkBits

	if chunkIdx >= len(state.chunks) {
		return false
	}

	chunk := state.chunks[chunkIdx]
	if chunk == nil {
		return false
	}

	delLSN := chunk.lsns[int(rowID)&chunkMask]
	return delLSN != 0 && delLSN <= snapshotLSN
}

// ToBitmap converts the tombstones to a LocalBitmap visible at snapshotLSN.
func (vt *VersionedTombstones) ToBitmap(snapshotLSN uint64) *imetadata.LocalBitmap {
	state := vt.state.Load()
	bm := imetadata.NewLocalBitmap()

	for i, chunk := range state.chunks {
		if chunk == nil {
			continue
		}
		baseID := uint32(i) << chunkBits
		for j, lsn := range chunk.lsns {
			if lsn != 0 && lsn <= snapshotLSN {
				bm.Add(baseID + uint32(j))
			}
		}
	}
	return bm
}

// LoadFromBitmap populates tombstones from a bitmap with LSN 1 (Legacy/Restart).
// This is a bulk operation used during recovery; not a hot path.
func (vt *VersionedTombstones) LoadFromBitmap(bm *imetadata.LocalBitmap) {
	if bm == nil || bm.Cardinality() == 0 {
		return
	}

	// Acquire grow lock for bulk operation (recovery is single-threaded)
	vt.growMu.Lock()
	defer vt.growMu.Unlock()

	// Collect all IDs and find max
	var maxID uint32
	ids := make([]uint32, 0, bm.Cardinality())
	for id := range bm.Iterator() {
		ids = append(ids, uint32(id))
		if uint32(id) > maxID {
			maxID = uint32(id)
		}
	}

	curr := vt.state.Load()

	// Ensure directory is large enough
	maxChunkIdx := int(maxID) >> chunkBits
	newNumChunks := len(curr.chunks)
	if maxChunkIdx >= newNumChunks {
		newNumChunks = maxChunkIdx + 1
	}

	newChunks := make([]*tombstoneChunk, newNumChunks)
	copy(newChunks, curr.chunks)

	// Track modified chunks (COW)
	modifiedChunks := make(map[int]*tombstoneChunk, len(ids)/chunkSize+1)

	for _, id := range ids {
		cIdx := int(id) >> chunkBits
		off := int(id) & chunkMask

		chunk, exists := modifiedChunks[cIdx]
		if !exists {
			if cIdx < len(curr.chunks) && curr.chunks[cIdx] != nil {
				copied := *curr.chunks[cIdx]
				chunk = &copied
			} else {
				chunk = &tombstoneChunk{}
			}
			modifiedChunks[cIdx] = chunk
			newChunks[cIdx] = chunk
		}
		chunk.lsns[off] = 1 // LSN 1 for legacy/bitmap override
	}

	minLSN := curr.minLSN
	if minLSN > 1 {
		minLSN = 1
	}

	vt.state.Store(&tombstoneState{
		chunks: newChunks,
		minLSN: minLSN,
	})
}

// TombstoneFilter adapts VersionedTombstones to segment.Filter interface
type TombstoneFilter struct {
	state       *tombstoneState // Immutable snapshot of deletion state
	snapshotLSN uint64
}

// NewTombstoneFilter creates a new filter with a snapshot of the current state.
func NewTombstoneFilter(vt *VersionedTombstones, snapshotLSN uint64) *TombstoneFilter {
	st := vt.state.Load()
	if st.minLSN > snapshotLSN {
		st = nil
	}
	return &TombstoneFilter{
		state:       st,
		snapshotLSN: snapshotLSN,
	}
}

func (tf *TombstoneFilter) Matches(rowID uint32) bool {
	if tf.state == nil {
		return true
	}

	chunkIdx := int(rowID) >> chunkBits
	if chunkIdx >= len(tf.state.chunks) {
		return true // Chunk not allocated, impossible to be deleted
	}

	chunk := tf.state.chunks[chunkIdx]
	if chunk == nil {
		return true
	}

	lsn := chunk.lsns[int(rowID)&chunkMask]
	return lsn == 0 || lsn > tf.snapshotLSN
}

func (tf *TombstoneFilter) MatchesBatch(ids []uint32, out []bool) {
	// Optimization: Load state once
	state := tf.state
	if state == nil {
		for i := range ids {
			out[i] = true
		}
		return
	}

	for i, id := range ids {
		chunkIdx := int(id) >> chunkBits
		if chunkIdx >= len(state.chunks) {
			out[i] = true // Alive
			continue
		}
		chunk := state.chunks[chunkIdx]
		if chunk == nil {
			out[i] = true // Alive
			continue
		}
		delLSN := chunk.lsns[int(id)&chunkMask]
		out[i] = delLSN == 0 || delLSN > tf.snapshotLSN
	}
}

func (tf *TombstoneFilter) MatchesBlock(stats map[string]segment.FieldStats) bool {
	return true
}

func (tf *TombstoneFilter) AsBitmap() segment.Bitmap {
	// TombstoneFilter is an exclusion filter (sparse deletions).
	// Returning nil forces usage of Matches/MatchesBatch.
	return nil
}

// Count returns the number of deleted rows visible at the given snapshot LSN.
func (vt *VersionedTombstones) Count(snapshotLSN uint64) int {
	state := vt.state.Load()
	count := 0
	for _, chunk := range state.chunks {
		if chunk == nil {
			continue
		}
		for _, lsn := range chunk.lsns {
			if lsn != 0 && lsn <= snapshotLSN {
				count++
			}
		}
	}
	return count
}

// acquireTombstoneFilter gets a filter from the pool or creates a new one.
func (e *Engine) acquireTombstoneFilter(vt *VersionedTombstones, lsn uint64) *TombstoneFilter {
	if v := e.tombstoneFilterPool.Get(); v != nil {
		tf := v.(*TombstoneFilter)
		st := vt.state.Load()
		if st.minLSN > lsn {
			st = nil
		}
		tf.state = st
		tf.snapshotLSN = lsn
		return tf
	}
	return NewTombstoneFilter(vt, lsn)
}

// releaseTombstoneFilter puts the filter back into the pool.
// It accepts segment.Filter interface to simplify call sites.
func (e *Engine) releaseTombstoneFilter(f segment.Filter) {
	if f == nil {
		return
	}
	if tf, ok := f.(*TombstoneFilter); ok {
		tf.state = nil // clear reference
		e.tombstoneFilterPool.Put(tf)
	}
}
