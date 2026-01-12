package engine

import (
	"math"
	"sync/atomic"

	imetadata "github.com/hupe1980/vecgo/internal/metadata"
	"github.com/hupe1980/vecgo/internal/segment"
)

const (
	chunkBits = 12             // 4096 items per chunk
	chunkSize = 1 << chunkBits // 4096
	chunkMask = chunkSize - 1
)

// tombstoneChunk holds a page of deletion LSNs.
type tombstoneChunk struct {
	lsns [chunkSize]uint64
}

// tombstoneState holds the directory of chunks for VersionedTombstones (COW).
type tombstoneState struct {
	chunks []*tombstoneChunk
	minLSN uint64
}

// VersionedTombstones tracks deletion LSNs using a paged/chunked Copy-On-Write structure.
// This architecture ensures O(1) reads and O(ChunkSize) writes (vs O(TotalRows)),
// preventing memory bandwidth saturation during deletes on large segments.
//
// 0 means alive (or deleted at future indefinite).
// value > 0 means deleted at LSN `value`.
type VersionedTombstones struct {
	state atomic.Pointer[tombstoneState]
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
// Uses a CAS loop to ensure thread safety while allowing concurrent lock-free reads.
func (vt *VersionedTombstones) MarkDeleted(rowID uint32, lsn uint64) {
	chunkIdx := int(rowID) >> chunkBits
	offset := int(rowID) & chunkMask

	for {
		curr := vt.state.Load()

		// 1. Check if already deleted (Optimization)
		currentLSN := uint64(0)
		if chunkIdx < len(curr.chunks) {
			chunk := curr.chunks[chunkIdx]
			if chunk != nil {
				currentLSN = chunk.lsns[offset]
			}
		}

		if currentLSN != 0 && currentLSN <= lsn {
			// Already deleted at an earlier (or same) time
			return
		}

		// 2. Prepare new state
		newNumChunks := len(curr.chunks)
		if chunkIdx >= newNumChunks {
			// Grow directory
			newNumChunks = chunkIdx + 1
			// Maybe double growth strategy for directory?
			if newNumChunks < 2*len(curr.chunks) {
				newNumChunks = 2 * len(curr.chunks)
			}
		}

		newChunks := make([]*tombstoneChunk, newNumChunks)
		copy(newChunks, curr.chunks) // Copy directory (pointers only)

		// 3. Prepare new chunk (COW)
		var newChunk *tombstoneChunk
		if chunkIdx < len(curr.chunks) && curr.chunks[chunkIdx] != nil {
			// Copy existing chunk
			existing := curr.chunks[chunkIdx]
			copied := *existing // Shallow copy of struct (array copies by value?)
			// Wait, fixed array in struct copies by value in Go? Yes.
			newChunk = &copied
		} else {
			// Create new chunk
			newChunk = &tombstoneChunk{}
		}

		// Update LSN in new chunk
		newChunk.lsns[offset] = lsn

		// Update directory
		newChunks[chunkIdx] = newChunk

		minLSN := curr.minLSN
		if lsn < minLSN {
			minLSN = lsn
		}
		newState := &tombstoneState{
			chunks: newChunks,
			minLSN: minLSN,
		}

		if vt.state.CompareAndSwap(curr, newState) {
			return
		}
		// CAS failed, look again
	}
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
func (vt *VersionedTombstones) LoadFromBitmap(bm *imetadata.LocalBitmap) {
	if bm == nil {
		return
	}

	// Collect all IDs
	var maxID uint32
	var ids []uint32
	for id := range bm.Iterator() {
		ids = append(ids, uint32(id))
		if uint32(id) > maxID {
			maxID = uint32(id)
		}
	}

	if len(ids) == 0 {
		return
	}

	for {
		curr := vt.state.Load()

		maxChunkIdx := int(maxID) >> chunkBits
		newNumChunks := len(curr.chunks)
		if maxChunkIdx >= newNumChunks {
			newNumChunks = maxChunkIdx + 1
		}

		newChunks := make([]*tombstoneChunk, newNumChunks)
		copy(newChunks, curr.chunks)

		// Map chunkIdx -> *tombstoneChunk (mutable copy)
		modifiedChunks := make(map[int]*tombstoneChunk)

		for _, id := range ids {
			cIdx := int(id) >> chunkBits
			off := int(id) & chunkMask

			chunk, exists := modifiedChunks[cIdx]
			if !exists {
				// Need to COW this chunk from newChunks (which is copy of curr)
				// Check if newChunks has it populated (from copy)
				if cIdx < len(curr.chunks) && curr.chunks[cIdx] != nil {
					existing := curr.chunks[cIdx]
					copied := *existing
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
		newState := &tombstoneState{
			chunks: newChunks,
			minLSN: minLSN,
		}

		if vt.state.CompareAndSwap(curr, newState) {
			return
		}
	}
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
	return !(lsn != 0 && lsn <= tf.snapshotLSN)
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
		out[i] = !(delLSN != 0 && delLSN <= tf.snapshotLSN)
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
