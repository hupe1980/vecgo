package pk

import (
	"github.com/hupe1980/vecgo/model"
)

const (
	// chunkSize is the number of locations per chunk.
	// 4096 * 12 bytes = 48KB per chunk.
	chunkSize = 4096
)

// PersistentIndex is an immutable, persistent index mapping ID -> Location.
// It uses a chunked array (trie of depth 1) to allow cheap COWs.
type PersistentIndex struct {
	chunks []*chunk
	count  uint64 // Max ID tracked + 1
}

type chunk struct {
	locs [chunkSize]model.Location
}

// NewPersistentIndex creates a new empty persistent index.
func NewPersistentIndex() *PersistentIndex {
	return &PersistentIndex{
		chunks: make([]*chunk, 0, 16),
		count:  0,
	}
}

// Get returns the location for the given ID.
func (idx *PersistentIndex) Get(id model.ID) (model.Location, bool) {
	if uint64(id) >= idx.count {
		return model.Location{}, false
	}
	chunkIdx := uint64(id) / chunkSize
	offset := uint64(id) % chunkSize

	if chunkIdx >= uint64(len(idx.chunks)) {
		return model.Location{}, false
	}

	c := idx.chunks[chunkIdx]
	if c == nil {
		return model.Location{}, false
	}

	loc := c.locs[offset]
	return loc, true
}

// Set adds or updates a location for an ID, returning a new PersistentIndex (COW).
// It auto-expands the index if ID is beyond current bounds.
func (idx *PersistentIndex) Insert(id model.ID, loc model.Location) *PersistentIndex {
	chunkIdx := uint64(id) / chunkSize
	offset := uint64(id) % chunkSize

	newIdx := &PersistentIndex{
		chunks: make([]*chunk, len(idx.chunks)),
		count:  idx.count,
	}
	copy(newIdx.chunks, idx.chunks)

	// Expand if necessary
	if chunkIdx >= uint64(len(newIdx.chunks)) {
		// Grow capacity
		needed := chunkIdx + 1
		grown := make([]*chunk, needed)
		copy(grown, newIdx.chunks)
		newIdx.chunks = grown
	}

	// Update count if we are growing the ID space
	if uint64(id) >= newIdx.count {
		newIdx.count = uint64(id) + 1
	}

	// COW the chunk
	oldChunk := newIdx.chunks[chunkIdx]
	var newChunk *chunk
	if oldChunk != nil {
		// Shallow clone the struct (copy array)
		copied := *oldChunk
		newChunk = &copied
	} else {
		newChunk = &chunk{}
	}

	newChunk.locs[offset] = loc
	newIdx.chunks[chunkIdx] = newChunk

	return newIdx
}

// Len returns the number of IDs tracked.
func (idx *PersistentIndex) Len() int {
	return int(idx.count)
}

// Scan iterates over all entries.
func (idx *PersistentIndex) Scan() func(func(model.ID, model.Location) bool) {
	return func(yield func(model.ID, model.Location) bool) {
		for i := uint64(0); i < idx.count; i++ {
			chunkIdx := i / chunkSize
			offset := i % chunkSize
			if chunkIdx >= uint64(len(idx.chunks)) || idx.chunks[chunkIdx] == nil {
				continue
			}
			loc := idx.chunks[chunkIdx].locs[offset]
			if !yield(model.ID(i), loc) {
				return
			}
		}
	}
}
