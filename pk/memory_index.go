package pk

import (
	"bufio"
	"encoding/binary"
	"io"
	"sync"

	"github.com/hupe1980/vecgo/model"
)

// MemoryIndex is an in-memory implementation of Index backed by a Go map.
// It supports persistence via Save/Load.
type MemoryIndex struct {
	mu sync.RWMutex
	m  map[model.PrimaryKey]model.Location
}

// NewMemoryIndex creates a new in-memory index.
func NewMemoryIndex() *MemoryIndex {
	return &MemoryIndex{
		m: make(map[model.PrimaryKey]model.Location),
	}
}

// Lookup returns the location for the given primary key.
func (idx *MemoryIndex) Lookup(pk model.PrimaryKey) (model.Location, bool) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	loc, ok := idx.m[pk]
	return loc, ok
}

// Upsert updates the location for the given primary key.
func (idx *MemoryIndex) Upsert(pk model.PrimaryKey, loc model.Location) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	idx.m[pk] = loc
	return nil
}

// Delete removes the primary key from the index.
func (idx *MemoryIndex) Delete(pk model.PrimaryKey) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	delete(idx.m, pk)
	return nil
}

// Save persists the index to w.
// Format: [Count: 8 bytes] [Entry...]
// Entry: [PK: 8 bytes] [SegmentID: 8 bytes] [RowID: 4 bytes]
func (idx *MemoryIndex) Save(w io.Writer) error {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	bw := bufio.NewWriter(w)

	// Write count
	if err := binary.Write(bw, binary.LittleEndian, uint64(len(idx.m))); err != nil {
		return err
	}

	// Write entries
	buf := make([]byte, 20) // 8 + 8 + 4
	for pk, loc := range idx.m {
		binary.LittleEndian.PutUint64(buf[0:], uint64(pk))
		binary.LittleEndian.PutUint64(buf[8:], uint64(loc.SegmentID))
		binary.LittleEndian.PutUint32(buf[16:], uint32(loc.RowID))
		if _, err := bw.Write(buf); err != nil {
			return err
		}
	}

	return bw.Flush()
}

// Load populates the index from r.
func (idx *MemoryIndex) Load(r io.Reader) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	br := bufio.NewReader(r)

	var count uint64
	if err := binary.Read(br, binary.LittleEndian, &count); err != nil {
		return err
	}

	// Reset map
	idx.m = make(map[model.PrimaryKey]model.Location, count)

	buf := make([]byte, 20)
	for i := uint64(0); i < count; i++ {
		if _, err := io.ReadFull(br, buf); err != nil {
			return err
		}
		pk := model.PrimaryKey(binary.LittleEndian.Uint64(buf[0:]))
		segID := model.SegmentID(binary.LittleEndian.Uint64(buf[8:]))
		rowID := model.RowID(binary.LittleEndian.Uint32(buf[16:]))

		idx.m[pk] = model.Location{SegmentID: segID, RowID: rowID}
	}

	return nil
}
