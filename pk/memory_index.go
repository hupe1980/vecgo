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
// Entry: [PKBlob] [SegmentID: 8 bytes] [RowID: 4 bytes]
// PKBlob: [Kind: 1 byte] [Data...]
func (idx *MemoryIndex) Save(w io.Writer) error {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	bw := bufio.NewWriter(w)

	// Write count
	if err := binary.Write(bw, binary.LittleEndian, uint64(len(idx.m))); err != nil {
		return err
	}

	// Write entries
	buf := make([]byte, 12) // 8 + 4 (SegID + RowID)
	for pk, loc := range idx.m {
		// Write PK
		if err := binary.Write(bw, binary.LittleEndian, byte(pk.Kind())); err != nil {
			return err
		}
		if pk.Kind() == model.PKKindUint64 {
			u64, _ := pk.Uint64()
			if err := binary.Write(bw, binary.LittleEndian, u64); err != nil {
				return err
			}
		} else {
			s, _ := pk.StringValue()
			strBytes := []byte(s)
			if err := binary.Write(bw, binary.LittleEndian, uint32(len(strBytes))); err != nil {
				return err
			}
			if _, err := bw.Write(strBytes); err != nil {
				return err
			}
		}

		// Write Location
		binary.LittleEndian.PutUint64(buf[0:], uint64(loc.SegmentID))
		binary.LittleEndian.PutUint32(buf[8:], uint32(loc.RowID))
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

	buf := make([]byte, 12) // SegID + RowID
	for i := uint64(0); i < count; i++ {
		// Read PK
		kindByte, err := br.ReadByte()
		if err != nil {
			return err
		}
		kind := model.PKKind(kindByte)
		var pk model.PK

		if kind == model.PKKindUint64 {
			var u64 uint64
			if err := binary.Read(br, binary.LittleEndian, &u64); err != nil {
				return err
			}
			pk = model.PKUint64(u64)
		} else {
			var strLen uint32
			if err := binary.Read(br, binary.LittleEndian, &strLen); err != nil {
				return err
			}
			strBytes := make([]byte, strLen)
			if _, err := io.ReadFull(br, strBytes); err != nil {
				return err
			}
			pk = model.PKString(string(strBytes))
		}

		// Read Location
		if _, err := io.ReadFull(br, buf); err != nil {
			return err
		}
		segID := model.SegmentID(binary.LittleEndian.Uint64(buf[0:]))
		rowID := model.RowID(binary.LittleEndian.Uint32(buf[8:]))

		idx.m[pk] = model.Location{SegmentID: segID, RowID: rowID}
	}

	return nil
}
