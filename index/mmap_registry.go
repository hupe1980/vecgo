package index

import (
	"encoding/binary"
	"fmt"
	"sync"

	"github.com/hupe1980/vecgo/persistence"
)

// MmapBinaryLoader constructs an index instance by reading its binary
// representation from a byte slice. It returns the index and the number of
// bytes consumed from the start of data.
type MmapBinaryLoader func(data []byte) (idx Index, consumed int, err error)

var (
	mmapLoaderMu sync.RWMutex
	mmapLoaders  = map[uint8]MmapBinaryLoader{}
)

// RegisterMmapBinaryLoader registers a mmap loader for a specific on-disk index type.
func RegisterMmapBinaryLoader(indexType uint8, loader MmapBinaryLoader) {
	mmapLoaderMu.Lock()
	defer mmapLoaderMu.Unlock()
	mmapLoaders[indexType] = loader
}

// LoadBinaryIndexMmap reads an index from a mmapped byte slice.
//
// After this returns successfully, the returned consumed offset points to the first byte
// immediately after the index bytes within the containing snapshot/container.
func LoadBinaryIndexMmap(data []byte) (Index, int, error) {
	minHdr := binary.Size(persistence.FileHeader{})
	if minHdr <= 0 {
		return nil, 0, fmt.Errorf("invalid FileHeader size: %d", minHdr)
	}
	if len(data) < minHdr {
		return nil, 0, fmt.Errorf("snapshot too small for index header: %d", len(data))
	}
	magic := binary.LittleEndian.Uint32(data[0:4])
	if magic != persistence.MagicNumber {
		return nil, 0, fmt.Errorf("invalid magic number: expected 0x%08x, got 0x%08x", persistence.MagicNumber, magic)
	}
	indexType := data[8]

	mmapLoaderMu.RLock()
	loader, ok := mmapLoaders[indexType]
	mmapLoaderMu.RUnlock()
	if !ok {
		return nil, 0, fmt.Errorf("unknown index type: %d", indexType)
	}
	return loader(data)
}
