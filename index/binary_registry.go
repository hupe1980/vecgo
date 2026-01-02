package index

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"sync"
)

// BinaryLoader constructs an index instance by reading its binary representation
// from r. The reader will begin at the start of the binary index stream.
type BinaryLoader func(r io.Reader) (Index, error)

var (
	binaryLoaderMu sync.RWMutex
	binaryLoaders  = map[uint8]BinaryLoader{}
)

// RegisterBinaryLoader registers a loader for a specific on-disk index type.
//
// Index implementations should typically call this from an init() function.
func RegisterBinaryLoader(indexType uint8, loader BinaryLoader) {
	binaryLoaderMu.Lock()
	defer binaryLoaderMu.Unlock()
	binaryLoaders[indexType] = loader
}

// LoadBinaryIndex reads an index from r.
//
// It peeks the 64-byte file header to detect the index type, then dispatches
// to a registered loader. After this returns successfully, r will be positioned
// immediately after the index bytes, ready for subsequent reads (e.g. snapshot payload sections).
func LoadBinaryIndex(r io.Reader) (Index, error) {
	var header [64]byte
	if _, err := io.ReadFull(r, header[:]); err != nil {
		return nil, fmt.Errorf("failed to read header: %w", err)
	}

	magic := binary.LittleEndian.Uint32(header[0:4])
	if magic != 0x56454330 {
		return nil, fmt.Errorf("invalid magic number: expected 0x56454330, got 0x%08x", magic)
	}

	// Ensure header is large enough (already checked by ReadFull, but explicit check satisfies linter)
	if len(header) <= 8 {
		return nil, fmt.Errorf("header too short")
	}
	indexType := header[8]

	binaryLoaderMu.RLock()
	loader, ok := binaryLoaders[indexType]
	binaryLoaderMu.RUnlock()
	if !ok {
		return nil, fmt.Errorf("unknown index type: %d", indexType)
	}

	headerReader := bytes.NewReader(header[:])
	combinedReader := io.MultiReader(headerReader, r)

	idx, err := loader(combinedReader)
	if err != nil {
		return nil, err
	}
	return idx, nil
}
