package persistence

import "errors"

const (
	// MagicNumber identifies VecDB binary files (ASCII: "VEC0")
	MagicNumber = 0x56454330
	// Version is the current file format version (v1.0.1)
	Version = 0x00010001

	// Index types
	IndexTypeFlat = 1
	IndexTypeHNSW = 2
)

var (
	ErrInvalidMagic   = errors.New("invalid magic number")
	ErrInvalidVersion = errors.New("unsupported version")
	ErrInvalidIndex   = errors.New("invalid index type")
)

// FileHeader is the 64-byte header at the start of every index file.
// Layout optimized for mmap compatibility and cache alignment.
type FileHeader struct {
	Magic       uint32 // 0x56454330 ("VEC0")
	Version     uint32 // File format version
	IndexType   uint8  // 1=Flat, 2=HNSW
	Padding1    [3]byte
	VectorCount uint64 // Total number of vectors
	Dimension   uint32 // Vector dimensionality
	DataOffset  uint64 // Offset to vector data section
	MetaOffset  uint64 // Offset to metadata section
	Checksum    uint32 // CRC32 of entire file
	Padding2    [4]byte
	Reserved    [16]byte // Future use
}

// NodeHeader is the 16-byte header for each HNSW node.
type NodeHeader struct {
	ID      uint64  // Node ID
	Layer   uint16  // Max layer
	VecLen  uint16  // Vector length (dimensions)
	Padding [4]byte // Align to 16 bytes
}
