package columnar

import (
	"encoding/binary"
	"errors"
	"hash/crc32"
	"io"
)

const (
	// FormatMagic identifies columnar vector files (ASCII: "COL0")
	FormatMagic = 0x434F4C30

	// FormatVersion is the current columnar file format version
	FormatVersion uint32 = 1

	// HeaderSize is the size of the file header in bytes
	HeaderSize = 64

	// FlagHasVersions indicates that the file contains version data.
	FlagHasVersions uint32 = 1 << 0 // File contains version data
	// FlagCompressed indicates that the vector data is compressed.
	FlagCompressed uint32 = 1 << 1 // Vector data is compressed (future)
)

var (
	// ErrInvalidMagic is returned when a file has an invalid magic number.
	ErrInvalidMagic = errors.New("columnar: invalid magic number")

	// ErrInvalidVersion is returned when a file has an unsupported version.
	ErrInvalidVersion = errors.New("columnar: unsupported format version")

	// ErrCorrupted is returned when a file fails checksum validation.
	ErrCorrupted = errors.New("columnar: file corrupted (checksum mismatch)")

	// ErrDimensionMismatch is returned when vector dimensions don't match.
	ErrDimensionMismatch = errors.New("columnar: vector dimension mismatch")

	// ErrOutOfBounds is returned when accessing an invalid vector ID.
	ErrOutOfBounds = errors.New("columnar: vector ID out of bounds")
)

// FileHeader is the 64-byte header at the start of columnar files.
//
// All multi-byte fields are little-endian.
type FileHeader struct {
	Magic      uint32  // 0x434F4C30 ("COL0")
	Version    uint32  // Format version (currently 1)
	Flags      uint32  // Feature flags
	Dimension  uint32  // Vector dimensionality
	Count      uint64  // Total number of vectors (including deleted)
	LiveCount  uint64  // Number of non-deleted vectors
	DataOffset uint64  // Offset to vector data section
	BitmapOff  uint64  // Offset to deletion bitmap
	VersionOff uint64  // Offset to version data (0 if no versions)
	Checksum   uint32  // CRC32 checksum of header (excluding this field)
	Reserved   [4]byte // Padding to 64 bytes
}

// Validate checks that the header is valid.
func (h *FileHeader) Validate() error {
	if h.Magic != FormatMagic {
		return ErrInvalidMagic
	}
	if h.Version > FormatVersion {
		return ErrInvalidVersion
	}
	return nil
}

// HasVersions returns true if the file contains version data.
func (h *FileHeader) HasVersions() bool {
	return h.Flags&FlagHasVersions != 0
}

// VectorDataSize returns the size of the vector data section in bytes.
func (h *FileHeader) VectorDataSize() int64 {
	return int64(h.Count) * int64(h.Dimension) * 4 //nolint:gosec
}

// BitmapSize returns the size of the deletion bitmap in bytes.
func (h *FileHeader) BitmapSize() int64 {
	return (int64(h.Count) + 7) / 8 //nolint:gosec
}

// VersionDataSize returns the size of the version data section in bytes.
func (h *FileHeader) VersionDataSize() int64 {
	if !h.HasVersions() {
		return 0
	}
	return int64(h.Count) * 8 //nolint:gosec
}

// TotalSize returns the total file size in bytes.
func (h *FileHeader) TotalSize() int64 {
	size := int64(HeaderSize)
	size += h.VectorDataSize()
	size += h.BitmapSize()
	size += h.VersionDataSize()
	size += 4 // Final checksum
	return size
}

// WriteTo writes the header to w.
func (h *FileHeader) WriteTo(w io.Writer) (int64, error) {
	// Calculate checksum of header fields (excluding checksum field itself)
	buf := make([]byte, HeaderSize)
	binary.LittleEndian.PutUint32(buf[0:4], h.Magic)
	binary.LittleEndian.PutUint32(buf[4:8], h.Version)
	binary.LittleEndian.PutUint32(buf[8:12], h.Flags)
	binary.LittleEndian.PutUint32(buf[12:16], h.Dimension)
	binary.LittleEndian.PutUint64(buf[16:24], h.Count)
	binary.LittleEndian.PutUint64(buf[24:32], h.LiveCount)
	binary.LittleEndian.PutUint64(buf[32:40], h.DataOffset)
	binary.LittleEndian.PutUint64(buf[40:48], h.BitmapOff)
	binary.LittleEndian.PutUint64(buf[48:56], h.VersionOff)

	// Compute checksum over first 56 bytes (excludes checksum + reserved)
	h.Checksum = crc32.ChecksumIEEE(buf[:56])
	binary.LittleEndian.PutUint32(buf[56:60], h.Checksum)
	// Reserved bytes remain zero

	n, err := w.Write(buf)
	return int64(n), err
}

// ReadFrom reads the header from r.
func (h *FileHeader) ReadFrom(r io.Reader) (int64, error) {
	buf := make([]byte, HeaderSize)
	n, err := io.ReadFull(r, buf)
	if err != nil {
		return int64(n), err
	}

	h.Magic = binary.LittleEndian.Uint32(buf[0:4])
	h.Version = binary.LittleEndian.Uint32(buf[4:8])
	h.Flags = binary.LittleEndian.Uint32(buf[8:12])
	h.Dimension = binary.LittleEndian.Uint32(buf[12:16])
	h.Count = binary.LittleEndian.Uint64(buf[16:24])
	h.LiveCount = binary.LittleEndian.Uint64(buf[24:32])
	h.DataOffset = binary.LittleEndian.Uint64(buf[32:40])
	h.BitmapOff = binary.LittleEndian.Uint64(buf[40:48])
	h.VersionOff = binary.LittleEndian.Uint64(buf[48:56])
	h.Checksum = binary.LittleEndian.Uint32(buf[56:60])

	// Validate checksum
	expectedChecksum := crc32.ChecksumIEEE(buf[:56])
	if h.Checksum != expectedChecksum {
		return int64(n), ErrCorrupted
	}

	return int64(n), h.Validate()
}
