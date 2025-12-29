package diskann

import (
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
	"io"

	"github.com/hupe1980/vecgo/index"
)

// File format constants
const (
	// FormatMagic identifies DiskANN index files
	FormatMagic uint32 = 0x44414E4E // "DANN"

	// FormatVersion is the current format version
	FormatVersion uint32 = 1

	// HeaderSize is the size of the file header in bytes
	HeaderSize = 128

	// MetaFilename is the name of the metadata file
	MetaFilename = "index.meta"

	// GraphFilename is the name of the graph file
	GraphFilename = "index.graph"

	// PQCodesFilename is the name of the PQ codes file
	PQCodesFilename = "index.pqcodes"

	// BQCodesFilename is the name of the Binary Quantization codes file (optional).
	// It is only used as a coarse prefilter during search and is never used for graph construction.
	BQCodesFilename = "index.bqcodes"

	// VectorsFilename is the name of the vectors file
	VectorsFilename = "index.vectors"
)

// Flags for index configuration
const (
	FlagPQEnabled   uint32 = 1 << 0
	FlagMmapEnabled uint32 = 1 << 1
	FlagCompressed  uint32 = 1 << 2
	FlagBQEnabled   uint32 = 1 << 3
)

// FileHeader represents the DiskANN index metadata header.
type FileHeader struct {
	Magic        uint32 // Magic number (FormatMagic)
	Version      uint32 // Format version
	Flags        uint32 // Configuration flags
	Dimension    uint32 // Vector dimensionality
	Count        uint64 // Total number of vectors
	DistanceType uint32 // Distance function type

	// Vamana graph parameters
	R     uint32 // Max edges per node
	L     uint32 // Build list size
	Alpha uint32 // Pruning factor * 1000 (e.g., 1200 = 1.2)

	// PQ parameters
	PQSubvectors uint32 // Number of PQ subvectors (M)
	PQCentroids  uint32 // Centroids per subspace (K, typically 256)

	// File offsets
	GraphOffset   uint64 // Offset to graph data
	PQCodesOffset uint64 // Offset to PQ codes
	VectorsOffset uint64 // Offset to full vectors

	// Checksum
	Checksum uint32 // CRC32 of header (excluding this field)

	// Reserved for future use
	Reserved [60]byte
}

// Validate checks if the header is valid.
func (h *FileHeader) Validate() error {
	if h.Magic != FormatMagic {
		return fmt.Errorf("diskann: invalid magic number: 0x%08X (expected 0x%08X)", h.Magic, FormatMagic)
	}
	if h.Version != FormatVersion {
		return fmt.Errorf("diskann: unsupported version: %d (expected %d)", h.Version, FormatVersion)
	}
	if h.Dimension == 0 {
		return errors.New("diskann: dimension cannot be zero")
	}
	if h.R == 0 {
		return errors.New("diskann: R (max edges) cannot be zero")
	}

	// Validate checksum
	computed := h.computeChecksum()
	if h.Checksum != computed {
		return fmt.Errorf("diskann: header checksum mismatch: 0x%08X (expected 0x%08X)", h.Checksum, computed)
	}

	return nil
}

// computeChecksum calculates CRC32 of header fields (excluding Checksum itself).
func (h *FileHeader) computeChecksum() uint32 {
	// Calculate size: 4+4+4+4+8+4+4+4+4+4+4+8+8+8 = 72 bytes (before checksum and reserved)
	buf := make([]byte, 72)
	offset := 0

	binary.LittleEndian.PutUint32(buf[offset:], h.Magic)
	offset += 4
	binary.LittleEndian.PutUint32(buf[offset:], h.Version)
	offset += 4
	binary.LittleEndian.PutUint32(buf[offset:], h.Flags)
	offset += 4
	binary.LittleEndian.PutUint32(buf[offset:], h.Dimension)
	offset += 4
	binary.LittleEndian.PutUint64(buf[offset:], h.Count)
	offset += 8
	binary.LittleEndian.PutUint32(buf[offset:], h.DistanceType)
	offset += 4
	binary.LittleEndian.PutUint32(buf[offset:], h.R)
	offset += 4
	binary.LittleEndian.PutUint32(buf[offset:], h.L)
	offset += 4
	binary.LittleEndian.PutUint32(buf[offset:], h.Alpha)
	offset += 4
	binary.LittleEndian.PutUint32(buf[offset:], h.PQSubvectors)
	offset += 4
	binary.LittleEndian.PutUint32(buf[offset:], h.PQCentroids)
	offset += 4
	binary.LittleEndian.PutUint64(buf[offset:], h.GraphOffset)
	offset += 8
	binary.LittleEndian.PutUint64(buf[offset:], h.PQCodesOffset)
	offset += 8
	binary.LittleEndian.PutUint64(buf[offset:], h.VectorsOffset)

	return crc32.ChecksumIEEE(buf[:offset])
}

// SetChecksum computes and sets the header checksum.
func (h *FileHeader) SetChecksum() {
	h.Checksum = h.computeChecksum()
}

// WriteTo writes the header to w.
func (h *FileHeader) WriteTo(w io.Writer) (int64, error) {
	h.SetChecksum()

	buf := make([]byte, HeaderSize)
	offset := 0

	binary.LittleEndian.PutUint32(buf[offset:], h.Magic)
	offset += 4
	binary.LittleEndian.PutUint32(buf[offset:], h.Version)
	offset += 4
	binary.LittleEndian.PutUint32(buf[offset:], h.Flags)
	offset += 4
	binary.LittleEndian.PutUint32(buf[offset:], h.Dimension)
	offset += 4
	binary.LittleEndian.PutUint64(buf[offset:], h.Count)
	offset += 8
	binary.LittleEndian.PutUint32(buf[offset:], h.DistanceType)
	offset += 4
	binary.LittleEndian.PutUint32(buf[offset:], h.R)
	offset += 4
	binary.LittleEndian.PutUint32(buf[offset:], h.L)
	offset += 4
	binary.LittleEndian.PutUint32(buf[offset:], h.Alpha)
	offset += 4
	binary.LittleEndian.PutUint32(buf[offset:], h.PQSubvectors)
	offset += 4
	binary.LittleEndian.PutUint32(buf[offset:], h.PQCentroids)
	offset += 4
	binary.LittleEndian.PutUint64(buf[offset:], h.GraphOffset)
	offset += 8
	binary.LittleEndian.PutUint64(buf[offset:], h.PQCodesOffset)
	offset += 8
	binary.LittleEndian.PutUint64(buf[offset:], h.VectorsOffset)
	offset += 8
	binary.LittleEndian.PutUint32(buf[offset:], h.Checksum)
	offset += 4
	copy(buf[offset:], h.Reserved[:])

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

	offset := 0
	h.Magic = binary.LittleEndian.Uint32(buf[offset:])
	offset += 4
	h.Version = binary.LittleEndian.Uint32(buf[offset:])
	offset += 4
	h.Flags = binary.LittleEndian.Uint32(buf[offset:])
	offset += 4
	h.Dimension = binary.LittleEndian.Uint32(buf[offset:])
	offset += 4
	h.Count = binary.LittleEndian.Uint64(buf[offset:])
	offset += 8
	h.DistanceType = binary.LittleEndian.Uint32(buf[offset:])
	offset += 4
	h.R = binary.LittleEndian.Uint32(buf[offset:])
	offset += 4
	h.L = binary.LittleEndian.Uint32(buf[offset:])
	offset += 4
	h.Alpha = binary.LittleEndian.Uint32(buf[offset:])
	offset += 4
	h.PQSubvectors = binary.LittleEndian.Uint32(buf[offset:])
	offset += 4
	h.PQCentroids = binary.LittleEndian.Uint32(buf[offset:])
	offset += 4
	h.GraphOffset = binary.LittleEndian.Uint64(buf[offset:])
	offset += 8
	h.PQCodesOffset = binary.LittleEndian.Uint64(buf[offset:])
	offset += 8
	h.VectorsOffset = binary.LittleEndian.Uint64(buf[offset:])
	offset += 8
	h.Checksum = binary.LittleEndian.Uint32(buf[offset:])
	offset += 4
	copy(h.Reserved[:], buf[offset:])

	return int64(n), nil
}

// DistType returns the distance type.
func (h *FileHeader) DistType() index.DistanceType {
	return index.DistanceType(h.DistanceType)
}

// AlphaFloat returns the alpha parameter as a float.
func (h *FileHeader) AlphaFloat() float32 {
	return float32(h.Alpha) / 1000.0
}
