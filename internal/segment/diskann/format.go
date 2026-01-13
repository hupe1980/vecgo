package diskann

import (
	"encoding/binary"
	"errors"
)

const (
	MagicNumber = 0x4449534B // "DISK"
	Version     = 2          // Bumped to 2 for compression support
)

var (
	ErrInvalidMagic   = errors.New("invalid magic number")
	ErrInvalidVersion = errors.New("unsupported version")
)

// FileHeader describes the layout of a DiskANN segment file.
type FileHeader struct {
	Magic            uint32
	Version          uint32
	SegmentID        uint64
	RowCount         uint32
	Dim              uint32
	Metric           uint8
	MaxDegree        uint32  // R
	SearchListSize   uint32  // L
	Entrypoint       uint32  // ID of the entry point node
	QuantizationType uint8   // 0 = None, 1 = SQ8, 2 = PQ, 3 = BQ, 4 = RaBitQ, 5 = INT4
	PQSubvectors     uint16  // M
	PQCentroids      uint16  // K
	CompressionType  uint8   // 0 = None, 1 = LZ4
	_                [5]byte // Padding to align offsets to 8 bytes

	// Offsets
	VectorOffset        uint64   // Full precision vectors (or compressed block start)
	GraphOffset         uint64   // Adjacency list (fixed size per node: R * 4 bytes)
	PQCodesOffset       uint64   // PQ compressed vectors
	BQCodesOffset       uint64   // BQ or RaBitQ compressed vectors
	PQCodebookOffset    uint64   // PQ codebooks (if PQ enabled)
	PKOffset            uint64   // Primary keys (uint64)
	MetadataOffset      uint64   // Offset to start of metadata
	BlockStatsOffset    uint64   // Offset to start of block statistics
	MetadataIndexOffset uint64   // Offset to start of metadata inverted index
	Checksum            uint32   // CRC32C of the body
	_                   [36]byte // Reserved
}

const HeaderSize = 4 + 4 + 8 + 4 + 4 + 1 + 4 + 4 + 4 + 1 + 2 + 2 + 1 + 5 + 8 + 8 + 8 + 8 + 8 + 8 + 8 + 8 + 8 + 4 + 36

func (h *FileHeader) Encode() []byte {
	buf := make([]byte, HeaderSize)
	binary.LittleEndian.PutUint32(buf[0:], h.Magic)
	binary.LittleEndian.PutUint32(buf[4:], h.Version)
	binary.LittleEndian.PutUint64(buf[8:], h.SegmentID)
	binary.LittleEndian.PutUint32(buf[16:], h.RowCount)
	binary.LittleEndian.PutUint32(buf[20:], h.Dim)
	buf[24] = h.Metric
	binary.LittleEndian.PutUint32(buf[25:], h.MaxDegree)
	binary.LittleEndian.PutUint32(buf[29:], h.SearchListSize)
	binary.LittleEndian.PutUint32(buf[33:], h.Entrypoint)
	buf[37] = h.QuantizationType
	binary.LittleEndian.PutUint16(buf[38:], h.PQSubvectors)
	binary.LittleEndian.PutUint16(buf[40:], h.PQCentroids)
	buf[42] = h.CompressionType
	// Padding [43:48]
	binary.LittleEndian.PutUint64(buf[48:], h.VectorOffset)
	binary.LittleEndian.PutUint64(buf[56:], h.GraphOffset)
	binary.LittleEndian.PutUint64(buf[64:], h.PQCodesOffset)
	binary.LittleEndian.PutUint64(buf[72:], h.BQCodesOffset)
	binary.LittleEndian.PutUint64(buf[80:], h.PQCodebookOffset)
	binary.LittleEndian.PutUint64(buf[88:], h.PKOffset)
	binary.LittleEndian.PutUint64(buf[96:], h.MetadataOffset)
	binary.LittleEndian.PutUint64(buf[104:], h.BlockStatsOffset)
	binary.LittleEndian.PutUint64(buf[112:], h.MetadataIndexOffset)
	binary.LittleEndian.PutUint32(buf[120:], h.Checksum)
	return buf
}

func DecodeHeader(buf []byte) (*FileHeader, error) {
	if len(buf) < HeaderSize {
		return nil, errors.New("buffer too small for header")
	}
	h := &FileHeader{}
	h.Magic = binary.LittleEndian.Uint32(buf[0:])
	if h.Magic != MagicNumber {
		return nil, ErrInvalidMagic
	}
	h.Version = binary.LittleEndian.Uint32(buf[4:])
	if h.Version != Version && h.Version != 1 {
		return nil, ErrInvalidVersion
	}
	h.SegmentID = binary.LittleEndian.Uint64(buf[8:])
	h.RowCount = binary.LittleEndian.Uint32(buf[16:])
	h.Dim = binary.LittleEndian.Uint32(buf[20:])
	h.Metric = buf[24]
	h.MaxDegree = binary.LittleEndian.Uint32(buf[25:])
	h.SearchListSize = binary.LittleEndian.Uint32(buf[29:])
	h.Entrypoint = binary.LittleEndian.Uint32(buf[33:])
	h.QuantizationType = buf[37]
	h.PQSubvectors = binary.LittleEndian.Uint16(buf[38:])
	h.PQCentroids = binary.LittleEndian.Uint16(buf[40:])
	if h.Version >= 2 {
		h.CompressionType = buf[42]
	}

	h.VectorOffset = binary.LittleEndian.Uint64(buf[48:])
	h.GraphOffset = binary.LittleEndian.Uint64(buf[56:])
	h.PQCodesOffset = binary.LittleEndian.Uint64(buf[64:])
	h.BQCodesOffset = binary.LittleEndian.Uint64(buf[72:])
	h.PQCodebookOffset = binary.LittleEndian.Uint64(buf[80:])
	h.PKOffset = binary.LittleEndian.Uint64(buf[88:])
	h.MetadataOffset = binary.LittleEndian.Uint64(buf[96:])
	h.BlockStatsOffset = binary.LittleEndian.Uint64(buf[104:])
	h.MetadataIndexOffset = binary.LittleEndian.Uint64(buf[112:])
	h.Checksum = binary.LittleEndian.Uint32(buf[120:])

	return h, nil
}
