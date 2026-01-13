package flat

import (
	"encoding/binary"
	"errors"
	"math"

	"github.com/hupe1980/vecgo/internal/segment"
)

const (
	MagicNumber = 0x56454331 // "VEC1"
	Version     = 1
	BlockSize   = 1024
)

var (
	ErrInvalidMagic   = errors.New("invalid magic number")
	ErrInvalidVersion = errors.New("unsupported version")
)

const (
	QuantizationNone = 0
	QuantizationSQ8  = 1
	QuantizationPQ   = 2
)

// FileHeader describes the layout of a flat segment file.
// It is stored at the beginning of the file.
type FileHeader struct {
	Magic                 uint32
	Version               uint32
	SegmentID             uint64
	RowCount              uint32
	Dim                   uint32
	Metric                uint8
	_                     [3]byte // Padding
	NumPartitions         uint32
	QuantizationType      uint8   // 0 = None, 1 = SQ8, 2 = PQ
	_                     [7]byte // Padding to align offsets to 8 bytes
	CentroidOffset        uint64
	PartitionOffsetOffset uint64
	QuantizationOffset    uint64   // Offset to start of quantization metadata (mins/maxs)
	CodesOffset           uint64   // Offset to start of quantized codes
	VectorOffset          uint64   // Offset to start of vector data
	PKOffset              uint64   // Offset to start of PK data (uint64 array)
	MetadataOffset        uint64   // Offset to start of metadata (offsets + data)
	BlockStatsOffset      uint64   // Offset to start of block statistics
	Checksum              uint32   // CRC32C of the body (everything after header)
	_                     [44]byte // Reserved for future use
}

// BlockStats stores statistics for a block of rows.
type BlockStats struct {
	Fields map[string]segment.FieldStats `json:"fields"`
}

func (bs *BlockStats) MarshalBinary() ([]byte, error) {
	var buf []byte
	// Count
	buf = binary.AppendUvarint(buf, uint64(len(bs.Fields)))
	for k, v := range bs.Fields {
		// Key
		buf = binary.AppendUvarint(buf, uint64(len(k)))
		buf = append(buf, k...)
		// Min/Max
		buf = binary.LittleEndian.AppendUint64(buf, math.Float64bits(v.Min))
		buf = binary.LittleEndian.AppendUint64(buf, math.Float64bits(v.Max))
	}
	return buf, nil
}

func (bs *BlockStats) UnmarshalBinary(data []byte) error {
	if len(data) == 0 {
		return nil
	}
	count, n := binary.Uvarint(data)
	if n <= 0 {
		return errors.New("invalid block stats count")
	}
	data = data[n:]
	bs.Fields = make(map[string]segment.FieldStats, count)
	for i := 0; i < int(count); i++ {
		// Key
		kLen, n := binary.Uvarint(data)
		if n <= 0 {
			return errors.New("invalid key length")
		}
		data = data[n:]
		if uint64(len(data)) < kLen {
			return errors.New("key too short")
		}
		key := string(data[:kLen])
		data = data[kLen:]

		// Min/Max
		if len(data) < 16 {
			return errors.New("stats too short")
		}
		minVal := math.Float64frombits(binary.LittleEndian.Uint64(data[:8]))
		maxVal := math.Float64frombits(binary.LittleEndian.Uint64(data[8:16]))
		data = data[16:]

		bs.Fields[key] = segment.FieldStats{Min: minVal, Max: maxVal}
	}
	return nil
}

// Size of the header in bytes.
const HeaderSize = 4 + 4 + 8 + 4 + 4 + 1 + 3 + 4 + 1 + 7 + 8 + 8 + 8 + 8 + 8 + 8 + 8 + 8 + 48

func (h *FileHeader) Encode() []byte {
	buf := make([]byte, HeaderSize)
	binary.LittleEndian.PutUint32(buf[0:], h.Magic)
	binary.LittleEndian.PutUint32(buf[4:], h.Version)
	binary.LittleEndian.PutUint64(buf[8:], h.SegmentID)
	binary.LittleEndian.PutUint32(buf[16:], h.RowCount)
	binary.LittleEndian.PutUint32(buf[20:], h.Dim)
	buf[24] = h.Metric
	binary.LittleEndian.PutUint32(buf[28:], h.NumPartitions)
	buf[32] = h.QuantizationType
	// Padding [33:40]
	binary.LittleEndian.PutUint64(buf[40:], h.CentroidOffset)
	binary.LittleEndian.PutUint64(buf[48:], h.PartitionOffsetOffset)
	binary.LittleEndian.PutUint64(buf[56:], h.QuantizationOffset)
	binary.LittleEndian.PutUint64(buf[64:], h.CodesOffset)
	binary.LittleEndian.PutUint64(buf[72:], h.VectorOffset)
	binary.LittleEndian.PutUint64(buf[80:], h.PKOffset)
	binary.LittleEndian.PutUint64(buf[88:], h.MetadataOffset)
	binary.LittleEndian.PutUint64(buf[96:], h.BlockStatsOffset)
	binary.LittleEndian.PutUint32(buf[104:], h.Checksum)
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
	if h.Version != Version {
		return nil, ErrInvalidVersion
	}
	h.SegmentID = binary.LittleEndian.Uint64(buf[8:])
	h.RowCount = binary.LittleEndian.Uint32(buf[16:])
	h.Dim = binary.LittleEndian.Uint32(buf[20:])
	h.Metric = buf[24]
	h.NumPartitions = binary.LittleEndian.Uint32(buf[28:])
	h.QuantizationType = buf[32]
	// Padding [33:40]
	h.CentroidOffset = binary.LittleEndian.Uint64(buf[40:])
	h.PartitionOffsetOffset = binary.LittleEndian.Uint64(buf[48:])
	h.QuantizationOffset = binary.LittleEndian.Uint64(buf[56:])
	h.CodesOffset = binary.LittleEndian.Uint64(buf[64:])
	h.VectorOffset = binary.LittleEndian.Uint64(buf[72:])
	h.PKOffset = binary.LittleEndian.Uint64(buf[80:])
	h.MetadataOffset = binary.LittleEndian.Uint64(buf[88:])
	h.BlockStatsOffset = binary.LittleEndian.Uint64(buf[96:])
	h.Checksum = binary.LittleEndian.Uint32(buf[104:])
	return h, nil
}
