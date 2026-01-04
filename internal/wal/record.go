package wal

import (
	"encoding/binary"
	"errors"
	"hash/crc32"
	"io"
	"math"

	"github.com/hupe1980/vecgo/model"
)

// RecordType identifies the type of WAL record.
type RecordType uint8

const (
	RecordTypeUpsert RecordType = 1
	RecordTypeDelete RecordType = 2
)

var (
	ErrInvalidCRC     = errors.New("invalid WAL record checksum")
	ErrInvalidType    = errors.New("invalid WAL record type")
	ErrShortRead      = errors.New("short read in WAL record")
	ErrRecordTooLarge = errors.New("WAL record too large")
)

// Record represents a single operation in the WAL.
type Record struct {
	LSN      uint64
	Type     RecordType
	PK       model.PrimaryKey
	Vector   []float32
	Metadata []byte
	Payload  []byte
}

// Encode writes the record to w.
// Format:
// [CRC32: 4 bytes] [Type: 1 byte] [LSN: 8 bytes] [Length: 4 bytes] [Payload: Length bytes]
// Payload for Upsert: [PK: 8 bytes] [Dim: 4 bytes] [Vector: Dim*4 bytes] [MetaLen: 4 bytes] [Metadata: MetaLen bytes] [PayloadLen: 4 bytes] [Payload: PayloadLen bytes]
// Payload for Delete: [PK: 8 bytes]
func (r *Record) Encode(w io.Writer) error {
	// Calculate payload length
	var payloadLen uint32
	if r.Type == RecordTypeUpsert {
		payloadLen = 8 + 4 + uint32(len(r.Vector))*4 + 4 + uint32(len(r.Metadata)) + 4 + uint32(len(r.Payload))
	} else {
		payloadLen = 8
	}

	// Header: Type (1) + LSN (8) + Length (4) = 13 bytes
	header := make([]byte, 13)
	header[0] = byte(r.Type)
	binary.LittleEndian.PutUint64(header[1:], r.LSN)
	binary.LittleEndian.PutUint32(header[9:], payloadLen)

	// Calculate CRC
	crc := crc32.NewIEEE()
	crc.Write(header)

	// Scratch buffer for primitives
	scratch := make([]byte, 8)

	updateCRCUint64 := func(v uint64) {
		binary.LittleEndian.PutUint64(scratch, v)
		crc.Write(scratch)
	}
	updateCRCUint32 := func(v uint32) {
		binary.LittleEndian.PutUint32(scratch[:4], v)
		crc.Write(scratch[:4])
	}

	if r.Type == RecordTypeUpsert {
		updateCRCUint64(uint64(r.PK))
		updateCRCUint32(uint32(len(r.Vector)))
		for _, v := range r.Vector {
			updateCRCUint32(math.Float32bits(v))
		}
		updateCRCUint32(uint32(len(r.Metadata)))
		crc.Write(r.Metadata)
		updateCRCUint32(uint32(len(r.Payload)))
		crc.Write(r.Payload)
	} else {
		updateCRCUint64(uint64(r.PK))
	}

	checksum := crc.Sum32()

	// Write CRC
	if err := binary.Write(w, binary.LittleEndian, checksum); err != nil {
		return err
	}
	// Write Header
	if _, err := w.Write(header); err != nil {
		return err
	}

	// Write Payload
	writeUint64 := func(v uint64) error {
		binary.LittleEndian.PutUint64(scratch, v)
		_, err := w.Write(scratch)
		return err
	}
	writeUint32 := func(v uint32) error {
		binary.LittleEndian.PutUint32(scratch[:4], v)
		_, err := w.Write(scratch[:4])
		return err
	}

	if r.Type == RecordTypeUpsert {
		if err := writeUint64(uint64(r.PK)); err != nil {
			return err
		}
		if err := writeUint32(uint32(len(r.Vector))); err != nil {
			return err
		}
		for _, v := range r.Vector {
			if err := writeUint32(math.Float32bits(v)); err != nil {
				return err
			}
		}
		if err := writeUint32(uint32(len(r.Metadata))); err != nil {
			return err
		}
		if _, err := w.Write(r.Metadata); err != nil {
			return err
		}
		if err := writeUint32(uint32(len(r.Payload))); err != nil {
			return err
		}
		if _, err := w.Write(r.Payload); err != nil {
			return err
		}
	} else {
		if err := writeUint64(uint64(r.PK)); err != nil {
			return err
		}
	}

	return nil
}

// Decode reads a record from r.
func Decode(r io.Reader) (*Record, error) {
	// Read CRC
	var checksum uint32
	if err := binary.Read(r, binary.LittleEndian, &checksum); err != nil {
		return nil, err
	}

	// Read Header (Type + LSN + Length)
	header := make([]byte, 13)
	if _, err := io.ReadFull(r, header); err != nil {
		return nil, err
	}

	recType := RecordType(header[0])
	lsn := binary.LittleEndian.Uint64(header[1:])
	length := binary.LittleEndian.Uint32(header[9:])

	// Sanity check length (e.g. 100MB limit)
	if length > 100*1024*1024 {
		return nil, ErrRecordTooLarge
	}

	// Read Payload
	payload := make([]byte, length)
	if _, err := io.ReadFull(r, payload); err != nil {
		return nil, err
	}

	// Verify CRC
	crc := crc32.NewIEEE()
	crc.Write(header)
	crc.Write(payload)
	if crc.Sum32() != checksum {
		return nil, ErrInvalidCRC
	}

	// Parse Payload
	rec := &Record{Type: recType, LSN: lsn}
	switch recType {
	case RecordTypeUpsert:
		if err := parseUpsert(payload, rec); err != nil {
			return nil, err
		}
	case RecordTypeDelete:
		if err := parseDelete(payload, rec); err != nil {
			return nil, err
		}
	default:
		return nil, ErrInvalidType
	}

	return rec, nil
}

func parseUpsert(payload []byte, r *Record) error {
	if len(payload) < 12 { // PK(8) + Dim(4)
		return ErrShortRead
	}
	offset := 0
	r.PK = model.PrimaryKey(binary.LittleEndian.Uint64(payload[offset:]))
	offset += 8

	dim := binary.LittleEndian.Uint32(payload[offset:])
	offset += 4

	if len(payload) < offset+int(dim)*4+4 {
		return ErrShortRead
	}

	r.Vector = make([]float32, dim)
	for i := 0; i < int(dim); i++ {
		bits := binary.LittleEndian.Uint32(payload[offset:])
		r.Vector[i] = fromBits(bits)
		offset += 4
	}

	metaLen := binary.LittleEndian.Uint32(payload[offset:])
	offset += 4

	if len(payload) < offset+int(metaLen) {
		return ErrShortRead
	}
	r.Metadata = make([]byte, metaLen)
	copy(r.Metadata, payload[offset:offset+int(metaLen)])
	offset += int(metaLen)

	if len(payload) < offset+4 {
		return ErrShortRead
	}
	payloadLen := binary.LittleEndian.Uint32(payload[offset:])
	offset += 4

	if len(payload) < offset+int(payloadLen) {
		return ErrShortRead
	}
	r.Payload = make([]byte, payloadLen)
	copy(r.Payload, payload[offset:offset+int(payloadLen)])

	return nil
}

func parseDelete(payload []byte, r *Record) error {
	if len(payload) < 8 {
		return ErrShortRead
	}
	r.PK = model.PrimaryKey(binary.LittleEndian.Uint64(payload))
	return nil
}

// Helper to avoid math import if possible, or just use math

func toBits(f float32) uint32 {
	return math.Float32bits(f)
}

func fromBits(b uint32) float32 {
	return math.Float32frombits(b)
}
