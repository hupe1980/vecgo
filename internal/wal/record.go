package wal

import (
	"encoding/binary"
	"errors"
	"hash"
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
	PK       model.PK
	Vector   []float32
	Metadata []byte
	Payload  []byte
}

func pkSize(pk model.PK) uint32 {
	if pk.Kind() == model.PKKindUint64 {
		return 1 + 8 // Kind + U64
	}
	s, _ := pk.StringValue()
	return 1 + 4 + uint32(len(s)) // Kind + Len + String
}

func writePK(pk model.PK, w io.Writer, crc hash.Hash32, scratch []byte) error {
	// Kind
	scratch[0] = byte(pk.Kind())
	if w != nil {
		if _, err := w.Write(scratch[:1]); err != nil {
			return err
		}
	}
	if crc != nil {
		crc.Write(scratch[:1])
	}

	if pk.Kind() == model.PKKindUint64 {
		u64, _ := pk.Uint64()
		binary.LittleEndian.PutUint64(scratch, u64)
		if w != nil {
			if _, err := w.Write(scratch[:8]); err != nil {
				return err
			}
		}
		if crc != nil {
			crc.Write(scratch[:8])
		}
	} else {
		s, _ := pk.StringValue()
		strBytes := []byte(s)
		binary.LittleEndian.PutUint32(scratch, uint32(len(strBytes)))
		if w != nil {
			if _, err := w.Write(scratch[:4]); err != nil {
				return err
			}
			if _, err := w.Write(strBytes); err != nil {
				return err
			}
		}
		if crc != nil {
			crc.Write(scratch[:4])
			crc.Write(strBytes)
		}
	}
	return nil
}

// Encode writes the record to w.
// Format:
// [CRC32: 4 bytes] [Type: 1 byte] [LSN: 8 bytes] [Length: 4 bytes] [Payload: Length bytes]
// Payload for Upsert: [PK: Var] [Dim: 4 bytes] [Vector: Dim*4 bytes] [MetaLen: 4 bytes] [Metadata: MetaLen bytes] [PayloadLen: 4 bytes] [Payload: PayloadLen bytes]
// Payload for Delete: [PK: Var]
func (r *Record) Size() int {
	var payloadLen uint32
	pkLen := pkSize(r.PK)
	if r.Type == RecordTypeUpsert {
		payloadLen = pkLen + 4 + uint32(len(r.Vector))*4 + 4 + uint32(len(r.Metadata)) + 4 + uint32(len(r.Payload))
	} else {
		payloadLen = pkLen
	}
	return int(4 + 1 + 8 + 4 + payloadLen)
}

func (r *Record) Encode(w io.Writer) error {
	// Calculate payload length
	var payloadLen uint32
	pkLen := pkSize(r.PK)
	if r.Type == RecordTypeUpsert {
		payloadLen = pkLen + 4 + uint32(len(r.Vector))*4 + 4 + uint32(len(r.Metadata)) + 4 + uint32(len(r.Payload))
	} else {
		payloadLen = pkLen
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

	updateCRCUint32 := func(v uint32) {
		binary.LittleEndian.PutUint32(scratch[:4], v)
		crc.Write(scratch[:4])
	}

	if r.Type == RecordTypeUpsert {
		writePK(r.PK, nil, crc, scratch)
		updateCRCUint32(uint32(len(r.Vector)))
		for _, v := range r.Vector {
			updateCRCUint32(math.Float32bits(v))
		}
		updateCRCUint32(uint32(len(r.Metadata)))
		crc.Write(r.Metadata)
		updateCRCUint32(uint32(len(r.Payload)))
		crc.Write(r.Payload)
	} else {
		writePK(r.PK, nil, crc, scratch)
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
	writeUint32 := func(v uint32) error {
		binary.LittleEndian.PutUint32(scratch[:4], v)
		_, err := w.Write(scratch[:4])
		return err
	}

	if r.Type == RecordTypeUpsert {
		if err := writePK(r.PK, w, nil, scratch); err != nil {
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
		if err := writePK(r.PK, w, nil, scratch); err != nil {
			return err
		}
	}

	return nil
}

// Decode reads a record from r.
func Decode(r io.Reader) (*Record, int64, error) {
	// Read CRC
	var checksum uint32
	if err := binary.Read(r, binary.LittleEndian, &checksum); err != nil {
		return nil, 0, err
	}

	// Read Header (Type + LSN + Length)
	header := make([]byte, 13)
	if _, err := io.ReadFull(r, header); err != nil {
		// We read 4 bytes (CRC) already if ReadFull fails
		return nil, 4, err
	}

	recType := RecordType(header[0])
	lsn := binary.LittleEndian.Uint64(header[1:])
	length := binary.LittleEndian.Uint32(header[9:])

	// Sanity check length (e.g. 100MB limit)
	if length > 100*1024*1024 {
		return nil, 4 + 13, ErrRecordTooLarge
	}

	// Read Payload
	payload := make([]byte, length)
	if _, err := io.ReadFull(r, payload); err != nil {
		return nil, 4 + 13, err
	}

	// Verify CRC
	crc := crc32.NewIEEE()
	crc.Write(header)
	crc.Write(payload)
	if crc.Sum32() != checksum {
		return nil, 4 + 13 + int64(length), ErrInvalidCRC
	}

	// Parse Payload
	rec := &Record{Type: recType, LSN: lsn}
	switch recType {
	case RecordTypeUpsert:
		if err := parseUpsert(payload, rec); err != nil {
			return nil, 4 + 13 + int64(length), err
		}
	case RecordTypeDelete:
		if err := parseDelete(payload, rec); err != nil {
			return nil, 4 + 13 + int64(length), err
		}
	default:
		return nil, 4 + 13 + int64(length), ErrInvalidType
	}

	return rec, 4 + 13 + int64(length), nil
}

func parsePK(payload []byte, offset int) (model.PK, int, error) {
	if len(payload) < offset+1 {
		return model.PK{}, 0, ErrShortRead
	}
	kind := model.PKKind(payload[offset])
	offset++

	if kind == model.PKKindUint64 {
		if len(payload) < offset+8 {
			return model.PK{}, 0, ErrShortRead
		}
		val := binary.LittleEndian.Uint64(payload[offset:])
		return model.PKUint64(val), offset + 8, nil
	} else {
		if len(payload) < offset+4 {
			return model.PK{}, 0, ErrShortRead
		}
		strLen := binary.LittleEndian.Uint32(payload[offset:])
		offset += 4
		if len(payload) < offset+int(strLen) {
			return model.PK{}, 0, ErrShortRead
		}
		str := string(payload[offset : offset+int(strLen)])
		return model.PKString(str), offset + int(strLen), nil
	}
}

func parseUpsert(payload []byte, r *Record) error {
	pk, offset, err := parsePK(payload, 0)
	if err != nil {
		return err
	}
	r.PK = pk

	if len(payload) < offset+4 {
		return ErrShortRead
	}
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
	pk, _, err := parsePK(payload, 0)
	if err != nil {
		return err
	}
	r.PK = pk
	return nil
}

// Helper to avoid math import if possible, or just use math


func fromBits(b uint32) float32 {
	return math.Float32frombits(b)
}
