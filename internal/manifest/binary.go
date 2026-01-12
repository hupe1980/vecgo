package manifest

import (
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"io"
	"time"

	"github.com/hupe1980/vecgo/model"
)

const (
	binaryMagic   = 0x56454347 // "VECG"
	binaryVersion = 3
)

// WriteBinary writes the manifest in binary format.
// Format:
// Magic (4 bytes)
// Version (4 bytes)
// Checksum (4 bytes) - CRC32 of payload
// PayloadLength (4 bytes)
// Payload:
//
//	ID (8 bytes)
//	CreatedAt (8 bytes) - UnixNano
//	Dim (8 bytes)
//	Metric (string)
//	NextSegmentID (8 bytes)
//	MaxLSN (8 bytes)
//	NumSegments (4 bytes)
//	Segments...
//	  ID (8 bytes)
//	  Level (4 bytes)
//	  RowCount (4 bytes)
//	  PathLen (2 bytes)
//	  Path (bytes)
func (m *Manifest) WriteBinary(w io.Writer) error {
	// Buffer for payload
	// Estimate size: 8*5 + 4 + N*(8+4+4+2+64) ~ 44 + N*82
	payloadSize := 44 + len(m.Segments)*100
	buf := make([]byte, 0, payloadSize)
	pb := newPayloadBuffer(buf)

	pb.writeUint64(m.ID)
	pb.writeUint64(uint64(m.CreatedAt.UnixNano()))
	pb.writeUint64(uint64(m.Dim))
	pb.writeString(m.Metric)
	pb.writeUint64(uint64(m.NextSegmentID))
	pb.writeUint64(m.MaxLSN)
	pb.writeUint32(uint32(len(m.Segments)))

	for _, s := range m.Segments {
		pb.writeUint64(uint64(s.ID))
		pb.writeUint32(uint32(s.Level))
		pb.writeUint32(s.RowCount)
		pb.writeString(s.Path)
	}

	payload := pb.buf
	checksum := crc32.ChecksumIEEE(payload)

	// Write header
	header := make([]byte, 16)
	binary.LittleEndian.PutUint32(header[0:4], binaryMagic)
	binary.LittleEndian.PutUint32(header[4:8], binaryVersion)
	binary.LittleEndian.PutUint32(header[8:12], checksum)
	binary.LittleEndian.PutUint32(header[12:16], uint32(len(payload)))

	if _, err := w.Write(header); err != nil {
		return err
	}
	if _, err := w.Write(payload); err != nil {
		return err
	}
	return nil
}

// ReadBinary reads the manifest from binary format.
func ReadBinary(r io.Reader) (*Manifest, error) {
	header := make([]byte, 16)
	if _, err := io.ReadFull(r, header); err != nil {
		return nil, err
	}

	magic := binary.LittleEndian.Uint32(header[0:4])
	if magic != binaryMagic {
		return nil, fmt.Errorf("invalid magic: %x", magic)
	}
	version := binary.LittleEndian.Uint32(header[4:8])
	if version != 2 && version != 3 {
		return nil, fmt.Errorf("unsupported version: %d", version)
	}
	checksum := binary.LittleEndian.Uint32(header[8:12])
	length := binary.LittleEndian.Uint32(header[12:16])

	payload := make([]byte, length)
	if _, err := io.ReadFull(r, payload); err != nil {
		return nil, err
	}

	if crc32.ChecksumIEEE(payload) != checksum {
		return nil, fmt.Errorf("checksum mismatch")
	}

	pb := newPayloadBuffer(payload)
	m := &Manifest{Version: int(version)}

	m.ID = pb.readUint64()
	if version >= 3 {
		m.CreatedAt = time.Unix(0, int64(pb.readUint64()))
	}
	m.Dim = int(pb.readUint64())
	m.Metric = pb.readString()
	m.NextSegmentID = model.SegmentID(pb.readUint64())
	m.MaxLSN = pb.readUint64()

	numSegments := pb.readUint32()
	m.Segments = make([]SegmentInfo, numSegments)
	for i := 0; i < int(numSegments); i++ {
		m.Segments[i].ID = model.SegmentID(pb.readUint64())
		m.Segments[i].Level = int(pb.readUint32())
		m.Segments[i].RowCount = pb.readUint32()
		m.Segments[i].Path = pb.readString()
	}

	if pb.err != nil {
		return nil, pb.err
	}

	return m, nil
}

type payloadBuffer struct {
	buf []byte
	pos int
	err error
}

func newPayloadBuffer(b []byte) *payloadBuffer {
	return &payloadBuffer{buf: b}
}

func (p *payloadBuffer) writeUint64(v uint64) {
	p.buf = binary.LittleEndian.AppendUint64(p.buf, v)
}

func (p *payloadBuffer) writeUint32(v uint32) {
	p.buf = binary.LittleEndian.AppendUint32(p.buf, v)
}

func (p *payloadBuffer) writeString(s string) {
	if len(s) > 65535 {
		p.err = fmt.Errorf("string too long")
		return
	}
	p.buf = binary.LittleEndian.AppendUint16(p.buf, uint16(len(s)))
	p.buf = append(p.buf, s...)
}

func (p *payloadBuffer) readUint64() uint64 {
	if p.err != nil {
		return 0
	}
	if p.pos+8 > len(p.buf) {
		p.err = io.ErrUnexpectedEOF
		return 0
	}
	v := binary.LittleEndian.Uint64(p.buf[p.pos:])
	p.pos += 8
	return v
}

func (p *payloadBuffer) readUint32() uint32 {
	if p.err != nil {
		return 0
	}
	if p.pos+4 > len(p.buf) {
		p.err = io.ErrUnexpectedEOF
		return 0
	}
	v := binary.LittleEndian.Uint32(p.buf[p.pos:])
	p.pos += 4
	return v
}

func (p *payloadBuffer) readString() string {
	if p.err != nil {
		return ""
	}
	if p.pos+2 > len(p.buf) {
		p.err = io.ErrUnexpectedEOF
		return ""
	}
	l := binary.LittleEndian.Uint16(p.buf[p.pos:])
	p.pos += 2

	if p.pos+int(l) > len(p.buf) {
		p.err = io.ErrUnexpectedEOF
		return ""
	}
	s := string(p.buf[p.pos : p.pos+int(l)])
	p.pos += int(l)
	return s
}
