package manifest

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"time"

	"github.com/hupe1980/vecgo/internal/hash"
	"github.com/hupe1980/vecgo/model"
)

// binaryMagic identifies vecgo manifest files.
const binaryMagic = 0x56454347 // "VECG"

// binaryVersion is the current manifest format version.
// Version 3: Added Bloom filters for categorical fields.
// Version 2: Enhanced SegmentStats with row counts, entropy, shape, and richer vector stats.
const binaryVersion = 3

// WriteBinary writes the manifest in binary format (version 1).
// Format:
// Header (16 bytes):
//
//	Magic (4 bytes) - 0x56454347 "VECG"
//	Version (4 bytes) - currently 1
//	Checksum (4 bytes) - CRC32 of payload
//	PayloadLength (4 bytes)
//
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
//	  Size (8 bytes)
//	  MinID (8 bytes)
//	  MaxID (8 bytes)
//	  PathLen (2 bytes)
//	  Path (bytes)
//	PKIndex.Path (string)
func (m *Manifest) WriteBinary(w io.Writer) error {
	// Buffer for payload
	// Estimate size: 8*5 + 4 + N*(8+4+4+8+8+8+2+64) + 2+64 ~ 50 + N*102 + 66
	payloadSize := 120 + len(m.Segments)*110
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
		pb.writeUint64(uint64(s.Size))
		pb.writeUint64(uint64(s.MinID))
		pb.writeUint64(uint64(s.MaxID))
		pb.writeString(s.Path)
		pb.writeSegmentStats(s.Stats) // Stats for segment pruning
	}

	pb.writeString(m.PKIndex.Path)

	// Check for any errors during payload construction (e.g., string too long)
	if pb.err != nil {
		return pb.err
	}

	payload := pb.buf
	checksum := hash.CRC32C(payload)

	// Write header: magic(4) + version(4) + checksum(4) + length(4) = 16 bytes
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
// Supports reading both v1 (legacy) and v2 (enhanced stats) formats.
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
	if version != 1 && version != binaryVersion {
		return nil, fmt.Errorf("unsupported version: %d (expected 1 or %d)", version, binaryVersion)
	}
	checksum := binary.LittleEndian.Uint32(header[8:12])
	length := binary.LittleEndian.Uint32(header[12:16])

	payload := make([]byte, length)
	if _, err := io.ReadFull(r, payload); err != nil {
		return nil, err
	}

	if hash.CRC32C(payload) != checksum {
		return nil, fmt.Errorf("checksum mismatch")
	}

	pb := newPayloadBuffer(payload)
	m := &Manifest{}
	m.Version = int(version)

	m.ID = pb.readUint64()
	m.CreatedAt = time.Unix(0, int64(pb.readUint64()))
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
		m.Segments[i].Size = int64(pb.readUint64())
		m.Segments[i].MinID = model.ID(pb.readUint64())
		m.Segments[i].MaxID = model.ID(pb.readUint64())
		m.Segments[i].Path = pb.readString()
		switch version {
		case 1:
			m.Segments[i].Stats = pb.readSegmentStatsV1()
		case 2:
			m.Segments[i].Stats = pb.readSegmentStatsV2()
		default:
			m.Segments[i].Stats = pb.readSegmentStats()
		}
	}

	m.PKIndex.Path = pb.readString()

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
	if p.err != nil {
		return
	}
	p.buf = binary.LittleEndian.AppendUint64(p.buf, v)
}

func (p *payloadBuffer) writeUint32(v uint32) {
	if p.err != nil {
		return
	}
	p.buf = binary.LittleEndian.AppendUint32(p.buf, v)
}

func (p *payloadBuffer) writeString(s string) {
	if p.err != nil {
		return
	}
	if len(s) > 65535 {
		p.err = fmt.Errorf("string too long: %d", len(s))
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

func (p *payloadBuffer) writeUint16(v uint16) {
	if p.err != nil {
		return
	}
	p.buf = binary.LittleEndian.AppendUint16(p.buf, v)
}

func (p *payloadBuffer) readUint16() uint16 {
	if p.err != nil {
		return 0
	}
	if p.pos+2 > len(p.buf) {
		p.err = io.ErrUnexpectedEOF
		return 0
	}
	v := binary.LittleEndian.Uint16(p.buf[p.pos:])
	p.pos += 2
	return v
}

func (p *payloadBuffer) writeFloat64(v float64) {
	if p.err != nil {
		return
	}
	p.buf = binary.LittleEndian.AppendUint64(p.buf, math.Float64bits(v))
}

func (p *payloadBuffer) readFloat64() float64 {
	if p.err != nil {
		return 0
	}
	if p.pos+8 > len(p.buf) {
		p.err = io.ErrUnexpectedEOF
		return 0
	}
	v := math.Float64frombits(binary.LittleEndian.Uint64(p.buf[p.pos:]))
	p.pos += 8
	return v
}

func (p *payloadBuffer) writeFloat32(v float32) {
	if p.err != nil {
		return
	}
	p.buf = binary.LittleEndian.AppendUint32(p.buf, math.Float32bits(v))
}

func (p *payloadBuffer) readFloat32() float32 {
	if p.err != nil {
		return 0
	}
	if p.pos+4 > len(p.buf) {
		p.err = io.ErrUnexpectedEOF
		return 0
	}
	v := math.Float32frombits(binary.LittleEndian.Uint32(p.buf[p.pos:]))
	p.pos += 4
	return v
}

func (p *payloadBuffer) writeBool(v bool) {
	if p.err != nil {
		return
	}
	if v {
		p.buf = append(p.buf, 1)
	} else {
		p.buf = append(p.buf, 0)
	}
}

func (p *payloadBuffer) readBool() bool {
	if p.err != nil {
		return false
	}
	if p.pos >= len(p.buf) {
		p.err = io.ErrUnexpectedEOF
		return false
	}
	v := p.buf[p.pos] != 0
	p.pos++
	return v
}

func (p *payloadBuffer) writeBytes(b []byte) {
	if p.err != nil {
		return
	}
	if len(b) > 65535 {
		p.err = fmt.Errorf("bytes too long: %d", len(b))
		return
	}
	p.buf = binary.LittleEndian.AppendUint16(p.buf, uint16(len(b)))
	p.buf = append(p.buf, b...)
}

func (p *payloadBuffer) readBytes() []byte {
	if p.err != nil {
		return nil
	}
	if p.pos+2 > len(p.buf) {
		p.err = io.ErrUnexpectedEOF
		return nil
	}
	l := binary.LittleEndian.Uint16(p.buf[p.pos:])
	p.pos += 2

	if p.pos+int(l) > len(p.buf) {
		p.err = io.ErrUnexpectedEOF
		return nil
	}
	b := make([]byte, l)
	copy(b, p.buf[p.pos:p.pos+int(l)])
	p.pos += int(l)
	return b
}

// writeSegmentStats serializes segment stats for persistence (v2 format).
// Format:
//
//	hasStats (1 byte): 0 = nil, 1 = present
//	if present:
//	  === Core Counts ===
//	  totalRows (4 bytes)
//	  liveRows (4 bytes)
//	  deletedRatio (float32)
//	  filterEntropy (float32)
//
//	  === Numeric Fields ===
//	  numNumeric (2 bytes)
//	  for each:
//	    key (string), min (float64), max (float64), hasNaN (bool)
//	    histogram (16 x uint32)
//	    histogramMin (16 x float64)
//	    histogramMax (16 x float64)
//	    sum (float64), sumSq (float64), count (uint32)
//
//	  === Categorical Fields ===
//	  numCategorical (2 bytes)
//	  for each:
//	    key (string), distinctCount (4 bytes), dominantValue (string), dominantRatio (float32)
//	    entropy (float32)
//	    numTopK (2 bytes), topK...
//
//	  === HasFields ===
//	  numHasFields (2 bytes)
//	  for each:
//	    key (string)
//
//	  === Vector Stats ===
//	  hasVector (1 byte)
//	  if hasVector:
//	    minNorm, maxNorm, meanNorm (float32 each)
//	    avgDistToCentroid, radius95, radiusMax (float32 each)
//	    centroidLen (2 bytes), centroid (bytes)
//
//	  === Shape Stats ===
//	  hasShape (1 byte)
//	  if hasShape:
//	    isSortedByTimestamp (bool), timestampField (string)
//	    isAppendOnly (bool), isClustered (bool), clusterTightness (float32)
func (p *payloadBuffer) writeSegmentStats(stats *SegmentStats) {
	if p.err != nil {
		return
	}
	if stats == nil {
		p.writeBool(false)
		return
	}
	p.writeBool(true)

	// Core counts
	p.writeUint32(stats.TotalRows)
	p.writeUint32(stats.LiveRows)
	p.writeFloat32(stats.DeletedRatio)
	p.writeFloat32(stats.FilterEntropy)

	// Numeric stats (with enhanced histogram)
	p.writeUint16(uint16(len(stats.Numeric)))
	for k, v := range stats.Numeric { //nolint:gocritic // can't take address of map value
		p.writeString(k)
		p.writeFloat64(v.Min)
		p.writeFloat64(v.Max)
		p.writeBool(v.HasNaN)
		// Write histogram (16 bins)
		for i := range HistogramBins {
			p.writeUint32(v.Histogram[i])
		}
		// Write per-bin min/max
		for i := range HistogramBins {
			p.writeFloat64(v.HistogramMin[i])
		}
		for i := range HistogramBins {
			p.writeFloat64(v.HistogramMax[i])
		}
		// Write variance tracking
		p.writeFloat64(v.Sum)
		p.writeFloat64(v.SumSq)
		p.writeUint32(v.Count)
	}

	// Categorical stats (with entropy and Bloom filter)
	p.writeUint16(uint16(len(stats.Categorical)))
	for k, v := range stats.Categorical {
		p.writeString(k)
		p.writeUint32(v.DistinctCount)
		p.writeString(v.DominantValue)
		p.writeFloat32(v.DominantRatio)
		p.writeFloat32(v.Entropy)
		p.writeUint16(uint16(len(v.TopK)))
		for _, vf := range v.TopK {
			p.writeString(vf.Value)
			p.writeUint32(vf.Count)
		}
		// Write Bloom filter (v3)
		if v.Bloom != nil {
			p.writeBool(true)
			var bloomBuf bytes.Buffer
			if _, err := v.Bloom.WriteTo(&bloomBuf); err != nil {
				// If Bloom serialization fails, write as no Bloom
				p.buf = p.buf[:len(p.buf)-1] // Remove the true we wrote
				p.writeBool(false)
			} else {
				p.writeBytes(bloomBuf.Bytes())
			}
		} else {
			p.writeBool(false)
		}
	}

	// HasFields
	p.writeUint16(uint16(len(stats.HasFields)))
	for k := range stats.HasFields {
		p.writeString(k)
	}

	// Vector stats (enhanced)
	if stats.Vector == nil {
		p.writeBool(false)
	} else {
		p.writeBool(true)
		p.writeFloat32(stats.Vector.MinNorm)
		p.writeFloat32(stats.Vector.MaxNorm)
		p.writeFloat32(stats.Vector.MeanNorm)
		p.writeFloat32(stats.Vector.AvgDistanceToCentroid)
		p.writeFloat32(stats.Vector.Radius95)
		p.writeFloat32(stats.Vector.RadiusMax)
		// Centroid as int8 slice
		centroidBytes := make([]byte, len(stats.Vector.Centroid))
		for i, v := range stats.Vector.Centroid {
			centroidBytes[i] = byte(v)
		}
		p.writeBytes(centroidBytes)
	}

	// Shape stats
	if stats.Shape == nil {
		p.writeBool(false)
	} else {
		p.writeBool(true)
		p.writeBool(stats.Shape.IsSortedByTimestamp)
		p.writeString(stats.Shape.TimestampField)
		p.writeBool(stats.Shape.IsAppendOnly)
		p.writeBool(stats.Shape.IsClustered)
		p.writeFloat32(stats.Shape.ClusterTightness)
	}
}

func (p *payloadBuffer) readSegmentStats() *SegmentStats {
	if p.err != nil {
		return nil
	}
	if !p.readBool() {
		return nil
	}

	stats := &SegmentStats{}

	// Core counts
	stats.TotalRows = p.readUint32()
	stats.LiveRows = p.readUint32()
	stats.DeletedRatio = p.readFloat32()
	stats.FilterEntropy = p.readFloat32()

	// Numeric stats (with enhanced histogram)
	numNumeric := p.readUint16()
	if numNumeric > 0 {
		stats.Numeric = make(map[string]NumericFieldStats, numNumeric)
		for i := 0; i < int(numNumeric); i++ {
			k := p.readString()
			v := NumericFieldStats{
				Min:    p.readFloat64(),
				Max:    p.readFloat64(),
				HasNaN: p.readBool(),
			}
			// Read histogram (16 bins)
			for j := 0; j < HistogramBins; j++ {
				v.Histogram[j] = p.readUint32()
			}
			// Read per-bin min/max
			for j := 0; j < HistogramBins; j++ {
				v.HistogramMin[j] = p.readFloat64()
			}
			for j := 0; j < HistogramBins; j++ {
				v.HistogramMax[j] = p.readFloat64()
			}
			// Read variance tracking
			v.Sum = p.readFloat64()
			v.SumSq = p.readFloat64()
			v.Count = p.readUint32()
			stats.Numeric[k] = v
		}
	}

	// Categorical stats (with entropy and Bloom filter in v3)
	numCategorical := p.readUint16()
	if numCategorical > 0 {
		stats.Categorical = make(map[string]CategoricalStats, numCategorical)
		for i := 0; i < int(numCategorical); i++ {
			k := p.readString()
			cs := CategoricalStats{
				DistinctCount: p.readUint32(),
				DominantValue: p.readString(),
				DominantRatio: p.readFloat32(),
				Entropy:       p.readFloat32(),
			}
			numTopK := p.readUint16()
			if numTopK > 0 {
				cs.TopK = make([]ValueFreq, numTopK)
				for j := 0; j < int(numTopK); j++ {
					cs.TopK[j].Value = p.readString()
					cs.TopK[j].Count = p.readUint32()
				}
			}
			// Read Bloom filter (v3+)
			if p.readBool() {
				bloomData := p.readBytes()
				if len(bloomData) > 0 && p.err == nil {
					bloomReader := bytes.NewReader(bloomData)
					cs.Bloom, _ = ReadBloomFilter(bloomReader) // Ignore error, Bloom is optional
				}
			}
			stats.Categorical[k] = cs
		}
	}

	// HasFields
	numHasFields := p.readUint16()
	if numHasFields > 0 {
		stats.HasFields = make(map[string]bool, numHasFields)
		for i := 0; i < int(numHasFields); i++ {
			k := p.readString()
			stats.HasFields[k] = true
		}
	}

	// Vector stats (enhanced)
	if p.readBool() {
		stats.Vector = &VectorStats{
			MinNorm:               p.readFloat32(),
			MaxNorm:               p.readFloat32(),
			MeanNorm:              p.readFloat32(),
			AvgDistanceToCentroid: p.readFloat32(),
			Radius95:              p.readFloat32(),
			RadiusMax:             p.readFloat32(),
		}
		centroidBytes := p.readBytes()
		if len(centroidBytes) > 0 {
			stats.Vector.Centroid = make([]int8, len(centroidBytes))
			for i, b := range centroidBytes {
				stats.Vector.Centroid[i] = int8(b)
			}
		}
	}

	// Shape stats
	if p.readBool() {
		stats.Shape = &ShapeStats{
			IsSortedByTimestamp: p.readBool(),
			TimestampField:      p.readString(),
			IsAppendOnly:        p.readBool(),
			IsClustered:         p.readBool(),
			ClusterTightness:    p.readFloat32(),
		}
	}

	return stats
}

// readSegmentStatsV1 reads legacy v1 format stats.
func (p *payloadBuffer) readSegmentStatsV1() *SegmentStats {
	if p.err != nil {
		return nil
	}
	if !p.readBool() {
		return nil
	}

	stats := &SegmentStats{}

	// Numeric stats (v1: no enhanced histogram fields)
	numNumeric := p.readUint16()
	if numNumeric > 0 {
		stats.Numeric = make(map[string]NumericFieldStats, numNumeric)
		for i := 0; i < int(numNumeric); i++ {
			k := p.readString()
			v := NumericFieldStats{
				Min:    p.readFloat64(),
				Max:    p.readFloat64(),
				HasNaN: p.readBool(),
			}
			// Read histogram (16 bins)
			for j := 0; j < HistogramBins; j++ {
				v.Histogram[j] = p.readUint32()
			}
			// v1 doesn't have HistogramMin/Max, Sum, SumSq, Count
			stats.Numeric[k] = v
		}
	}

	// Categorical stats (v1: no entropy)
	numCategorical := p.readUint16()
	if numCategorical > 0 {
		stats.Categorical = make(map[string]CategoricalStats, numCategorical)
		for i := 0; i < int(numCategorical); i++ {
			k := p.readString()
			cs := CategoricalStats{
				DistinctCount: p.readUint32(),
				DominantValue: p.readString(),
				DominantRatio: p.readFloat32(),
				// v1 doesn't have Entropy
			}
			numTopK := p.readUint16()
			if numTopK > 0 {
				cs.TopK = make([]ValueFreq, numTopK)
				for j := 0; j < int(numTopK); j++ {
					cs.TopK[j].Value = p.readString()
					cs.TopK[j].Count = p.readUint32()
				}
			}
			stats.Categorical[k] = cs
		}
	}

	// HasFields
	numHasFields := p.readUint16()
	if numHasFields > 0 {
		stats.HasFields = make(map[string]bool, numHasFields)
		for i := 0; i < int(numHasFields); i++ {
			k := p.readString()
			stats.HasFields[k] = true
		}
	}

	// Vector stats (v1: basic only)
	if p.readBool() {
		stats.Vector = &VectorStats{
			MinNorm:  p.readFloat32(),
			MaxNorm:  p.readFloat32(),
			MeanNorm: p.readFloat32(),
			// v1 doesn't have AvgDistanceToCentroid, Radius95, RadiusMax
		}
		centroidBytes := p.readBytes()
		if len(centroidBytes) > 0 {
			stats.Vector.Centroid = make([]int8, len(centroidBytes))
			for i, b := range centroidBytes {
				stats.Vector.Centroid[i] = int8(b)
			}
		}
	}

	// v1 doesn't have Shape stats, TotalRows, LiveRows, DeletedRatio, FilterEntropy

	return stats
}

// readSegmentStatsV2 reads v2 format stats (no Bloom filters).
func (p *payloadBuffer) readSegmentStatsV2() *SegmentStats {
	if p.err != nil {
		return nil
	}
	if !p.readBool() {
		return nil
	}

	stats := &SegmentStats{
		TotalRows:     p.readUint32(),
		LiveRows:      p.readUint32(),
		DeletedRatio:  p.readFloat32(),
		FilterEntropy: p.readFloat32(),
	}

	// Numeric stats
	numNumeric := p.readUint16()
	if numNumeric > 0 {
		stats.Numeric = make(map[string]NumericFieldStats, numNumeric)
		for i := 0; i < int(numNumeric); i++ {
			k := p.readString()
			v := NumericFieldStats{
				Min:    p.readFloat64(),
				Max:    p.readFloat64(),
				HasNaN: p.readBool(),
			}
			// Read histogram (16 bins)
			for j := 0; j < HistogramBins; j++ {
				v.Histogram[j] = p.readUint32()
			}
			// Read per-bin min/max
			for j := 0; j < HistogramBins; j++ {
				v.HistogramMin[j] = p.readFloat64()
			}
			for j := 0; j < HistogramBins; j++ {
				v.HistogramMax[j] = p.readFloat64()
			}
			// Read variance tracking
			v.Sum = p.readFloat64()
			v.SumSq = p.readFloat64()
			v.Count = p.readUint32()
			stats.Numeric[k] = v
		}
	}

	// Categorical stats (v2: with entropy, no Bloom)
	numCategorical := p.readUint16()
	if numCategorical > 0 {
		stats.Categorical = make(map[string]CategoricalStats, numCategorical)
		for i := 0; i < int(numCategorical); i++ {
			k := p.readString()
			cs := CategoricalStats{
				DistinctCount: p.readUint32(),
				DominantValue: p.readString(),
				DominantRatio: p.readFloat32(),
				Entropy:       p.readFloat32(),
			}
			numTopK := p.readUint16()
			if numTopK > 0 {
				cs.TopK = make([]ValueFreq, numTopK)
				for j := 0; j < int(numTopK); j++ {
					cs.TopK[j].Value = p.readString()
					cs.TopK[j].Count = p.readUint32()
				}
			}
			// v2 doesn't have Bloom filter
			stats.Categorical[k] = cs
		}
	}

	// HasFields
	numHasFields := p.readUint16()
	if numHasFields > 0 {
		stats.HasFields = make(map[string]bool, numHasFields)
		for i := 0; i < int(numHasFields); i++ {
			k := p.readString()
			stats.HasFields[k] = true
		}
	}

	// Vector stats (enhanced)
	if p.readBool() {
		stats.Vector = &VectorStats{
			MinNorm:               p.readFloat32(),
			MaxNorm:               p.readFloat32(),
			MeanNorm:              p.readFloat32(),
			AvgDistanceToCentroid: p.readFloat32(),
			Radius95:              p.readFloat32(),
			RadiusMax:             p.readFloat32(),
		}
		centroidBytes := p.readBytes()
		if len(centroidBytes) > 0 {
			stats.Vector.Centroid = make([]int8, len(centroidBytes))
			for i, b := range centroidBytes {
				stats.Vector.Centroid[i] = int8(b)
			}
		}
	}

	// Shape stats
	if p.readBool() {
		stats.Shape = &ShapeStats{
			IsSortedByTimestamp: p.readBool(),
			TimestampField:      p.readString(),
			IsAppendOnly:        p.readBool(),
			IsClustered:         p.readBool(),
			ClusterTightness:    p.readFloat32(),
		}
	}

	return stats
}
