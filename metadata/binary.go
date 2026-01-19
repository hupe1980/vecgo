package metadata

import (
	"encoding/binary"
	"errors"
	"math"
	"unique"
)

// Binary format (self-describing, no redundant length prefixes):
//
// Metadata:
//   [count: uvarint] [key-value pairs...]
//   Key-Value: [keyLen: uvarint] [key: bytes] [value]
//
// Value:
//   [kind: 1 byte] [payload...]
//   KindNull:   no payload
//   KindInt:    varint
//   KindFloat:  8 bytes (IEEE 754 little-endian)
//   KindString: [len: uvarint] [bytes]
//   KindBool:   1 byte (0 or 1)
//   KindArray:  [len: uvarint] [values...]
//
// MetadataMap:
//   [count: uvarint] [entries...]
//   Entry: [id: uint64 LE] [metadata] (no length prefix - self-terminating)

var (
	errInvalidLength = errors.New("metadata: invalid length encoding")
	errShortBuffer   = errors.New("metadata: short buffer")
	errUnknownKind   = errors.New("metadata: unknown value kind")
	errInvalidInt    = errors.New("metadata: invalid int encoding")
)

// MarshalBinary implements encoding.BinaryMarshaler.
// For zero-allocation marshaling into an existing buffer, use AppendBinary.
func (m Metadata) MarshalBinary() ([]byte, error) {
	// Estimate: 4 bytes count + (avg key 8 + avg val 8) * count
	buf := make([]byte, 0, 4+len(m)*16)
	return m.AppendBinary(buf)
}

// AppendBinary appends the binary encoding of Metadata to buf and returns the extended buffer.
// This is the zero-allocation path for marshaling into an existing buffer.
func (m Metadata) AppendBinary(buf []byte) ([]byte, error) {
	buf = binary.AppendUvarint(buf, uint64(len(m)))

	for k, v := range m {
		buf = binary.AppendUvarint(buf, uint64(len(k)))
		buf = append(buf, k...)

		var err error
		buf, err = appendValue(buf, v)
		if err != nil {
			return nil, err
		}
	}
	return buf, nil
}

// UnmarshalBinary implements encoding.BinaryUnmarshaler.
func (m *Metadata) UnmarshalBinary(data []byte) error {
	n, err := m.unmarshalBinaryN(data)
	if err != nil {
		return err
	}
	if n != len(data) {
		return errors.New("metadata: trailing data after unmarshal")
	}
	return nil
}

// unmarshalBinaryN decodes Metadata from data and returns the number of bytes consumed.
// This is the streaming path that allows reading from a larger buffer.
func (m *Metadata) unmarshalBinaryN(data []byte) (int, error) {
	if len(data) == 0 {
		return 0, errShortBuffer
	}

	count, n := binary.Uvarint(data)
	if n <= 0 {
		return 0, errInvalidLength
	}
	consumed := n
	data = data[n:]

	if *m == nil {
		*m = make(Metadata, count)
	}

	for range count {
		// Read key length
		kLen, n := binary.Uvarint(data)
		if n <= 0 {
			return 0, errInvalidLength
		}
		consumed += n
		data = data[n:]

		// Read key
		if uint64(len(data)) < kLen {
			return 0, errShortBuffer
		}
		key := string(data[:kLen])
		consumed += int(kLen)
		data = data[kLen:]

		// Read value
		val, bytesRead, err := parseValueN(data)
		if err != nil {
			return 0, err
		}
		consumed += bytesRead
		data = data[bytesRead:]

		(*m)[key] = val
	}
	return consumed, nil
}

// MarshalMetadataMap encodes a map of metadata documents keyed by internal ID.
// For zero-allocation marshaling, use AppendMetadataMap.
func MarshalMetadataMap(m map[uint64]Metadata) ([]byte, error) {
	// Estimate: 4 bytes count + (8 bytes ID + avg metadata 40) * count
	buf := make([]byte, 0, 4+len(m)*48)
	return AppendMetadataMap(buf, m)
}

// AppendMetadataMap appends the binary encoding of the metadata map to buf.
// This is the zero-allocation path - no intermediate buffers, no per-metadata allocations.
func AppendMetadataMap(buf []byte, m map[uint64]Metadata) ([]byte, error) {
	buf = binary.AppendUvarint(buf, uint64(len(m)))

	for id, meta := range m {
		buf = binary.LittleEndian.AppendUint64(buf, id)

		var err error
		buf, err = meta.AppendBinary(buf)
		if err != nil {
			return nil, err
		}
	}
	return buf, nil
}

// UnmarshalMetadataMap decodes a map of metadata documents.
// The format is self-describing - no length prefix needed per metadata entry.
func UnmarshalMetadataMap(data []byte) (map[uint64]Metadata, error) {
	count, n := binary.Uvarint(data)
	if n <= 0 {
		return nil, errInvalidLength
	}
	data = data[n:]

	m := make(map[uint64]Metadata, count)

	for range count {
		// Read ID (8 bytes, little-endian)
		if len(data) < 8 {
			return nil, errShortBuffer
		}
		id := binary.LittleEndian.Uint64(data)
		data = data[8:]

		// Read Metadata (self-terminating, no length prefix)
		var meta Metadata
		bytesRead, err := meta.unmarshalBinaryN(data)
		if err != nil {
			return nil, err
		}
		data = data[bytesRead:]

		m[id] = meta
	}
	return m, nil
}

func appendValue(buf []byte, v Value) ([]byte, error) {
	buf = append(buf, byte(v.Kind))

	switch v.Kind {
	case KindNull:
		// No payload
	case KindInt:
		buf = binary.AppendVarint(buf, v.I64)
	case KindFloat:
		buf = binary.LittleEndian.AppendUint64(buf, math.Float64bits(v.F64))
	case KindString:
		s := v.s.Value()
		buf = binary.AppendUvarint(buf, uint64(len(s)))
		buf = append(buf, s...)
	case KindBool:
		if v.B {
			buf = append(buf, 1)
		} else {
			buf = append(buf, 0)
		}
	case KindArray:
		buf = binary.AppendUvarint(buf, uint64(len(v.A)))
		for _, item := range v.A {
			var err error
			buf, err = appendValue(buf, item)
			if err != nil {
				return nil, err
			}
		}
	default:
		return nil, errUnknownKind
	}
	return buf, nil
}

// parseValueN parses a Value from data and returns the value and bytes consumed.
func parseValueN(data []byte) (Value, int, error) {
	if len(data) == 0 {
		return Value{}, 0, errShortBuffer
	}

	kind := Kind(data[0])
	consumed := 1
	data = data[1:]

	var v Value
	v.Kind = kind

	switch kind {
	case KindNull:
		// No payload

	case KindInt:
		i, n := binary.Varint(data)
		if n <= 0 {
			return v, 0, errInvalidInt
		}
		v.I64 = i
		consumed += n

	case KindFloat:
		if len(data) < 8 {
			return v, 0, errShortBuffer
		}
		v.F64 = math.Float64frombits(binary.LittleEndian.Uint64(data))
		consumed += 8

	case KindString:
		sLen, n := binary.Uvarint(data)
		if n <= 0 {
			return v, 0, errInvalidLength
		}
		consumed += n
		data = data[n:]
		if uint64(len(data)) < sLen {
			return v, 0, errShortBuffer
		}
		v.s = unique.Make(string(data[:sLen]))
		consumed += int(sLen)

	case KindBool:
		if len(data) == 0 {
			return v, 0, errShortBuffer
		}
		v.B = data[0] != 0
		consumed++

	case KindArray:
		aLen, n := binary.Uvarint(data)
		if n <= 0 {
			return v, 0, errInvalidLength
		}
		consumed += n
		data = data[n:]

		v.A = make([]Value, aLen)
		for i := uint64(0); i < aLen; i++ {
			item, itemBytes, err := parseValueN(data)
			if err != nil {
				return v, 0, err
			}
			v.A[i] = item
			consumed += itemBytes
			data = data[itemBytes:]
		}

	default:
		return v, 0, errUnknownKind
	}

	return v, consumed, nil
}
