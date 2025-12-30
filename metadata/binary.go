package metadata

import (
	"encoding/binary"
	"errors"
	"math"
	"unique"
)

// MarshalBinary implements encoding.BinaryMarshaler.
// It uses a compact binary format optimized for Vecgo metadata.
func (m Metadata) MarshalBinary() ([]byte, error) {
	// Estimate size: 4 bytes count + (avg key len 5 + avg val len 5) * count
	// A rough guess to avoid some allocations.
	buf := make([]byte, 0, 4+len(m)*16)

	// Write map size (uvarint)
	buf = binary.AppendUvarint(buf, uint64(len(m)))

	for k, v := range m {
		// Write Key (string)
		buf = binary.AppendUvarint(buf, uint64(len(k)))
		buf = append(buf, k...)

		// Write Value
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
	count, n := binary.Uvarint(data)
	if n <= 0 {
		return errors.New("invalid metadata length")
	}
	data = data[n:]

	if *m == nil {
		*m = make(Metadata, count)
	}

	for range count {
		// Read Key
		kLen, n := binary.Uvarint(data)
		if n <= 0 {
			return errors.New("invalid key length")
		}
		data = data[n:]
		if uint64(len(data)) < kLen {
			return errors.New("short buffer for key")
		}
		key := string(data[:kLen])
		data = data[kLen:]

		// Read Value
		val, remaining, err := parseValue(data)
		if err != nil {
			return err
		}
		(*m)[key] = val
		data = remaining
	}
	return nil
}

// MarshalMetadataMap encodes a map of metadata documents (keyed by internal ID).
func MarshalMetadataMap(m map[uint64]Metadata) ([]byte, error) {
	// Estimate size: 4 bytes count + (8 bytes ID + avg metadata len 50) * count
	buf := make([]byte, 0, 4+len(m)*58)

	// Write map size (uvarint)
	buf = binary.AppendUvarint(buf, uint64(len(m)))

	for k, v := range m {
		// Write ID (uint64)
		buf = binary.LittleEndian.AppendUint64(buf, k)

		// Write Metadata
		b, err := v.MarshalBinary()
		if err != nil {
			return nil, err
		}
		buf = binary.AppendUvarint(buf, uint64(len(b)))
		buf = append(buf, b...)
	}
	return buf, nil
}

// UnmarshalMetadataMap decodes a map of metadata documents.
func UnmarshalMetadataMap(data []byte) (map[uint64]Metadata, error) {
	count, n := binary.Uvarint(data)
	if n <= 0 {
		return nil, errors.New("invalid metadata map length")
	}
	data = data[n:]

	m := make(map[uint64]Metadata, count)

	for range count {
		// Read ID
		if len(data) < 8 {
			return nil, errors.New("short buffer for ID")
		}
		id := binary.LittleEndian.Uint64(data)
		data = data[8:]

		// Read Metadata
		mLen, n := binary.Uvarint(data)
		if n <= 0 {
			return nil, errors.New("invalid metadata length")
		}
		data = data[n:]
		if uint64(len(data)) < mLen {
			return nil, errors.New("short buffer for metadata")
		}

		var meta Metadata
		if err := meta.UnmarshalBinary(data[:mLen]); err != nil {
			return nil, err
		}
		m[id] = meta
		data = data[mLen:]
	}
	return m, nil
}

func appendValue(buf []byte, v Value) ([]byte, error) {
	// Write Kind (byte)
	buf = append(buf, byte(v.Kind))

	switch v.Kind {
	case KindNull:
		// No payload
	case KindInt:
		buf = binary.AppendVarint(buf, v.I64)
	case KindFloat:
		bits := math.Float64bits(v.F64)
		buf = binary.LittleEndian.AppendUint64(buf, bits)
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
		return nil, errors.New("unknown metadata kind")
	}
	return buf, nil
}

func parseValue(data []byte) (Value, []byte, error) {
	if len(data) == 0 {
		return Value{}, nil, errors.New("short buffer for value kind")
	}
	kind := Kind(data[0])
	data = data[1:]

	var v Value
	v.Kind = kind

	switch kind {
	case KindNull:
		// No payload
	case KindInt:
		i, n := binary.Varint(data)
		if n <= 0 {
			return v, nil, errors.New("invalid int value")
		}
		v.I64 = i
		data = data[n:]
	case KindFloat:
		if len(data) < 8 {
			return v, nil, errors.New("short buffer for float")
		}
		bits := binary.LittleEndian.Uint64(data)
		v.F64 = math.Float64frombits(bits)
		data = data[8:]
	case KindString:
		sLen, n := binary.Uvarint(data)
		if n <= 0 {
			return v, nil, errors.New("invalid string length")
		}
		data = data[n:]
		if uint64(len(data)) < sLen {
			return v, nil, errors.New("short buffer for string")
		}
		v.s = unique.Make(string(data[:sLen]))
		data = data[sLen:]
	case KindBool:
		if len(data) == 0 {
			return v, nil, errors.New("short buffer for bool")
		}
		v.B = data[0] != 0
		data = data[1:]
	case KindArray:
		aLen, n := binary.Uvarint(data)
		if n <= 0 {
			return v, nil, errors.New("invalid array length")
		}
		data = data[n:]
		v.A = make([]Value, aLen)
		for i := uint64(0); i < aLen; i++ {
			item, remaining, err := parseValue(data)
			if err != nil {
				return v, nil, err
			}
			v.A[i] = item
			data = remaining
		}
	default:
		return v, nil, errors.New("unknown metadata kind")
	}
	return v, data, nil
}
