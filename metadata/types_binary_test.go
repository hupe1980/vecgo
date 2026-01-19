package metadata

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestBinarySerialization(t *testing.T) {
	// 1. Metadata Roundtrip (indirectly tests Value serialization)
	tests := []struct {
		name string
		val  Value
	}{
		{"Null", Null()},
		{"Int", Int(math.MinInt64)},
		{"IntMax", Int(math.MaxInt64)},
		// {"Int0", Int(0)}, // Int(0) might be skipped in some encodings? No, we write it.
		{"Float", Float(3.14159)},
		{"FloatNeg", Float(-1.23)},
		// {"FloatNaN", Float(math.NaN())}, // NaN comparison is tricky
		{"FloatInf", Float(math.Inf(1))},
		{"String", String("hello world")},
		{"StringEmpty", String("")},
		{"StringNonAscii", String("こんにちは")},
		{"BoolTrue", Bool(true)},
		{"BoolFalse", Bool(false)},
		{"Array", Array([]Value{Int(1), String("a")})},
		{"NestedArray", Array([]Value{Int(1), Array([]Value{Int(2)})})},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := Metadata{"key": tt.val}

			b, err := m.MarshalBinary()
			require.NoError(t, err)

			var got Metadata
			err = got.UnmarshalBinary(b)
			require.NoError(t, err)

			assert.Equal(t, 1, len(got))
			v, ok := got["key"]
			assert.True(t, ok)

			if tt.val.Kind == KindString {
				assert.Equal(t, tt.val.StringValue(), v.StringValue())
			} else if tt.val.Kind == KindArray {
				assert.Equal(t, tt.val.Kind, v.Kind)
				assert.Equal(t, len(tt.val.A), len(v.A))
				// Shallow check for first level
				if len(tt.val.A) > 0 {
					assert.Equal(t, tt.val.A[0].Kind, v.A[0].Kind)
				}
			} else {
				assert.Equal(t, tt.val, v)
			}
		})
	}

	// 2. Map Roundtrip
	t.Run("MetadataMap", func(t *testing.T) {
		m1 := Metadata{"a": Int(1)}
		m2 := Metadata{"b": String("foo")}

		mm := map[uint64]Metadata{
			10: m1,
			20: m2,
		}

		b, err := MarshalMetadataMap(mm)
		require.NoError(t, err)

		got, err := UnmarshalMetadataMap(b)
		require.NoError(t, err)

		assert.Equal(t, len(mm), len(got))
		assert.Equal(t, int64(1), got[10]["a"].I64)
		assert.Equal(t, "foo", got[20]["b"].StringValue())
	})

	// 3. Corrupt Data
	t.Run("Corrupt", func(t *testing.T) {
		var m Metadata
		// Empty buffer
		err := m.UnmarshalBinary([]byte{})
		assert.Error(t, err)

		// Short buffer for Uvarint (high bit set but no more bytes)
		err = m.UnmarshalBinary([]byte{0xFF})
		assert.Error(t, err, "Expected error on truncated uvarint")

		// Truncated MetadataMap
		_, _ = UnmarshalMetadataMap([]byte{0x00}) // Valid count 0?
		// 0x00 is Uvarint(0). Valid empty map.

		_, err = UnmarshalMetadataMap([]byte{0x01}) // Count 1, but no data
		assert.Error(t, err)
	})
}

func TestValueHelpers(t *testing.T) {
	// Test As* methods
	i := Int(42)

	v, ok := i.AsInt64()
	assert.True(t, ok)
	assert.Equal(t, int64(42), v)

	_, ok = i.AsFloat64()
	assert.False(t, ok)

	f := Float(3.5)
	v2, ok := f.AsFloat64()
	assert.True(t, ok)
	assert.Equal(t, 3.5, v2)

	_, ok = f.AsInt64()
	assert.False(t, ok)

	s := String("42")
	v3, ok := s.AsString()
	assert.True(t, ok)
	assert.Equal(t, "42", v3)

	_, ok = s.AsInt64()
	assert.False(t, ok)
}

func TestInterning(t *testing.T) {
	doc := Document{"k": String("v")}
	interned := Intern(doc)

	assert.NotNil(t, interned)

	uninterned := Unintern(interned)
	assert.Equal(t, doc["k"].StringValue(), uninterned["k"].StringValue())

	assert.Nil(t, Intern(nil))
	assert.Nil(t, Unintern(nil))
}

func TestClone(t *testing.T) {
	v := Array([]Value{Int(1)})
	// Value.clone() is private (lowercase c) according to grep?
	// grep said `func (v Value) clone() Value`
	// But `func (d Document) Clone() Document` is public

	doc := Document{"a": v}
	c := doc.Clone()

	// Modify c deep
	// c["a"].A[0] = Int(2) // Cannot assign to field of struct in map directly usually?
	// Need to reassign value to map

	val := c["a"]
	val.A[0] = Int(2)
	c["a"] = val

	// doc should be unchanged
	assert.Equal(t, int64(1), doc["a"].A[0].I64)
}

func TestAdapter(t *testing.T) {
	// FromMap (replaces DocumentFromAny which I guessed previously)
	input := map[string]interface{}{
		"i": 123, // int
		"f": 3.14,
		"s": "str",
		"b": true,
		// "a": []interface{}{1, 2}, // Array conversion might be complex in FromMap
	}

	// The grep showed `func FromMap(m map[string]interface{}) (Document, error)`
	doc, err := FromMap(input)
	require.NoError(t, err)

	// Note: int literals in map[string]interface{} invoke as `int` (arch dependent)
	// FromMap likely handles `int`.

	iVal, ok := doc["i"].AsInt64()
	assert.True(t, ok)
	assert.Equal(t, int64(123), iVal)

	fVal, ok := doc["f"].AsFloat64()
	assert.True(t, ok)
	assert.Equal(t, 3.14, fVal)

	sVal, ok := doc["s"].AsString()
	assert.True(t, ok)
	assert.Equal(t, "str", sVal)
}

// =============================================================================
// Zero-Allocation Binary API Tests
// =============================================================================

func TestAppendBinary(t *testing.T) {
	tests := []struct {
		name string
		meta Metadata
	}{
		{"Empty", Metadata{}},
		{"SingleInt", Metadata{"x": Int(42)}},
		{"SingleString", Metadata{"name": String("vecgo")}},
		{"Mixed", Metadata{
			"id":     Int(123),
			"name":   String("test"),
			"score":  Float(0.95),
			"active": Bool(true),
		}},
		{"NestedArray", Metadata{
			"tags": Array([]Value{String("a"), String("b"), Int(1)}),
		}},
		{"DeepNested", Metadata{
			"matrix": Array([]Value{
				Array([]Value{Int(1), Int(2)}),
				Array([]Value{Int(3), Int(4)}),
			}),
		}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// AppendBinary to empty buffer should roundtrip correctly
			got, err := tt.meta.AppendBinary(nil)
			require.NoError(t, err)

			// Verify roundtrip
			var decoded Metadata
			err = decoded.UnmarshalBinary(got)
			require.NoError(t, err)
			assert.Equal(t, len(tt.meta), len(decoded), "decoded should have same number of keys")

			for k, v := range tt.meta {
				gotV, ok := decoded[k]
				require.True(t, ok, "key %q should exist", k)
				if v.Kind == KindString {
					assert.Equal(t, v.StringValue(), gotV.StringValue())
				} else if v.Kind == KindArray {
					assert.Equal(t, v.Kind, gotV.Kind)
					assert.Equal(t, len(v.A), len(gotV.A))
				} else {
					assert.Equal(t, v, gotV)
				}
			}

			// AppendBinary to pre-existing buffer (tests that it appends, not overwrites)
			prefix := []byte("PREFIX")
			got2, err := tt.meta.AppendBinary(prefix)
			require.NoError(t, err)
			assert.Equal(t, prefix, got2[:len(prefix)], "prefix should be preserved")

			// Verify roundtrip of appended portion
			var decoded2 Metadata
			err = decoded2.UnmarshalBinary(got2[len(prefix):])
			require.NoError(t, err)
			assert.Equal(t, len(tt.meta), len(decoded2))
		})
	}
}

func TestUnmarshalBinaryN(t *testing.T) {
	tests := []struct {
		name string
		meta Metadata
	}{
		{"Empty", Metadata{}},
		{"SingleInt", Metadata{"x": Int(42)}},
		{"Mixed", Metadata{"a": Int(1), "b": String("foo")}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data, err := tt.meta.MarshalBinary()
			require.NoError(t, err)

			// Append trailing garbage - unmarshalBinaryN should ignore it
			dataWithTrailing := append(data, []byte("TRAILING_GARBAGE")...)

			var got Metadata
			n, err := got.unmarshalBinaryN(dataWithTrailing)
			require.NoError(t, err)
			assert.Equal(t, len(data), n, "should return exact bytes consumed")

			// Verify content
			assert.Equal(t, len(tt.meta), len(got))
			for k, v := range tt.meta {
				gotV, ok := got[k]
				require.True(t, ok, "key %q should exist", k)
				if v.Kind == KindString {
					assert.Equal(t, v.StringValue(), gotV.StringValue())
				} else {
					assert.Equal(t, v.Kind, gotV.Kind)
				}
			}
		})
	}
}

func TestAppendMetadataMap(t *testing.T) {
	mm := map[uint64]Metadata{
		1:   {"a": Int(1)},
		100: {"b": String("test"), "c": Float(3.14)},
		999: {"nested": Array([]Value{Int(1), Int(2)})},
	}

	// AppendMetadataMap should produce valid output that roundtrips
	got, err := AppendMetadataMap(nil, mm)
	require.NoError(t, err)

	// Roundtrip
	decoded, err := UnmarshalMetadataMap(got)
	require.NoError(t, err)
	assert.Equal(t, len(mm), len(decoded))

	for id, meta := range mm {
		gotMeta, ok := decoded[id]
		require.True(t, ok, "id %d should exist", id)
		assert.Equal(t, len(meta), len(gotMeta))

		for k, v := range meta {
			gotV, ok := gotMeta[k]
			require.True(t, ok, "key %q should exist in id %d", k, id)
			if v.Kind == KindString {
				assert.Equal(t, v.StringValue(), gotV.StringValue())
			} else if v.Kind == KindArray {
				assert.Equal(t, len(v.A), len(gotV.A))
			} else {
				assert.Equal(t, v, gotV)
			}
		}
	}

	// Also verify MarshalMetadataMap produces valid output
	got2, err := MarshalMetadataMap(mm)
	require.NoError(t, err)
	decoded2, err := UnmarshalMetadataMap(got2)
	require.NoError(t, err)
	assert.Equal(t, len(mm), len(decoded2))
}

func TestMetadataMapStreamingDecode(t *testing.T) {
	// Test that MetadataMap decoding works without length prefix
	// This tests the self-describing nature of the format
	mm := map[uint64]Metadata{
		1: {"x": Int(42)},
		2: {"y": String("hello")},
	}

	data, err := MarshalMetadataMap(mm)
	require.NoError(t, err)

	// Decode should work correctly
	got, err := UnmarshalMetadataMap(data)
	require.NoError(t, err)
	assert.Equal(t, int64(42), got[1]["x"].I64)
	assert.Equal(t, "hello", got[2]["y"].StringValue())
}

func TestBinaryCorruption(t *testing.T) {
	t.Run("EmptyBuffer", func(t *testing.T) {
		var m Metadata
		_, err := m.unmarshalBinaryN([]byte{})
		assert.Error(t, err)
	})

	t.Run("TruncatedUvarint", func(t *testing.T) {
		var m Metadata
		// 0x80 has continuation bit set but no following byte
		_, err := m.unmarshalBinaryN([]byte{0x80})
		assert.Error(t, err)
	})

	t.Run("TruncatedKey", func(t *testing.T) {
		// count=1, keyLen=10, but only 3 bytes of key
		data := []byte{0x01, 0x0A, 'a', 'b', 'c'}
		var m Metadata
		_, err := m.unmarshalBinaryN(data)
		assert.Error(t, err)
	})

	t.Run("UnknownKind", func(t *testing.T) {
		// count=1, keyLen=1, key="x", kind=0xFF (invalid)
		data := []byte{0x01, 0x01, 'x', 0xFF}
		var m Metadata
		_, err := m.unmarshalBinaryN(data)
		assert.Error(t, err)
	})
}

// =============================================================================
// Benchmarks
// =============================================================================

func BenchmarkMetadataMarshal(b *testing.B) {
	meta := Metadata{
		"id":       Int(12345),
		"name":     String("benchmark-record"),
		"score":    Float(0.9876),
		"active":   Bool(true),
		"category": String("test"),
	}

	b.Run("MarshalBinary", func(b *testing.B) {
		b.ReportAllocs()
		for range b.N {
			_, _ = meta.MarshalBinary()
		}
	})

	b.Run("AppendBinary/Fresh", func(b *testing.B) {
		b.ReportAllocs()
		for range b.N {
			_, _ = meta.AppendBinary(nil)
		}
	})

	b.Run("AppendBinary/Reuse", func(b *testing.B) {
		b.ReportAllocs()
		buf := make([]byte, 0, 256)
		for range b.N {
			buf = buf[:0]
			_, _ = meta.AppendBinary(buf)
		}
	})
}

func BenchmarkMetadataMapMarshal(b *testing.B) {
	mm := make(map[uint64]Metadata, 100)
	for i := uint64(0); i < 100; i++ {
		mm[i] = Metadata{
			"id":    Int(int64(i)),
			"name":  String("record"),
			"score": Float(float64(i) / 100.0),
		}
	}

	b.Run("MarshalMetadataMap", func(b *testing.B) {
		b.ReportAllocs()
		for range b.N {
			_, _ = MarshalMetadataMap(mm)
		}
	})

	b.Run("AppendMetadataMap/Fresh", func(b *testing.B) {
		b.ReportAllocs()
		for range b.N {
			_, _ = AppendMetadataMap(nil, mm)
		}
	})

	b.Run("AppendMetadataMap/Reuse", func(b *testing.B) {
		b.ReportAllocs()
		buf := make([]byte, 0, 8192)
		for range b.N {
			buf = buf[:0]
			_, _ = AppendMetadataMap(buf, mm)
		}
	})
}
