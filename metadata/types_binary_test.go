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
		_, err = UnmarshalMetadataMap([]byte{0x00}) // Valid count 0?
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
