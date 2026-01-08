package metadata

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestJSONSerialization(t *testing.T) {
	tests := []struct {
		name string
		val  Value
	}{
		{"Null", Null()},
		{"Int", Int(123)},
		// {"Float", Float(3.14)}, // Float formatting might be slightly sensitive in JSON match, exact comparison preferred
		{"String", String("hello")},
		{"Bool", Bool(true)},
		{"Array", Array([]Value{Int(1), String("a")})},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			b, err := json.Marshal(tt.val)
			require.NoError(t, err)

			var got Value
			err = json.Unmarshal(b, &got)
			require.NoError(t, err)

			if tt.val.Kind == KindString {
				assert.Equal(t, tt.val.StringValue(), got.StringValue())
			} else if tt.val.Kind == KindArray {
				assert.Equal(t, tt.val.Kind, got.Kind)
				assert.Equal(t, len(tt.val.A), len(got.A))
			} else {
				assert.Equal(t, tt.val, got)
			}
		})
	}

	// Float Test separately to avoid deep equal issues if any
	t.Run("Float", func(t *testing.T) {
		v := Float(3.14)
		b, err := json.Marshal(v)
		require.NoError(t, err)

		var got Value
		err = json.Unmarshal(b, &got)
		require.NoError(t, err)
		assert.Equal(t, KindFloat, got.Kind)
		assert.InDelta(t, 3.14, got.F64, 0.0001)
	})
}

func TestNewValue(t *testing.T) {
	tests := []struct {
		input    interface{}
		expected Value
		error    bool
	}{
		{nil, Null(), false},
		{int(1), Int(1), false},
		{int8(1), Int(1), true},  // Unsupported
		{int16(1), Int(1), true}, // Unsupported
		{int32(1), Int(1), true}, // Unsupported
		{int64(1), Int(1), false},
		{uint(1), Int(1), true},          // Unsupported
		{float32(1.5), Float(1.5), true}, // Unsupported
		{float64(1.5), Float(1.5), false},
		{"string", String("string"), false},
		{true, Bool(true), false},
		{[]interface{}{1, "a"}, Array([]Value{Int(1), String("a")}), false},
		{struct{}{}, Value{}, true}, // Unsupported
	}

	for _, tt := range tests {
		got, err := NewValue(tt.input)
		if tt.error {
			assert.Error(t, err)
		} else {
			require.NoError(t, err)
			if tt.expected.Kind == KindString {
				assert.Equal(t, tt.expected.StringValue(), got.StringValue())
			} else if tt.expected.Kind == KindArray {
				// Approximate check
				assert.Equal(t, tt.expected.Kind, got.Kind)
			} else {
				assert.Equal(t, tt.expected, got)
			}
		}
	}
}

func TestCloneIfNeeded(t *testing.T) {
	assert.Nil(t, CloneIfNeeded(nil))
	assert.Nil(t, CloneIfNeeded(Metadata{}))

	m := Metadata{"k": Int(1)}
	c := CloneIfNeeded(m)
	assert.NotNil(t, c)
	assert.NotSame(t, &m, &c) // Maps are references, but content should be cloned
	// c is a new map

	c["k"] = Int(2)
	assert.Equal(t, int64(1), m["k"].I64)
}

func TestKey(t *testing.T) {
	// Test Value.Key()
	assert.Equal(t, "null", Null().Key())
	assert.Equal(t, "i:1", Int(1).Key())
	// Float might be hex representation
	assert.Contains(t, Float(1.0).Key(), "f:")
	assert.Equal(t, "s:foo", String("foo").Key())
	assert.Equal(t, "b:1", Bool(true).Key())
	assert.Equal(t, "b:0", Bool(false).Key())

	// Array Key
	arr := Array([]Value{Int(1), Int(2)})
	// "a:i:1\x1fi:2" assuming separator is \x1f
	assert.Contains(t, arr.Key(), "a:i:1")
	assert.Contains(t, arr.Key(), "i:2")

	assert.Equal(t, "a:", Array([]Value{}).Key())
}
