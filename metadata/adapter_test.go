package metadata

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestFromAny(t *testing.T) {
	t.Run("Scalars", func(t *testing.T) {
		tests := []struct {
			name     string
			input    any
			expected Value
		}{
			{"nil", nil, Null()},
			{"Value", Int(1), Int(1)},
			{"bool true", true, Bool(true)},
			{"bool false", false, Bool(false)},
			{"string", "hello", String("hello")},
			{"float64", 3.14, Float(3.14)},
			{"float32", float32(1.5), Float(1.5)},
			{"int", int(1), Int(1)},
			{"int8", int8(1), Int(1)},
			{"int16", int16(1), Int(1)},
			{"int32", int32(1), Int(1)},
			{"int64", int64(1), Int(1)},
			{"uint", uint(1), Int(1)},
			{"uint8", uint8(1), Int(1)},
			{"uint16", uint16(1), Int(1)},
			{"uint32", uint32(1), Int(1)},
			{"uint32 max", uint32(math.MaxUint32), Int(int64(math.MaxUint32))},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				v, err := FromAny(tc.input)
				assert.NoError(t, err)
				assert.Equal(t, tc.expected, v)
			})
		}
	})

	t.Run("Uint64 Range", func(t *testing.T) {
		// Code seems to restrict uint64 to uint32 range
		v, err := FromAny(uint64(math.MaxUint32))
		assert.NoError(t, err)
		assert.Equal(t, int64(math.MaxUint32), v.I64)

		// This should fail based on implementation
		_, err = FromAny(uint64(math.MaxUint32 + 1))
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "out of range")
	})

	t.Run("Slices", func(t *testing.T) {
		t.Run("[]Value", func(t *testing.T) {
			input := []Value{Int(1), String("s")}
			v, err := FromAny(input)
			assert.NoError(t, err)
			arr, _ := v.AsArray()
			assert.Equal(t, input, arr)
		})

		t.Run("[]any", func(t *testing.T) {
			input := []any{1, "s", true}
			v, err := FromAny(input)
			assert.NoError(t, err)
			arr, _ := v.AsArray()
			assert.Len(t, arr, 3)
			assert.Equal(t, Int(1), arr[0])
			assert.Equal(t, String("s"), arr[1])
			assert.Equal(t, Bool(true), arr[2])
		})

		t.Run("[]any error", func(t *testing.T) {
			input := []any{make(chan int)} // unsupported
			_, err := FromAny(input)
			assert.Error(t, err)
		})

		t.Run("[]string", func(t *testing.T) {
			input := []string{"a", "b"}
			v, err := FromAny(input)
			assert.NoError(t, err)
			arr, _ := v.AsArray()
			assert.Len(t, arr, 2)
			assert.Equal(t, String("a"), arr[0])
		})

		t.Run("[]int", func(t *testing.T) {
			input := []int{1, 2}
			v, err := FromAny(input)
			assert.NoError(t, err)
			arr, _ := v.AsArray()
			assert.Len(t, arr, 2)
			assert.Equal(t, Int(1), arr[0])
		})

		t.Run("[]float64", func(t *testing.T) {
			input := []float64{1.1, 2.2}
			v, err := FromAny(input)
			assert.NoError(t, err)
			arr, _ := v.AsArray()
			assert.Len(t, arr, 2)
			assert.Equal(t, Float(1.1), arr[0])
		})
	})

	t.Run("Unsupported", func(t *testing.T) {
		_, err := FromAny(make(chan int))
		assert.Error(t, err)

		_, err = FromAny([]byte("bytes")) // bytes treat as unsupported currently?
		assert.Error(t, err)
	})
}

func TestDocumentFromAny(t *testing.T) {
	t.Run("Success", func(t *testing.T) {
		m := map[string]any{
			"i": 123,
			"s": "foo",
		}
		doc, err := DocumentFromAny(m)
		require.NoError(t, err)
		assert.Equal(t, Int(123), doc["i"])
		assert.Equal(t, String("foo"), doc["s"])
	})

	t.Run("Error", func(t *testing.T) {
		m := map[string]any{
			"bad": make(chan int),
		}
		_, err := DocumentFromAny(m)
		assert.Error(t, err)
	})
}
