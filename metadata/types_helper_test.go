package metadata

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestTypesHelper(t *testing.T) {
	t.Run("AsBool", func(t *testing.T) {
		b, ok := Bool(true).AsBool()
		assert.True(t, ok)
		assert.True(t, b)

		b, ok = Bool(false).AsBool()
		assert.True(t, ok)
		assert.False(t, b)

		_, ok = Int(1).AsBool()
		assert.False(t, ok)
	})

	t.Run("Document ToMap", func(t *testing.T) {
		doc := Document{
			"key": String("value"),
		}
		m := doc.ToMap()
		assert.Equal(t, "value", m["key"])
	})

	t.Run("Interface", func(t *testing.T) {
		assert.Equal(t, "test", String("test").Interface())
		assert.Equal(t, int64(123), Int(123).Interface())
		assert.Equal(t, 12.34, Float(12.34).Interface())
		assert.Equal(t, true, Bool(true).Interface())
		assert.Nil(t, Null().Interface())

		// Test array interface
		arr := Array([]Value{Int(1), String("a")})
		iface := arr.Interface()
		slice, ok := iface.([]interface{})
		assert.True(t, ok)
		assert.Equal(t, int64(1), slice[0])
		assert.Equal(t, "a", slice[1])
	})

	t.Run("Slice Helpers", func(t *testing.T) {
		strs := []string{"a", "b"}
		val := Strings(strs)
		arr, ok := val.AsArray()
		assert.True(t, ok)
		assert.Len(t, arr, 2)
		assert.Equal(t, "a", arr[0].StringValue())

		ints := []int{1, 2}
		valI := Ints(ints)
		arrI, ok := valI.AsArray()
		assert.True(t, ok)
		v1, _ := arrI[0].AsInt64()
		assert.Equal(t, int64(1), v1)

		floats := []float64{1.1, 2.2}
		valF := Floats(floats)
		arrF, ok := valF.AsArray()
		assert.True(t, ok)
		vF, _ := arrF[0].AsFloat64()
		assert.Equal(t, 1.1, vF)
	})

	t.Run("As Methods", func(t *testing.T) {
		// Int
		i, ok := Int(10).AsInt64()
		assert.True(t, ok)
		assert.Equal(t, int64(10), i)
		_, ok = String("s").AsInt64()
		assert.False(t, ok)

		// Float
		f, ok := Float(10.5).AsFloat64()
		assert.True(t, ok)
		assert.Equal(t, 10.5, f)
		_, ok = String("s").AsFloat64()
		assert.False(t, ok)

		// String
		s, ok := String("foo").AsString()
		assert.True(t, ok)
		assert.Equal(t, "foo", s)
		_, ok = Int(1).AsString()
		assert.False(t, ok)

		// Array
		a, ok := Array([]Value{Int(1)}).AsArray()
		assert.True(t, ok)
		assert.Len(t, a, 1)
		_, ok = Int(1).AsArray()
		assert.False(t, ok)
	})

	t.Run("Key", func(t *testing.T) {
		assert.Equal(t, "null", Null().Key())
		assert.Equal(t, "i:123", Int(123).Key())
		assert.Contains(t, Float(1.0).Key(), "f:")
		assert.Equal(t, "s:foo", String("foo").Key())
		assert.Equal(t, "b:1", Bool(true).Key())
		assert.Equal(t, "b:0", Bool(false).Key())

		arr := Array([]Value{Int(1), Int(2)})
		assert.Equal(t, "a:i:1\x1fi:2", arr.Key())

		assert.Equal(t, "a:", Array([]Value{}).Key())
	})

	t.Run("NewValue", func(t *testing.T) {
		v, err := NewValue(nil)
		assert.NoError(t, err)
		assert.Equal(t, KindNull, v.Kind)

		v, err = NewValue(int(1))
		assert.NoError(t, err)
		assert.Equal(t, KindInt, v.Kind)

		v, err = NewValue(int64(1))
		assert.NoError(t, err)
		assert.Equal(t, KindInt, v.Kind)

		v, err = NewValue(1.1)
		assert.NoError(t, err)
		assert.Equal(t, KindFloat, v.Kind)

		v, err = NewValue("s")
		assert.NoError(t, err)
		assert.Equal(t, KindString, v.Kind)

		v, err = NewValue(true)
		assert.NoError(t, err)
		assert.Equal(t, KindBool, v.Kind)

		v, err = NewValue([]interface{}{1, "s"})
		assert.NoError(t, err)
		assert.Equal(t, KindArray, v.Kind)

		_, err = NewValue(complex(1, 1))
		assert.Error(t, err)
	})
}
