package metadata

import "fmt"

// FromAny converts a Go value into a typed Value.
//
// This exists as an adapter layer for user input and legacy APIs.
func FromAny(v any) (Value, error) {
	switch x := v.(type) {
	case nil:
		return Null(), nil
	case Value:
		return x, nil
	case bool:
		return Bool(x), nil
	case string:
		return String(x), nil
	case float64:
		return Float(x), nil
	case float32:
		return Float(float64(x)), nil
	case int:
		return Int(int64(x)), nil
	case int8:
		return Int(int64(x)), nil
	case int16:
		return Int(int64(x)), nil
	case int32:
		return Int(int64(x)), nil
	case int64:
		return Int(x), nil
	case uint:
		return Int(int64(x)), nil
	case uint8:
		return Int(int64(x)), nil
	case uint16:
		return Int(int64(x)), nil
	case uint32:
		return Int(int64(x)), nil
	case uint64:
		if x > uint64(^uint32(0)) {
			// Avoid silently truncating large values.
			return Value{}, fmt.Errorf("metadata uint64 out of range: %d", x)
		}
		return Int(int64(x)), nil
	case []Value:
		return Array(x), nil
	case []any:
		arr := make([]Value, len(x))
		for i := range x {
			vv, err := FromAny(x[i])
			if err != nil {
				return Value{}, err
			}
			arr[i] = vv
		}
		return Array(arr), nil
	case []string:
		arr := make([]Value, len(x))
		for i := range x {
			arr[i] = String(x[i])
		}
		return Array(arr), nil
	case []int:
		arr := make([]Value, len(x))
		for i := range x {
			arr[i] = Int(int64(x[i]))
		}
		return Array(arr), nil
	case []float64:
		arr := make([]Value, len(x))
		for i := range x {
			arr[i] = Float(x[i])
		}
		return Array(arr), nil
	default:
		return Value{}, fmt.Errorf("unsupported metadata value type %T", v)
	}
}

// DocumentFromAny converts a legacy map[string]any document to a typed Document.
func DocumentFromAny(m map[string]any) (Document, error) {
	d := make(Document, len(m))
	for k, v := range m {
		vv, err := FromAny(v)
		if err != nil {
			return nil, err
		}
		d[k] = vv
	}
	return d, nil
}
