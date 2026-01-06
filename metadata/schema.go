package metadata

import (
	"fmt"
)

// FieldType defines the data type of a metadata field.
type FieldType uint8

const (
	FieldTypeAny FieldType = iota
	FieldTypeInt
	FieldTypeFloat
	FieldTypeString
	FieldTypeBool
	FieldTypeArray
)

// String returns the string representation of the FieldType.
func (t FieldType) String() string {
	switch t {
	case FieldTypeAny:
		return "Any"
	case FieldTypeInt:
		return "Int"
	case FieldTypeFloat:
		return "Float"
	case FieldTypeString:
		return "String"
	case FieldTypeBool:
		return "Bool"
	case FieldTypeArray:
		return "Array"
	default:
		return "Unknown"
	}
}

// Schema defines the expected structure of metadata.
type Schema map[string]FieldType

// Validate checks if the given metadata map conforms to the schema.
// It returns an error if a field has an incorrect type.
// Unknown fields are currently allowed (open schema).
func (s Schema) Validate(md map[string]any) error {
	if s == nil {
		return nil
	}
	for k, v := range md {
		expectedType, ok := s[k]
		if !ok {
			continue
		}

		if !checkType(v, expectedType) {
			return fmt.Errorf("field %q has invalid type %T, expected %s", k, v, expectedType)
		}
	}
	return nil
}

func checkType(v any, expected FieldType) bool {
	if v == nil {
		return true // Null is always valid? Or depends on schema? Let's assume nullable for now.
	}

	switch expected {
	case FieldTypeAny:
		return true
	case FieldTypeInt:
		switch v.(type) {
		case int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64:
			return true
		case float64:
			// JSON unmarshals numbers as float64. Check if it's an integer.
			f := v.(float64)
			return f == float64(int64(f))
		}
	case FieldTypeFloat:
		switch v.(type) {
		case float32, float64:
			return true
		case int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64:
			return true // Allow ints as floats
		}
	case FieldTypeString:
		_, ok := v.(string)
		return ok
	case FieldTypeBool:
		_, ok := v.(bool)
		return ok
	case FieldTypeArray:
		// Check if slice
		switch v.(type) {
		case []any, []string, []int, []float64, []bool:
			return true
		}
	}
	return false
}
