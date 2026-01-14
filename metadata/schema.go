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

// Validate checks if the given metadata document conforms to the schema.
func (s Schema) Validate(doc Document) error {
	if s == nil {
		return nil
	}
	for k, v := range doc {
		expectedType, ok := s[k]
		if !ok {
			continue
		}

		if !checkKind(v.Kind, expectedType) {
			return fmt.Errorf("field %q has invalid type %s, expected %s", k, v.Kind, expectedType)
		}
	}
	return nil
}

// ValidateMap checks if the given metadata map conforms to the schema.
// This is useful for validating untyped input (e.g. from JSON) before conversion.
func (s Schema) ValidateMap(md map[string]any) error {
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

func checkKind(k Kind, expected FieldType) bool {
	if k == KindNull {
		return true
	}
	switch expected {
	case FieldTypeAny:
		return true
	case FieldTypeInt:
		return k == KindInt
	case FieldTypeFloat:
		return k == KindFloat || k == KindInt // Allow upgrading Int to Float
	case FieldTypeString:
		return k == KindString
	case FieldTypeBool:
		return k == KindBool
	case FieldTypeArray:
		return k == KindArray
	}
	return false
}

func checkType(v any, expected FieldType) bool {
	if v == nil {
		return true // Null is always valid? Or depends on schema? Let's assume nullable for now.
	}

	switch expected {
	case FieldTypeAny:
		return true
	case FieldTypeInt:
		switch val := v.(type) {
		case int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64:
			return true
		case float64:
			// JSON unmarshals numbers as float64. Check if it's an integer.
			return val == float64(int64(val))
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
