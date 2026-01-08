package metadata

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestFieldTypeString(t *testing.T) {
	tests := []struct {
		ft       FieldType
		expected string
	}{
		{FieldTypeAny, "Any"},
		{FieldTypeInt, "Int"},
		{FieldTypeFloat, "Float"},
		{FieldTypeString, "String"},
		{FieldTypeBool, "Bool"},
		{FieldTypeArray, "Array"},
		{FieldType(99), "Unknown"},
	}

	for _, tt := range tests {
		assert.Equal(t, tt.expected, tt.ft.String())
	}
}

func TestSchemaValidate(t *testing.T) {
	s := Schema{
		"s": FieldTypeString,
		"i": FieldTypeInt,
		"f": FieldTypeFloat,
		"a": FieldTypeAny,
	}

	tests := []struct {
		name    string
		doc     Document
		wantErr bool
	}{
		{
			"Valid",
			Document{
				"s": String("val"),
				"i": Int(10),
				"f": Float(3.5),
				"a": Bool(true),
			},
			false,
		},
		{
			"Valid_IntAsFloat",
			Document{"f": Int(10)}, // Allowed upgrade
			false,
		},
		{
			"Valid_UnknownField",
			Document{"unknown": Int(1)}, // Should be ignored
			false,
		},
		{
			"Valid_Null",
			Document{"s": Null()},
			false,
		},
		{
			"Invalid_Type",
			Document{"s": Int(1)},
			true,
		},
		{
			"Invalid_IntAsBool",
			Document{"i": Bool(true)},
			true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := s.Validate(tt.doc)
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}

	// Nil schema
	var nilSchema Schema
	assert.NoError(t, nilSchema.Validate(Document{"a": Int(1)}))
}

func TestSchemaValidateMap(t *testing.T) {
	s := Schema{
		"s":   FieldTypeString,
		"i":   FieldTypeInt,
		"f":   FieldTypeFloat,
		"b":   FieldTypeBool,
		"arr": FieldTypeArray,
	}

	tests := []struct {
		name    string
		input   map[string]any
		wantErr bool
	}{
		{
			"Valid",
			map[string]any{
				"s":   "val",
				"i":   123,
				"f":   3.14,
				"b":   true,
				"arr": []any{1, 2},
			},
			false,
		},
		{
			"Valid_Subtypes",
			map[string]any{
				"i": int64(10), // int64 -> int
				"f": int(10),   // int -> float
			},
			false,
		},
		{
			"Valid_Null",
			map[string]any{"s": nil}, // nil allows any?
			false,
		},
		{
			"Invalid_StringAsInt",
			map[string]any{"i": "not_int"},
			true,
		},
		{
			"Invalid_BoolAsFloat",
			map[string]any{"f": true},
			true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := s.ValidateMap(tt.input)
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}

	// Nil schema
	var nilSchema Schema
	assert.NoError(t, nilSchema.ValidateMap(map[string]any{"a": 1}))
}
