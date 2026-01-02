package metadata

import (
	"encoding/json"
	"math"
	"strconv"
	"strings"
	"unique"
)

// Kind identifies the concrete type stored in a Value.
type Kind uint8

const (
	// KindInvalid represents an invalid kind.
	KindInvalid Kind = iota
	// KindNull represents a null value.
	KindNull
	// KindInt represents an integer value.
	KindInt
	// KindFloat represents a float value.
	KindFloat
	// KindString represents a string value.
	KindString
	// KindBool represents a boolean value.
	KindBool
	// KindArray represents an array value.
	KindArray
)

// Value is a small typed value used for metadata documents and filters.
//
// The representation is designed to make filtering fast and predictable:
// no reflection and no fmt-based stringification.
//
// NOTE: This is also used for persistence; keep it stable.
type Value struct {
	Kind Kind                  `json:"k"`
	I64  int64                 `json:"i,omitempty"`
	F64  float64               `json:"f,omitempty"`
	s    unique.Handle[string] `json:"-"` // Private interned string
	B    bool                  `json:"b,omitempty"`
	A    []Value               `json:"a,omitempty"`
}

// StringValue returns the string value if Kind is KindString, otherwise empty string.
func (v Value) StringValue() string {
	if v.Kind == KindString {
		return v.s.Value()
	}
	return ""
}

// MarshalJSON implements json.Marshaler.
func (v Value) MarshalJSON() ([]byte, error) {
	type Alias Value
	aux := &struct {
		S string `json:"s,omitempty"`
		*Alias
	}{
		Alias: (*Alias)(&v),
	}
	if v.Kind == KindString {
		aux.S = v.s.Value()
	}
	return json.Marshal(aux)
}

// UnmarshalJSON implements json.Unmarshaler.
func (v *Value) UnmarshalJSON(data []byte) error {
	type Alias Value
	aux := &struct {
		S string `json:"s,omitempty"`
		*Alias
	}{
		Alias: (*Alias)(v),
	}
	if err := json.Unmarshal(data, &aux); err != nil {
		return err
	}
	if v.Kind == KindString {
		v.s = unique.Make(aux.S)
	}
	return nil
}

// Key returns a stable string representation for use in maps.
//
// It is intended for internal indexing (inverted indexes) and must remain stable
// across versions for persisted metadata usage.
func (v Value) Key() string {
	switch v.Kind {
	case KindNull:
		return "null"
	case KindInt:
		return "i:" + strconv.FormatInt(v.I64, 10)
	case KindFloat:
		return "f:" + strconv.FormatUint(math.Float64bits(v.F64), 16)
	case KindString:
		return "s:" + v.s.Value()
	case KindBool:
		if v.B {
			return "b:1"
		}
		return "b:0"
	case KindArray:
		if len(v.A) == 0 {
			return "a:"
		}
		parts := make([]string, len(v.A))
		for i := range v.A {
			parts[i] = v.A[i].Key()
		}
		return "a:" + strings.Join(parts, "\x1f")
	default:
		return "invalid"
	}
}

// AsInt64 returns the int64 value if Kind is KindInt.
func (v Value) AsInt64() (int64, bool) {
	if v.Kind != KindInt {
		return 0, false
	}
	return v.I64, true
}

// AsFloat64 returns the float64 value if Kind is KindFloat.
func (v Value) AsFloat64() (float64, bool) {
	if v.Kind != KindFloat {
		return 0, false
	}
	return v.F64, true
}

// AsString returns the string value if Kind is KindString.
func (v Value) AsString() (string, bool) {
	if v.Kind != KindString {
		return "", false
	}
	return v.s.Value(), true
}

// AsBool returns the boolean value if Kind is KindBool.
func (v Value) AsBool() (bool, bool) {
	if v.Kind != KindBool {
		return false, false
	}
	return v.B, true
}

// AsArray returns the array value if Kind is KindArray.
func (v Value) AsArray() ([]Value, bool) {
	if v.Kind != KindArray {
		return nil, false
	}
	return v.A, true
}

// Null returns a null Value.
func Null() Value { return Value{Kind: KindNull} }

// Int returns an int64 Value.
func Int(v int64) Value { return Value{Kind: KindInt, I64: v} }

// Float returns a float64 Value.
func Float(v float64) Value { return Value{Kind: KindFloat, F64: v} }

// String returns a string Value.
func String(v string) Value { return Value{Kind: KindString, s: unique.Make(v)} }

// Bool returns a boolean Value.
func Bool(v bool) Value { return Value{Kind: KindBool, B: v} }

// Array returns an array Value.
func Array(v []Value) Value { return Value{Kind: KindArray, A: v} }

// Document is a typed metadata document.
type Document map[string]Value

// InternedDocument is the internal representation of a document using interned keys.
// It is used by the engine for memory efficiency.
type InternedDocument map[unique.Handle[string]]Value

// Clone creates a deep copy of the metadata document.
//
// This is the safe default to prevent external mutation after Insert().
// Values are deep copied, including arrays, ensuring the clone is completely
// independent from the original.
//
// Performance: Typically <1% overhead since metadata is small (2-10 fields).
func (d Document) Clone() Document {
	if d == nil {
		return nil
	}

	clone := make(Document, len(d))
	for k, v := range d {
		clone[k] = v.clone()
	}
	return clone
}

// clone creates a deep copy of a Value, including nested arrays.
func (v Value) clone() Value {
	if v.Kind != KindArray || len(v.A) == 0 {
		// Simple values are copied by value semantics
		return v
	}

	// Deep copy array
	arrayCopy := make([]Value, len(v.A))
	for i := range v.A {
		arrayCopy[i] = v.A[i].clone()
	}

	return Value{
		Kind: v.Kind,
		I64:  v.I64,
		F64:  v.F64,
		s:    v.s,
		B:    v.B,
		A:    arrayCopy,
	}
}

// CloneIfNeeded clones metadata only if it's non-nil and non-empty.
//
// This helper avoids allocation for empty metadata, which is common.
// Returns nil if the input is nil or empty.
func CloneIfNeeded(m Metadata) Metadata {
	if len(m) == 0 {
		return nil
	}
	return m.Clone()
}

// Metadata is the default metadata document type used by vecgo.
//
// It is intentionally a typed model (map[string]Value) to keep filtering fast.
// If you need to ingest legacy map[string]any data, use the adapter helpers in
// this package.
type Metadata = Document

// Operator represents a comparison operator for filtering.
type Operator string

const (
	// OpEqual represents the equality operator.
	OpEqual Operator = "eq" // Equal
	// OpNotEqual represents the inequality operator.
	OpNotEqual Operator = "ne"
	// OpGreaterThan represents the greater than operator.
	OpGreaterThan Operator = "gt"
	// OpGreaterEqual represents the greater than or equal operator.
	OpGreaterEqual Operator = "gte"
	// OpLessThan represents the less than operator.
	OpLessThan Operator = "lt"
	// OpLessEqual represents the less than or equal operator.
	OpLessEqual Operator = "lte"
	// OpIn represents the in list operator.
	OpIn Operator = "in"
	// OpContains represents the contains substring operator.
	OpContains Operator = "contains"
)

// Filter represents a single metadata filter condition.
type Filter struct {
	Key      string
	Operator Operator
	Value    Value
}

// FilterSet represents a set of filters that must all match (AND logic).
type FilterSet struct {
	Filters []Filter
}

// NewFilterSet creates a new filter set.
func NewFilterSet(filters ...Filter) *FilterSet {
	return &FilterSet{Filters: filters}
}
