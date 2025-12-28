package metadata

import (
	"math"
	"strconv"
	"strings"
)

// Kind identifies the concrete type stored in a Value.
type Kind uint8

const (
	KindInvalid Kind = iota
	KindNull
	KindInt
	KindFloat
	KindString
	KindBool
	KindArray
)

// Value is a small typed value used for metadata documents and filters.
//
// The representation is designed to make filtering fast and predictable:
// no reflection and no fmt-based stringification.
//
// NOTE: This is also used for persistence; keep it stable.
type Value struct {
	Kind Kind    `json:"k"`
	I64  int64   `json:"i,omitempty"`
	F64  float64 `json:"f,omitempty"`
	S    string  `json:"s,omitempty"`
	B    bool    `json:"b,omitempty"`
	A    []Value `json:"a,omitempty"`
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
		return "s:" + v.S
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

func (v Value) AsInt64() (int64, bool) {
	if v.Kind != KindInt {
		return 0, false
	}
	return v.I64, true
}

func (v Value) AsFloat64() (float64, bool) {
	if v.Kind != KindFloat {
		return 0, false
	}
	return v.F64, true
}

func (v Value) AsString() (string, bool) {
	if v.Kind != KindString {
		return "", false
	}
	return v.S, true
}

func (v Value) AsBool() (bool, bool) {
	if v.Kind != KindBool {
		return false, false
	}
	return v.B, true
}

func (v Value) AsArray() ([]Value, bool) {
	if v.Kind != KindArray {
		return nil, false
	}
	return v.A, true
}

func Null() Value           { return Value{Kind: KindNull} }
func Int(v int64) Value     { return Value{Kind: KindInt, I64: v} }
func Float(v float64) Value { return Value{Kind: KindFloat, F64: v} }
func String(v string) Value { return Value{Kind: KindString, S: v} }
func Bool(v bool) Value     { return Value{Kind: KindBool, B: v} }
func Array(v []Value) Value { return Value{Kind: KindArray, A: v} }

// Document is a typed metadata document.
type Document map[string]Value

// Metadata is the default metadata document type used by vecgo.
//
// It is intentionally a typed model (map[string]Value) to keep filtering fast.
// If you need to ingest legacy map[string]any data, use the adapter helpers in
// this package.
type Metadata = Document

// Operator represents a comparison operator for filtering.
type Operator string

const (
	OpEqual        Operator = "eq"       // Equal
	OpNotEqual     Operator = "ne"       // Not equal
	OpGreaterThan  Operator = "gt"       // Greater than
	OpGreaterEqual Operator = "gte"      // Greater than or equal
	OpLessThan     Operator = "lt"       // Less than
	OpLessEqual    Operator = "lte"      // Less than or equal
	OpIn           Operator = "in"       // In list
	OpContains     Operator = "contains" // Contains substring (for strings)
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
