package metadata

import (
	"strings"
	"unique"
)

// Matches checks if the provided metadata matches this filter.
func (f *Filter) Matches(doc Document) bool {
	value, exists := doc[f.Key]
	if !exists {
		return false
	}

	switch f.Operator {
	case OpEqual:
		return compareEqual(value, f.Value)
	case OpNotEqual:
		return !compareEqual(value, f.Value)
	case OpGreaterThan:
		return compareGreater(value, f.Value)
	case OpGreaterEqual:
		return compareGreater(value, f.Value) || compareEqual(value, f.Value)
	case OpLessThan:
		return compareLess(value, f.Value)
	case OpLessEqual:
		return compareLess(value, f.Value) || compareEqual(value, f.Value)
	case OpIn:
		return compareIn(value, f.Value)
	case OpContains:
		return compareContains(value, f.Value)
	default:
		return false
	}
}

// MatchesInterned checks if the provided interned metadata matches this filter.
func (f *Filter) MatchesInterned(doc InternedDocument) bool {
	value, exists := doc[unique.Make(f.Key)]
	if !exists {
		return false
	}

	switch f.Operator {
	case OpEqual:
		return compareEqual(value, f.Value)
	case OpNotEqual:
		return !compareEqual(value, f.Value)
	case OpGreaterThan:
		return compareGreater(value, f.Value)
	case OpGreaterEqual:
		return compareGreater(value, f.Value) || compareEqual(value, f.Value)
	case OpLessThan:
		return compareLess(value, f.Value)
	case OpLessEqual:
		return compareLess(value, f.Value) || compareEqual(value, f.Value)
	case OpIn:
		return compareIn(value, f.Value)
	case OpContains:
		return compareContains(value, f.Value)
	default:
		return false
	}
}

// Matches checks if the provided metadata matches all filters in the set.
func (fs *FilterSet) Matches(doc Document) bool {
	for _, filter := range fs.Filters {
		if !filter.Matches(doc) {
			return false
		}
	}
	return true
}

// MatchesInterned checks if the provided interned metadata matches all filters in the set.
func (fs *FilterSet) MatchesInterned(doc InternedDocument) bool {
	for _, filter := range fs.Filters {
		if !filter.MatchesInterned(doc) {
			return false
		}
	}
	return true
}

// compareEqual compares two values for equality.

func compareEqual(a, b Value) bool {
	if a.Kind == KindNull && b.Kind == KindNull {
		return true
	}
	if a.Kind == KindNull || b.Kind == KindNull {
		return false
	}

	if isNumber(a) && isNumber(b) {
		// Prefer exact int compare when possible.
		if a.Kind == KindInt && b.Kind == KindInt {
			return a.I64 == b.I64
		}
		return asFloat64(a) == asFloat64(b)
	}

	if a.Kind != b.Kind {
		return false
	}

	switch a.Kind {
	case KindString:
		return a.s == b.s
	case KindBool:
		return a.B == b.B
	case KindArray:
		if len(a.A) != len(b.A) {
			return false
		}
		for i := range a.A {
			if !compareEqual(a.A[i], b.A[i]) {
				return false
			}
		}
		return true
	default:
		return false
	}
}

func compareGreater(a, b Value) bool {
	if !isNumber(a) || !isNumber(b) {
		return false
	}
	return asFloat64(a) > asFloat64(b)
}

func compareLess(a, b Value) bool {
	if !isNumber(a) || !isNumber(b) {
		return false
	}
	return asFloat64(a) < asFloat64(b)
}

func compareIn(a, b Value) bool {
	if b.Kind != KindArray {
		return false
	}
	for _, item := range b.A {
		if compareEqual(a, item) {
			return true
		}
	}
	return false
}

func compareContains(a, b Value) bool {
	if a.Kind != KindString || b.Kind != KindString {
		return false
	}
	return strings.Contains(a.s.Value(), b.s.Value())
}

func isNumber(v Value) bool {
	return v.Kind == KindInt || v.Kind == KindFloat
}

func asFloat64(v Value) float64 {
	switch v.Kind {
	case KindInt:
		return float64(v.I64)
	case KindFloat:
		return v.F64
	default:
		return 0
	}
}
