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

// MatchesValue checks if the provided value matches this filter.
func (f *Filter) MatchesValue(value Value) bool {
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
// keyHandle is the pre-interned key for efficient lookup.
func (f *Filter) matchesInternedWithHandle(doc InternedDocument, keyHandle unique.Handle[string]) bool {
	value, exists := doc[keyHandle]
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
//
// Deprecated: Use FilterSet.MatchesInterned for better performance with cached keys.
func (f *Filter) MatchesInterned(doc InternedDocument) bool {
	return f.matchesInternedWithHandle(doc, unique.Make(f.Key))
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
// Uses cached interned keys for efficient repeated matching.
func (fs *FilterSet) MatchesInterned(doc InternedDocument) bool {
	// Lazily initialize interned keys on first call
	if fs.internedKeys == nil && len(fs.Filters) > 0 {
		fs.internedKeys = make([]unique.Handle[string], len(fs.Filters))
		for i := range fs.Filters {
			fs.internedKeys[i] = unique.Make(fs.Filters[i].Key)
		}
	}

	for i := range fs.Filters {
		if !fs.Filters[i].matchesInternedWithHandle(doc, fs.internedKeys[i]) {
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
		return compareNumbers(a, b)
	}

	if a.Kind != b.Kind {
		return false
	}

	return compareSameKind(a, b)
}

func compareNumbers(a, b Value) bool {
	// Prefer exact int compare when possible.
	if a.Kind == KindInt && b.Kind == KindInt {
		return a.I64 == b.I64
	}
	return asFloat64(a) == asFloat64(b)
}

func compareSameKind(a, b Value) bool {
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
