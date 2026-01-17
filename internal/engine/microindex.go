// Package engine provides the core database engine.
package engine

import (
	"sort"
	"sync"

	"github.com/hupe1980/vecgo/metadata"
)

// MicroIndex is a lightweight per-segment secondary index for a single column.
// It avoids bitmap creation entirely for narrow filters by storing a sorted
// array of (value, rowID) pairs.
//
// Use cases:
//   - Highly selective equality filters (category = "premium")
//   - Range filters on sorted/temporal columns
//   - Low-cardinality columns where bitmap overhead is high
//
// Size: ~1-2KB per column per segment (configurable max entries).
// Lookup: O(log n) binary search, no allocations.
type MicroIndex struct {
	// Field name this index covers
	Field string

	// Sorted entries for fast lookup
	entries []microIndexEntry

	// Value type
	valueKind metadata.Kind

	// Stats
	distinctValues int
	totalRows      int
}

type microIndexEntry struct {
	// Value (only one field is set based on valueKind)
	intVal    int64
	floatVal  float64
	stringVal string

	// Row ID
	rowID uint32
}

// MicroIndexConfig configures microindex creation.
type MicroIndexConfig struct {
	// MaxEntries is the maximum entries per microindex.
	// Default: 256 (1-2KB per column)
	MaxEntries int

	// MinSelectivity is the minimum selectivity (0.0-1.0) to create a microindex.
	// Only create for highly selective columns.
	// Default: 0.1 (top 10% most selective)
	MinSelectivity float64

	// Enabled controls whether microindexes are created.
	// Default: true
	Enabled bool
}

// DefaultMicroIndexConfig returns the default configuration.
func DefaultMicroIndexConfig() MicroIndexConfig {
	return MicroIndexConfig{
		MaxEntries:     256,
		MinSelectivity: 0.1,
		Enabled:        true,
	}
}

// MicroIndexBuilder builds a microindex during segment flush.
type MicroIndexBuilder struct {
	field      string
	entries    []microIndexEntry
	valueKind  metadata.Kind
	maxEntries int
	overflow   bool // true if we exceeded maxEntries
}

// NewMicroIndexBuilder creates a new microindex builder.
func NewMicroIndexBuilder(field string, maxEntries int) *MicroIndexBuilder {
	if maxEntries <= 0 {
		maxEntries = 256
	}
	return &MicroIndexBuilder{
		field:      field,
		entries:    make([]microIndexEntry, 0, maxEntries),
		maxEntries: maxEntries,
	}
}

// Add adds an entry to the microindex.
// Returns false if the index has overflowed and should not be used.
func (b *MicroIndexBuilder) Add(rowID uint32, value metadata.Value) bool {
	if b.overflow {
		return false
	}

	if len(b.entries) == 0 {
		b.valueKind = value.Kind
	}

	if len(b.entries) >= b.maxEntries {
		b.overflow = true
		b.entries = nil // Free memory
		return false
	}

	entry := microIndexEntry{rowID: rowID}
	switch value.Kind {
	case metadata.KindInt:
		entry.intVal = value.I64
	case metadata.KindFloat:
		entry.floatVal = value.F64
	case metadata.KindString:
		entry.stringVal = value.StringValue()
	default:
		// Unsupported type, skip
		return true
	}

	b.entries = append(b.entries, entry)
	return true
}

// Build finalizes the microindex.
// Returns nil if the index overflowed or has no entries.
func (b *MicroIndexBuilder) Build() *MicroIndex {
	if b.overflow || len(b.entries) == 0 {
		return nil
	}

	// Sort by value for binary search
	switch b.valueKind {
	case metadata.KindInt:
		sort.Slice(b.entries, func(i, j int) bool {
			return b.entries[i].intVal < b.entries[j].intVal
		})
	case metadata.KindFloat:
		sort.Slice(b.entries, func(i, j int) bool {
			return b.entries[i].floatVal < b.entries[j].floatVal
		})
	case metadata.KindString:
		sort.Slice(b.entries, func(i, j int) bool {
			return b.entries[i].stringVal < b.entries[j].stringVal
		})
	}

	// Count distinct values
	distinct := 1
	for i := 1; i < len(b.entries); i++ {
		switch b.valueKind {
		case metadata.KindInt:
			if b.entries[i].intVal != b.entries[i-1].intVal {
				distinct++
			}
		case metadata.KindFloat:
			if b.entries[i].floatVal != b.entries[i-1].floatVal {
				distinct++
			}
		case metadata.KindString:
			if b.entries[i].stringVal != b.entries[i-1].stringVal {
				distinct++
			}
		}
	}

	return &MicroIndex{
		Field:          b.field,
		entries:        b.entries,
		valueKind:      b.valueKind,
		distinctValues: distinct,
		totalRows:      len(b.entries),
	}
}

// LookupEqual returns all rowIDs where value equals the given value.
// Zero allocations if result fits in provided buffer.
func (mi *MicroIndex) LookupEqual(value metadata.Value, buf []uint32) []uint32 {
	if mi == nil || len(mi.entries) == 0 || value.Kind != mi.valueKind {
		return buf[:0]
	}

	result := buf[:0]

	switch mi.valueKind {
	case metadata.KindInt:
		target := value.I64
		// Binary search for first occurrence
		idx := sort.Search(len(mi.entries), func(i int) bool {
			return mi.entries[i].intVal >= target
		})
		// Collect all matches
		for idx < len(mi.entries) && mi.entries[idx].intVal == target {
			result = append(result, mi.entries[idx].rowID)
			idx++
		}

	case metadata.KindFloat:
		target := value.F64
		idx := sort.Search(len(mi.entries), func(i int) bool {
			return mi.entries[i].floatVal >= target
		})
		for idx < len(mi.entries) && mi.entries[idx].floatVal == target {
			result = append(result, mi.entries[idx].rowID)
			idx++
		}

	case metadata.KindString:
		target := value.StringValue()
		idx := sort.Search(len(mi.entries), func(i int) bool {
			return mi.entries[i].stringVal >= target
		})
		for idx < len(mi.entries) && mi.entries[idx].stringVal == target {
			result = append(result, mi.entries[idx].rowID)
			idx++
		}
	}

	return result
}

// LookupRange returns all rowIDs where value is in [lo, hi].
// Zero allocations if result fits in provided buffer.
func (mi *MicroIndex) LookupRange(lo, hi metadata.Value, buf []uint32) []uint32 {
	if mi == nil || len(mi.entries) == 0 {
		return buf[:0]
	}

	result := buf[:0]

	switch mi.valueKind {
	case metadata.KindInt:
		if lo.Kind != metadata.KindInt || hi.Kind != metadata.KindInt {
			return result
		}
		loVal, hiVal := lo.I64, hi.I64
		// Binary search for first >= lo
		start := sort.Search(len(mi.entries), func(i int) bool {
			return mi.entries[i].intVal >= loVal
		})
		// Collect all in range
		for i := start; i < len(mi.entries) && mi.entries[i].intVal <= hiVal; i++ {
			result = append(result, mi.entries[i].rowID)
		}

	case metadata.KindFloat:
		if lo.Kind != metadata.KindFloat || hi.Kind != metadata.KindFloat {
			return result
		}
		loVal, hiVal := lo.F64, hi.F64
		start := sort.Search(len(mi.entries), func(i int) bool {
			return mi.entries[i].floatVal >= loVal
		})
		for i := start; i < len(mi.entries) && mi.entries[i].floatVal <= hiVal; i++ {
			result = append(result, mi.entries[i].rowID)
		}

	case metadata.KindString:
		if lo.Kind != metadata.KindString || hi.Kind != metadata.KindString {
			return result
		}
		loVal, hiVal := lo.StringValue(), hi.StringValue()
		start := sort.Search(len(mi.entries), func(i int) bool {
			return mi.entries[i].stringVal >= loVal
		})
		for i := start; i < len(mi.entries) && mi.entries[i].stringVal <= hiVal; i++ {
			result = append(result, mi.entries[i].rowID)
		}
	}

	return result
}

// Selectivity returns the estimated selectivity for an equality filter.
// Lower = more selective = fewer rows match.
func (mi *MicroIndex) Selectivity() float64 {
	if mi == nil || mi.totalRows == 0 || mi.distinctValues == 0 {
		return 1.0
	}
	// Assume uniform distribution
	return 1.0 / float64(mi.distinctValues)
}

// Size returns the approximate memory size in bytes.
func (mi *MicroIndex) Size() int {
	if mi == nil {
		return 0
	}
	// Base struct + entries
	size := 48                   // struct overhead
	size += len(mi.entries) * 32 // entry size (conservative)
	// Add string lengths
	for _, e := range mi.entries {
		size += len(e.stringVal)
	}
	return size
}

// MicroIndexStore manages microindexes for a segment.
type MicroIndexStore struct {
	mu      sync.RWMutex
	indexes map[string]*MicroIndex // field -> index
}

// NewMicroIndexStore creates a new microindex store.
func NewMicroIndexStore() *MicroIndexStore {
	return &MicroIndexStore{
		indexes: make(map[string]*MicroIndex),
	}
}

// Add adds a microindex for a field.
func (s *MicroIndexStore) Add(mi *MicroIndex) {
	if mi == nil {
		return
	}
	s.mu.Lock()
	s.indexes[mi.Field] = mi
	s.mu.Unlock()
}

// Get returns the microindex for a field, or nil if none exists.
func (s *MicroIndexStore) Get(field string) *MicroIndex {
	s.mu.RLock()
	mi := s.indexes[field]
	s.mu.RUnlock()
	return mi
}

// Fields returns all indexed field names.
func (s *MicroIndexStore) Fields() []string {
	s.mu.RLock()
	defer s.mu.RUnlock()

	fields := make([]string, 0, len(s.indexes))
	for f := range s.indexes {
		fields = append(fields, f)
	}
	return fields
}

// TotalSize returns the total memory size of all microindexes.
func (s *MicroIndexStore) TotalSize() int {
	s.mu.RLock()
	defer s.mu.RUnlock()

	total := 0
	for _, mi := range s.indexes {
		total += mi.Size()
	}
	return total
}

// TryLookup attempts to use microindexes for a filter.
// Returns rowIDs and true if a microindex was used, nil and false otherwise.
func (s *MicroIndexStore) TryLookup(f metadata.Filter, buf []uint32) ([]uint32, bool) {
	mi := s.Get(f.Key)
	if mi == nil {
		return nil, false
	}

	switch f.Operator {
	case metadata.OpEqual:
		return mi.LookupEqual(f.Value, buf), true

	case metadata.OpGreaterEqual, metadata.OpLessEqual:
		// For single-bound ranges, use full range
		if f.Operator == metadata.OpGreaterEqual {
			hi := maxValue(mi.valueKind)
			return mi.LookupRange(f.Value, hi, buf), true
		}
		lo := minValue(mi.valueKind)
		return mi.LookupRange(lo, f.Value, buf), true

	default:
		return nil, false
	}
}

// minValue returns the minimum value for a type.
func minValue(kind metadata.Kind) metadata.Value {
	switch kind {
	case metadata.KindInt:
		return metadata.Value{Kind: metadata.KindInt, I64: -1 << 62}
	case metadata.KindFloat:
		return metadata.Value{Kind: metadata.KindFloat, F64: -1e308}
	case metadata.KindString:
		return metadata.Value{Kind: metadata.KindString}
	default:
		return metadata.Value{}
	}
}

// maxValue returns the maximum value for a type.
func maxValue(kind metadata.Kind) metadata.Value {
	switch kind {
	case metadata.KindInt:
		return metadata.Value{Kind: metadata.KindInt, I64: 1 << 62}
	case metadata.KindFloat:
		return metadata.Value{Kind: metadata.KindFloat, F64: 1e308}
	case metadata.KindString:
		// Use a high Unicode value string
		return metadata.String("\xff\xff\xff\xff")
	default:
		return metadata.Value{}
	}
}
