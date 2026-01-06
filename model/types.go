package model

import (
	"fmt"

	"github.com/hupe1980/vecgo/metadata"
)

// SegmentID is the unique identifier for a segment within a shard/engine.
type SegmentID uint64

// RowID is a dense, segment-local identifier for a record.
// It is transient and may change during compaction.
type RowID uint32

// Location identifies a specific version of a record in the engine.
type Location struct {
	SegmentID SegmentID
	RowID     RowID
}

// String returns a string representation of the Location.
func (l Location) String() string {
	return fmt.Sprintf("Loc(%d:%d)", l.SegmentID, l.RowID)
}

// PKKind distinguishes between different primary key types.
type PKKind uint8

const (
	PKKindUint64 PKKind = 0
	PKKindString PKKind = 1
)

// PK is the user-facing stable identifier.
// It is a tagged union supporting uint64 and string.
// It is comparable and can be used as a map key.
type PK struct {
	kind PKKind
	u64  uint64
	s    string
}

// PKUint64 creates a new uint64-based Primary Key.
func PKUint64(v uint64) PK {
	return PK{kind: PKKindUint64, u64: v}
}

// PKString creates a new string-based Primary Key.
func PKString(v string) PK {
	return PK{kind: PKKindString, s: v}
}

// Kind returns the type of the primary key.
func (pk PK) Kind() PKKind {
	return pk.kind
}

// Uint64 returns the uint64 value and true if the PK is a uint64.
func (pk PK) Uint64() (uint64, bool) {
	if pk.kind == PKKindUint64 {
		return pk.u64, true
	}
	return 0, false
}

// String returns the string value and true if the PK is a string.
// Note: This is different from the String() method which returns a debug representation.
func (pk PK) StringValue() (string, bool) {
	if pk.kind == PKKindString {
		return pk.s, true
	}
	return "", false
}

// String returns a string representation for debugging/logging.
func (pk PK) String() string {
	if pk.kind == PKKindString {
		return pk.s
	}
	return fmt.Sprintf("%d", pk.u64)
}

// PrimaryKey is an alias for PK to ease migration (optional, but helpful).
// We will use PK in new code.
type PrimaryKey = PK

// Record represents a full data record.
type Record struct {
	PK       PK
	Vector   []float32
	Metadata map[string]interface{}
	Payload  []byte
}

// Candidate represents a potential match found during search.
type Candidate struct {
	// PK is the user-facing primary key.
	// It may be zero/empty if not yet resolved from the segment.
	PK PK
	// Loc is the internal location of the match.
	Loc Location
	// Score is the distance/similarity score (metric-dependent).
	Score float32
	// Approx indicates if the score is approximate (e.g. from quantized data).
	Approx bool

	// Materialized data (optional)
	Vector   []float32
	Metadata map[string]interface{}
	Payload  []byte
}

// SearchOptions controls the execution of a search query.
type SearchOptions struct {
	// K is the number of nearest neighbors to return.
	K int

	// RefineFactor is the multiplier for the candidate pool size before reranking.
	// e.g., if K=10 and RefineFactor=2.0, we search for 20 candidates and rerank.
	RefineFactor float32

	// PreFilter indicates if the filter should be applied before vector search.
	// If nil, the engine decides based on heuristics.
	PreFilter *bool

	// Filter is the metadata filter to apply.
	Filter *metadata.FilterSet

	// NProbes is the number of partitions to probe (for IVF indexes).
	// If 0, a default is used.
	NProbes int

	// Materialization options
	IncludeVector   bool
	IncludeMetadata bool
	IncludePayload  bool
}
