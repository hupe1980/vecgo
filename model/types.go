package model

import (
	"fmt"
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

// PrimaryKey is the user-facing stable identifier.
// For vNext Phase 0, we start with uint64 as the primary key type.
// In the future, this could be an interface or a union type to support strings.
type PrimaryKey uint64

// Record represents a full data record.
type Record struct {
	PK       PrimaryKey
	Vector   []float32
	Metadata map[string]interface{}
	Payload  []byte
}

// Candidate represents a potential match found during search.
type Candidate struct {
	// PK is the user-facing primary key.
	// It may be zero/empty if not yet resolved from the segment.
	PK PrimaryKey
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
	// The type is interface{} to avoid a circular dependency on the metadata package.
	// It is expected to be *metadata.FilterSet.
	Filter interface{}

	// NProbes is the number of partitions to probe (for IVF indexes).
	// If 0, a default is used.
	NProbes int

	// Materialization options
	IncludeVector   bool
	IncludeMetadata bool
	IncludePayload  bool
}
