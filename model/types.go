package model

import (
	"fmt"

	"github.com/hupe1980/vecgo/metadata"
)

// ID is the globally unique, auto-incrementing primary key.
type ID uint64

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

// Candidate represents a search result candidate.
type Candidate struct {
	ID     ID
	Loc    Location
	Score  float32
	Approx bool

	// Materialized data (optional)
	Vector   []float32
	Metadata metadata.Document
	Payload  []byte
}

// Record represents a full data record.
type Record struct {
	ID       ID
	Vector   []float32
	Metadata metadata.Document
	Payload  []byte
}

// RecordBuilder provides a fluent API for constructing records.
// This enables type-safe, readable record creation:
//
//	rec := model.NewRecord(vec).
//	    WithMetadata("category", metadata.String("tech")).
//	    WithPayload(jsonBytes).
//	    Build()
type RecordBuilder struct {
	rec Record
}

// NewRecord creates a new RecordBuilder with the given vector.
func NewRecord(vector []float32) *RecordBuilder {
	return &RecordBuilder{
		rec: Record{
			Vector:   vector,
			Metadata: make(metadata.Document),
		},
	}
}

// WithMetadata adds a metadata field to the record.
// Uses type-safe metadata.Value for compile-time safety.
func (b *RecordBuilder) WithMetadata(key string, value metadata.Value) *RecordBuilder {
	b.rec.Metadata[key] = value
	return b
}

// WithPayload sets the payload (arbitrary bytes, typically JSON).
func (b *RecordBuilder) WithPayload(payload []byte) *RecordBuilder {
	b.rec.Payload = payload
	return b
}

// Build returns the constructed Record.
func (b *RecordBuilder) Build() Record {
	return b.rec
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
