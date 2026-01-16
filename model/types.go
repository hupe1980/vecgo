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

	// Stats receives detailed query execution statistics when non-nil.
	// Use this for query debugging, performance analysis, and cost estimation.
	Stats *QueryStats
}

// QueryStats provides detailed execution statistics for a search query.
// This enables query explainability: "Why did this query take 312μs?"
// All fields are populated after the query completes.
type QueryStats struct {
	// TotalDurationMicros is the total query execution time in microseconds.
	TotalDurationMicros int64

	// SegmentsSearched is the number of segments searched.
	SegmentsSearched int

	// SegmentStats contains per-segment execution details.
	SegmentStats []SegmentQueryStats

	// DistanceComputations is the total number of distance calculations performed.
	DistanceComputations int64

	// DistanceShortCircuits is the number of distance calculations that exited early
	// due to distance short-circuiting (partial sum exceeded bound).
	DistanceShortCircuits int64

	// NodesVisited is the total number of HNSW nodes visited (for HNSW segments).
	NodesVisited int64

	// FilterPassRate is the percentage of candidates that passed the filter (0.0-1.0).
	// Only populated when a filter is applied.
	FilterPassRate float64

	// CandidatesEvaluated is the total number of candidates evaluated across all segments.
	CandidatesEvaluated int64

	// CandidatesReturned is the number of candidates returned (≤ K).
	CandidatesReturned int

	// Strategy describes the search strategy used (e.g., "hnsw", "brute_force", "hybrid").
	Strategy string
}

// SegmentQueryStats contains execution statistics for a single segment.
type SegmentQueryStats struct {
	// SegmentID is the unique identifier of the segment.
	SegmentID SegmentID

	// IndexType is the index type (e.g., "hnsw", "diskann", "flat").
	IndexType string

	// DurationMicros is the time spent searching this segment in microseconds.
	DurationMicros int64

	// DistanceComputations is the number of distance calculations in this segment.
	DistanceComputations int64

	// NodesVisited is the number of nodes visited (for graph indexes).
	NodesVisited int64

	// CandidatesFound is the number of candidates found from this segment.
	CandidatesFound int
}

// EstimatedCost returns an estimated cost score for the query.
// Higher values indicate more expensive queries.
// This can be used for query planning and resource allocation.
func (s *QueryStats) EstimatedCost() float64 {
	if s == nil {
		return 0
	}
	// Cost model: dominated by distance computations (O(d) SIMD work)
	// Short-circuits count as 0.3x (partial work) instead of full
	fullComputations := float64(s.DistanceComputations - s.DistanceShortCircuits)
	partialComputations := float64(s.DistanceShortCircuits) * 0.3
	return fullComputations + partialComputations
}

// Explain returns a human-readable explanation of the query execution.
func (s *QueryStats) Explain() string {
	if s == nil {
		return "no stats collected"
	}
	return fmt.Sprintf(
		"Query took %dμs: searched %d segments, %d distance calcs (%d short-circuited), "+
			"%d nodes visited, %d candidates evaluated → %d returned. Strategy: %s",
		s.TotalDurationMicros,
		s.SegmentsSearched,
		s.DistanceComputations,
		s.DistanceShortCircuits,
		s.NodesVisited,
		s.CandidatesEvaluated,
		s.CandidatesReturned,
		s.Strategy,
	)
}
