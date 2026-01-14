package segment

import (
	"context"

	"github.com/hupe1980/vecgo/distance"
	"github.com/hupe1980/vecgo/internal/searcher"
	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/model"
)

// AccessPattern hints for memory access.
type AccessPattern int

const (
	AccessDefault AccessPattern = iota
	AccessSequential
	AccessRandom
	AccessWillNeed
	AccessDontNeed
)

// RecordBatch represents a batch of columnar data fetched from a segment.
type RecordBatch interface {
	RowCount() int
	ID(i int) model.ID
	Vector(i int) []float32
	Metadata(i int) metadata.Document
	Payload(i int) []byte
}

// SimpleRecordBatch is a basic implementation of RecordBatch.
type SimpleRecordBatch struct {
	IDs       []model.ID
	Vectors   [][]float32
	Metadatas []metadata.Document
	Payloads  [][]byte
}

func (b *SimpleRecordBatch) RowCount() int {
	return len(b.IDs)
}

func (b *SimpleRecordBatch) ID(i int) model.ID {
	return b.IDs[i]
}

func (b *SimpleRecordBatch) Vector(i int) []float32 {
	if b.Vectors == nil {
		return nil
	}
	return b.Vectors[i]
}

func (b *SimpleRecordBatch) Metadata(i int) metadata.Document {
	if b.Metadatas == nil {
		return nil
	}
	return b.Metadatas[i]
}

func (b *SimpleRecordBatch) Payload(i int) []byte {
	if b.Payloads == nil {
		return nil
	}
	return b.Payloads[i]
}

// Segment is the interface for an immutable data segment.
type Segment interface {
	ID() model.SegmentID
	RowCount() uint32
	Metric() distance.Metric

	// Search returns candidate locations (SegmentID, RowID) and approximate scores.
	// Filter is pushed down to the segment (e.g. for pre-filtering).
	// s is a reusable search context (optional, may be nil).
	// Results are added to s.Heap.
	Search(ctx context.Context, q []float32, k int, filter Filter, opts model.SearchOptions, s *searcher.Searcher) error

	// GetID returns the external ID for a given internal row ID.
	GetID(rowID uint32) (model.ID, bool)

	// Rerank computes exact distances for a candidate set.
	// Implementations MUST optimize for zero-copy access (e.g. unsafe.Pointer to mmap).
	// Results are appended to dst.
	Rerank(ctx context.Context, q []float32, cands []model.Candidate, dst []model.Candidate) ([]model.Candidate, error)

	// Fetch resolves RowIDs to payload columns.
	Fetch(ctx context.Context, rows []uint32, cols []string) (RecordBatch, error)

	// FetchIDs resolves RowIDs to IDs.
	// Results are written to dst.
	FetchIDs(ctx context.Context, rows []uint32, dst []model.ID) error

	// Iterate iterates over all vectors in the segment.
	// The context is used for cancellation during long iterations.
	Iterate(ctx context.Context, fn func(rowID uint32, id model.ID, vec []float32, md metadata.Document, payload []byte) error) error

	// EvaluateFilter returns a bitmap of rows matching the filter.
	// Returns (nil, nil) if the filter matches all rows (empty filter).
	// Returns error if the filter cannot be evaluated solely by the segment (e.g. missing index).
	EvaluateFilter(ctx context.Context, filter *metadata.FilterSet) (Bitmap, error)

	// Advise hints the kernel about access patterns.
	Advise(pattern AccessPattern) error

	// Size returns the size of the segment in bytes.
	Size() int64

	// Close releases resources associated with the segment.
	Close() error
}

// Bitmap is an abstract, segment-local set of RowIDs.
// It is intentionally minimal: implementations may wrap roaring-like bitmaps,
// simple bitsets, or posting lists.
type Bitmap interface {
	// Contains reports whether id is present in the set.
	Contains(id uint32) bool

	// Cardinality returns the number of elements in the set.
	Cardinality() uint64

	// ForEach calls fn for each id in the set (ascending order if available).
	// Stop early if fn returns false.
	ForEach(fn func(id uint32) bool)
}

// FieldStats stores min/max values for a numeric field in a block.
type FieldStats struct {
	Min float64 `json:"min"`
	Max float64 `json:"max"`
}

// Filter abstracts bitmap or predicate logic.
type Filter interface {
	// Matches checks if a RowID passes the filter.
	Matches(id uint32) bool

	// MatchesBatch checks a batch of RowIDs, writing results to out.
	// Contract: len(out) must be >= len(ids); only the first len(ids) entries are written.
	// Implementations should use vectorized comparisons where possible.
	MatchesBatch(ids []uint32, out []bool)

	// AsBitmap returns a bitmap view if the filter can be represented that way.
	// Implementations may still expose richer types via type assertions.
	AsBitmap() Bitmap

	// MatchesBlock checks if the filter might match any row in a block with the given stats.
	MatchesBlock(stats map[string]FieldStats) bool
}
