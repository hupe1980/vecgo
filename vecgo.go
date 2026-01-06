// Package vecgo provides a high-performance embedded vector database for Go.
package vecgo

import (
	"log/slog"

	"github.com/hupe1980/vecgo/blobstore"
	"github.com/hupe1980/vecgo/distance"
	"github.com/hupe1980/vecgo/engine"
	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/model"
)

// Engine is the main entry point for the vector database.
type Engine = engine.Engine

// Option configures the engine.
type Option = engine.Option

// PrimaryKey is the unique identifier for a vector.
type PrimaryKey = model.PrimaryKey

// PKUint64 creates a new uint64-based Primary Key.
func PKUint64(v uint64) PrimaryKey {
	return model.PKUint64(v)
}

// PKString creates a new string-based Primary Key.
func PKString(v string) PrimaryKey {
	return model.PKString(v)
}

// Candidate represents a search result.
type Candidate = model.Candidate

// Record represents a single item to be inserted.
type Record = model.Record

// Metric defines the distance metric used for vector comparison.
type Metric = distance.Metric

// Durability controls the durability guarantees of the WAL.
type Durability = engine.Durability

const (
	DurabilityAsync = engine.DurabilityAsync
	DurabilitySync  = engine.DurabilitySync
)

// WALOptions configures the write-ahead log.
type WALOptions = engine.WALOptions

func DefaultWALOptions() WALOptions {
	return engine.DefaultWALOptions()
}

// Public error taxonomy (re-exported from engine).
var (
	ErrClosed             = engine.ErrClosed
	ErrInvalidArgument    = engine.ErrInvalidArgument
	ErrCorrupt            = engine.ErrCorrupt
	ErrIncompatibleFormat = engine.ErrIncompatibleFormat
	ErrBackpressure       = engine.ErrBackpressure
)

const (
	MetricL2     = distance.MetricL2
	MetricCosine = distance.MetricCosine
	MetricDot    = distance.MetricDot
)

// Open opens or creates a new vector database.
func Open(dir string, dim int, metric Metric, opts ...Option) (*Engine, error) {
	return engine.Open(dir, dim, metric, opts...)
}

// WithBlobStore overrides the BlobStore used for reading immutable blobs (segments/payloads).
//
// Note: the engine still uses the local filesystem rooted at dir for manifest/WAL/PK index.
func WithBlobStore(st blobstore.BlobStore) Option {
	return engine.WithBlobStore(st)
}

// WithWALOptions sets the WAL configuration.
func WithWALOptions(opts WALOptions) Option {
	return engine.WithWALOptions(opts)
}

// WithLogger sets the logger for the engine.
func WithLogger(l *slog.Logger) Option {
	return engine.WithLogger(l)
}

// WithSchema sets the metadata schema for the engine.
func WithSchema(schema metadata.Schema) Option {
	return engine.WithSchema(schema)
}

// SearchOption configures a search query.
type SearchOption = func(*model.SearchOptions)

// WithRefineFactor sets the refinement factor for reranking.
func WithRefineFactor(factor float32) SearchOption {
	return func(o *model.SearchOptions) {
		o.RefineFactor = factor
	}
}

// WithFilter sets a typed metadata filter for the search.
func WithFilter(filter *metadata.FilterSet) SearchOption {
	return engine.WithFilter(filter)
}

// WithPreFilter forces pre-filtering (or post-filtering if false).
func WithPreFilter(preFilter bool) SearchOption {
	return engine.WithPreFilter(preFilter)
}

// WithMetadata requests the metadata to be returned in the search results.
func WithMetadata() SearchOption {
	return engine.WithMetadata()
}
