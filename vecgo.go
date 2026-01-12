// Package vecgo provides a high-performance embedded vector database for Go.
//
// Vecgo is an embeddable, hybrid vector database designed for production workloads.
// It combines LSM-tree architecture with HNSW indexing for optimal performance.
//
// # Quick Start
//
// Local mode (path string IS the index):
//
//	// Create new index
//	db, _ := vecgo.Open("./data", vecgo.Create(128, vecgo.MetricL2))
//
//	// Re-open existing index (dim/metric loaded from manifest)
//	db, _ := vecgo.Open("./data")
//
// Cloud mode (BlobStore IS the index):
//
//	// Create S3 store
//	s3Store, _ := s3.New(ctx, "my-bucket", s3.WithPrefix("vectors/"))
//
//	// Open from S3 (cache auto-created in temp dir)
//	db, _ := vecgo.Open(s3Store)
//
//	// Or with explicit cache directory
//	db, _ := vecgo.Open(s3Store, vecgo.WithCacheDir("/fast/nvme"))
package vecgo

import (
	"fmt"
	"log/slog"

	"github.com/hupe1980/vecgo/blobstore"
	"github.com/hupe1980/vecgo/distance"
	"github.com/hupe1980/vecgo/internal/engine"
	"github.com/hupe1980/vecgo/internal/quantization"
	"github.com/hupe1980/vecgo/lexical"
	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/model"
)

// DB is the main entry point for the vector database.
// It wraps the internal engine and provides a high-level API.
type DB struct {
	*engine.Engine
}

// Option configures the engine.
type Option = engine.Option

// ID is the unique identifier for a vector.
type ID = model.ID

// Candidate represents a search result.
type Candidate = model.Candidate

// Record represents a single item to be inserted.
type Record = model.Record

// Metric defines the distance comparison type.
type Metric = distance.Metric

// QuantizationType defines the vector quantization method.
type QuantizationType = quantization.Type

// Vecgo is purely commit-oriented. No WAL required.

// Public error taxonomy (re-exported from engine).
var (
	ErrClosed             = engine.ErrClosed
	ErrInvalidArgument    = engine.ErrInvalidArgument
	ErrCorrupt            = engine.ErrCorrupt
	ErrIncompatibleFormat = engine.ErrIncompatibleFormat
	ErrBackpressure       = engine.ErrBackpressure
	ErrReadOnly           = engine.ErrReadOnly
)

// Re-export Option constructors
var (
	WithDiskANNThreshold    = engine.WithDiskANNThreshold
	WithCompactionThreshold = engine.WithCompactionThreshold
	WithQuantization        = engine.WithQuantization
)

const (
	MetricL2     = distance.MetricL2
	MetricCosine = distance.MetricCosine
	MetricDot    = distance.MetricDot

	QuantizationTypeNone   = quantization.TypeNone
	QuantizationTypePQ     = quantization.TypePQ
	QuantizationTypeOPQ    = quantization.TypeOPQ
	QuantizationTypeSQ8    = quantization.TypeSQ8
	QuantizationTypeBQ     = quantization.TypeBQ
	QuantizationTypeRaBitQ = quantization.TypeRaBitQ
	QuantizationTypeINT4   = quantization.TypeINT4
)

// Source represents the location of a vector index.
// It can be either a local directory path (string) or a remote BlobStore.
type Source interface {
	isSource()
}

// LocalPath is a local filesystem path to the index.
type LocalPath string

func (LocalPath) isSource() {}

// RemoteStore wraps a BlobStore as an index source.
type RemoteStore struct {
	Store blobstore.BlobStore
}

func (RemoteStore) isSource() {}

// Create returns an Option that specifies creating a new index with the given dimension and metric.
// Use this when creating a new index for the first time.
// If omitted, vecgo will attempt to open an existing index and read dim/metric from the manifest.
func Create(dim int, metric Metric) Option {
	return func(e *engine.Engine) {
		engine.WithDimension(dim)(e)
		engine.WithMetric(metric)(e)
	}
}

// WithCacheDir sets the local directory for caching remote data.
// Only applicable when opening a remote store. Defaults to os.TempDir()/vecgo-cache-<random>.
func WithCacheDir(dir string) Option {
	return engine.WithCacheDir(dir)
}

// ReadOnly puts the engine in read-only mode.
// In this mode:
//   - No WAL is created or used
//   - Insert/Delete operations return ErrReadOnly
//   - No local state is required (pure memory cache)
//   - Ideal for stateless serverless search nodes
//
// This follows LanceDB's model for cloud deployments where the remote
// store (S3) is the source of truth and search nodes are stateless.
//
// Example:
//
//	// Stateless search node - boots from S3, no local storage needed
//	db, _ := vecgo.Open(s3Store, vecgo.ReadOnly())
func ReadOnly() Option {
	return engine.ReadOnly()
}

// Open opens or creates a vector database.
//
// The source parameter determines where the index data lives:
//   - string: Local filesystem path (e.g., "./data")
//   - blobstore.BlobStore: Remote store (e.g., S3, GCS)
//
// For new indexes, use the Create option:
//
//	db, _ := vecgo.Open("./data", vecgo.Create(128, vecgo.MetricL2))
//
// For existing indexes, dimension and metric are loaded from the manifest:
//
//	db, _ := vecgo.Open("./data")
//
// For remote stores, a local cache directory is auto-created:
//
//	db, _ := vecgo.Open(s3Store)
//	db, _ := vecgo.Open(s3Store, vecgo.WithCacheDir("/fast/nvme"))
func Open(source interface{}, opts ...Option) (*DB, error) {
	var e *engine.Engine
	var err error

	switch s := source.(type) {
	case string:
		// Local path - use existing Open logic
		e, err = engine.OpenLocal(s, opts...)
	case blobstore.BlobStore:
		// Remote store - use cloud logic with auto cache dir
		e, err = engine.OpenRemote(s, opts...)
	case LocalPath:
		e, err = engine.OpenLocal(string(s), opts...)
	case RemoteStore:
		e, err = engine.OpenRemote(s.Store, opts...)
	default:
		return nil, fmt.Errorf("vecgo.Open: unsupported source type %T, expected string or blobstore.BlobStore", source)
	}

	if err != nil {
		return nil, err
	}
	return &DB{Engine: e}, nil
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

// WithPayload requests the payload to be returned in the search results.
func WithPayload() SearchOption {
	return engine.WithPayload()
}

// WithVector requests the vector to be returned in the search results.
func WithVector() SearchOption {
	return engine.WithVector()
}

// MetricsObserver interfaces for observing engine metrics.
type MetricsObserver = engine.MetricsObserver

// CompactionConfig configures compaction.
type CompactionConfig = engine.CompactionConfig

// FlushConfig configures MemTable flushing.
type FlushConfig = engine.FlushConfig

// CompactionPolicy determines when to compact segments.
type CompactionPolicy = engine.CompactionPolicy

// RetentionPolicy defines rules for retaining old versions.
type RetentionPolicy = engine.RetentionPolicy

// WithFlushConfig sets the flush configuration.
func WithFlushConfig(cfg FlushConfig) Option {
	return engine.WithFlushConfig(cfg)
}

// WithCompactionConfig sets the compaction configuration.
func WithCompactionConfig(cfg CompactionConfig) Option {
	return engine.WithCompactionConfig(cfg)
}

// WithLexicalIndex sets the lexical index and the metadata field to index.
func WithLexicalIndex(idx lexical.Index, field string) Option {
	return engine.WithLexicalIndex(idx, field)
}

// WithCompactionPolicy sets the compaction policy.
func WithCompactionPolicy(policy CompactionPolicy) Option {
	return engine.WithCompactionPolicy(policy)
}

// WithMetricsObserver sets the metrics observer for the engine.
func WithMetricsObserver(observer MetricsObserver) Option {
	return engine.WithMetricsObserver(observer)
}

// WithBlockCacheSize sets the size of the block cache in bytes.
func WithBlockCacheSize(size int64) Option {
	return engine.WithBlockCacheSize(size)
}

// WithMemoryLimit sets the memory limit for the engine in bytes.
// If set to 0, memory is unlimited (no backpressure).
// Default is 1GB.
func WithMemoryLimit(bytes int64) Option {
	return engine.WithMemoryLimit(bytes)
}

// WithNProbes sets the number of probes for the search.
func WithNProbes(n int) SearchOption {
	return engine.WithNProbes(n)
}

// WithDimension sets the vector dimension.
func WithDimension(dim int) Option {
	return engine.WithDimension(dim)
}

// WithMetric sets the distance metric.
func WithMetric(m Metric) Option {
	return engine.WithMetric(m)
}
