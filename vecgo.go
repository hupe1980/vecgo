// Package vecgo provides a high-performance embedded vector database for Go.
//
// Vecgo is an embeddable, hybrid vector database designed for production workloads.
// It combines LSM-tree architecture with HNSW indexing for optimal performance.
//
// # Quick Start
//
// Local mode:
//
//	db, _ := vecgo.Open(vecgo.Local("./data"), vecgo.Create(128, vecgo.MetricL2))
//	db, _ := vecgo.Open(vecgo.Local("./data"))  // re-open existing
//
// Cloud mode:
//
//	s3Store, _ := s3.New(ctx, "my-bucket", s3.WithPrefix("vectors/"))
//	db, _ := vecgo.Open(vecgo.Remote(s3Store))
//	db, _ := vecgo.Open(vecgo.Remote(s3Store), vecgo.WithCacheDir("/fast/nvme"))
//
// # Search with Data
//
// By default, search returns IDs, scores, metadata, and payload:
//
//	results, _ := db.Search(ctx, query, 10)
//	for _, r := range results {
//	    fmt.Println(r.ID, r.Score, r.Metadata, r.Payload)
//	}
//
// For minimal results (IDs + scores only), use WithoutData():
//
//	results, _ := db.Search(ctx, query, 10, vecgo.WithoutData())
package vecgo

import (
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

// Backend represents a storage backend for the vector database.
// Use Local() for filesystem storage or Remote() for cloud storage.
type Backend interface {
	open(opts ...Option) (*engine.Engine, error)
}

// localBackend is a local filesystem backend.
type localBackend string

func (b localBackend) open(opts ...Option) (*engine.Engine, error) {
	return engine.OpenLocal(string(b), opts...)
}

// remoteBackend is a remote BlobStore backend (S3, GCS, Azure, etc.).
type remoteBackend struct {
	store blobstore.BlobStore
}

func (b remoteBackend) open(opts ...Option) (*engine.Engine, error) {
	return engine.OpenRemote(b.store, opts...)
}

// Local creates a local filesystem backend.
//
// Example:
//
//	db, _ := vecgo.Open(vecgo.Local("./data"), vecgo.Create(128, vecgo.MetricL2))
func Local(path string) Backend {
	return localBackend(path)
}

// Remote creates a remote storage backend (S3, GCS, Azure, etc.).
//
// Example:
//
//	s3Store, _ := s3.New(ctx, "my-bucket")
//	db, _ := vecgo.Open(vecgo.Remote(s3Store), vecgo.WithCacheDir("/fast/nvme"))
func Remote(store blobstore.BlobStore) Backend {
	return remoteBackend{store: store}
}

// Open opens or creates a vector database with the given backend.
//
// For new indexes, use the Create option:
//
//	db, _ := vecgo.Open(vecgo.Local("./data"), vecgo.Create(128, vecgo.MetricL2))
//
// For existing indexes, dimension and metric are loaded from the manifest:
//
//	db, _ := vecgo.Open(vecgo.Local("./data"))
//
// For cloud storage:
//
//	db, _ := vecgo.Open(vecgo.Remote(s3Store))
func Open(backend Backend, opts ...Option) (*DB, error) {
	e, err := backend.open(opts...)
	if err != nil {
		return nil, err
	}
	return &DB{Engine: e}, nil
}

// InsertRecord inserts a record built with the fluent RecordBuilder API.
//
// Example:
//
//	rec := vecgo.NewRecord(vec).
//	    WithMetadata("category", metadata.String("tech")).
//	    WithPayload(jsonData).
//	    Build()
//	id, err := db.InsertRecord(rec)
func (db *DB) InsertRecord(rec Record) (ID, error) {
	return db.Engine.Insert(rec.Vector, rec.Metadata, rec.Payload)
}

// BatchInsertRecords inserts multiple records in a single batch.
//
// Example:
//
//	records := []vecgo.Record{rec1, rec2, rec3}
//	ids, err := db.BatchInsertRecords(records)
func (db *DB) BatchInsertRecords(records []Record) ([]ID, error) {
	vectors := make([][]float32, len(records))
	mds := make([]metadata.Document, len(records))
	payloads := make([][]byte, len(records))

	for i, rec := range records {
		vectors[i] = rec.Vector
		mds[i] = rec.Metadata
		payloads[i] = rec.Payload
	}

	return db.Engine.BatchInsert(vectors, mds, payloads)
}

// NewRecord creates a new RecordBuilder for fluent record construction.
var NewRecord = model.NewRecord

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
// Only applicable when using Remote() backend. Defaults to os.TempDir()/vecgo-cache-<random>.
func WithCacheDir(dir string) Option {
	return engine.WithCacheDir(dir)
}

// ReadOnly puts the engine in read-only mode.
// In this mode:
//   - Insert/Delete operations return ErrReadOnly
//   - No local state is required (pure memory cache)
//   - Ideal for stateless serverless search nodes
//
// Example:
//
//	db, _ := vecgo.Open(vecgo.Remote(s3Store), vecgo.ReadOnly())
func ReadOnly() Option {
	return engine.ReadOnly()
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

// WithoutData disables automatic retrieval of metadata and payload.
// By default, search returns metadata and payload for each result.
// Use this option for high-throughput scenarios where only IDs and scores are needed.
//
// Example:
//
//	results, _ := db.Search(ctx, query, 10, vecgo.WithoutData())
func WithoutData() SearchOption {
	return func(o *model.SearchOptions) {
		o.IncludeMetadata = false
		o.IncludePayload = false
		o.IncludeVector = false
	}
}

// WithVector requests the vector to be returned in the search results.
// Vectors are NOT included by default due to their large size.
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
