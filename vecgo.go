package vecgo

import (
	"context"
	"log/slog"
	"time"

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
	open(ctx context.Context, opts ...Option) (*engine.Engine, error)
}

// localBackend is a local filesystem backend.
type localBackend string

func (b localBackend) open(ctx context.Context, opts ...Option) (*engine.Engine, error) {
	return engine.OpenLocal(ctx, string(b), opts...)
}

// remoteBackend is a remote BlobStore backend (S3, GCS, Azure, etc.).
type remoteBackend struct {
	store blobstore.BlobStore
}

func (b remoteBackend) open(ctx context.Context, opts ...Option) (*engine.Engine, error) {
	return engine.OpenRemote(ctx, b.store, opts...)
}

// Local creates a local filesystem backend.
//
// Example:
//
//	ctx := context.Background()
//	db, _ := vecgo.Open(ctx, vecgo.Local("./data"), vecgo.Create(128, vecgo.MetricL2))
func Local(path string) Backend {
	return localBackend(path)
}

// Remote creates a remote storage backend (S3, GCS, Azure, etc.).
//
// Example:
//
//	s3Store, _ := s3.New(ctx, "my-bucket")
//	db, _ := vecgo.Open(ctx, vecgo.Remote(s3Store), vecgo.WithCacheDir("/fast/nvme"))
func Remote(store blobstore.BlobStore) Backend {
	return remoteBackend{store: store}
}

// Open opens or creates a vector database with the given backend.
// The context is used for initialization I/O and can be used for timeouts.
//
// For new indexes, use the Create option:
//
//	ctx := context.Background()
//	db, _ := vecgo.Open(ctx, vecgo.Local("./data"), vecgo.Create(128, vecgo.MetricL2))
//
// For existing indexes, dimension and metric are loaded from the manifest:
//
//	db, _ := vecgo.Open(ctx, vecgo.Local("./data"))
//
// For cloud storage:
//
//	db, _ := vecgo.Open(ctx, vecgo.Remote(s3Store))
func Open(ctx context.Context, backend Backend, opts ...Option) (*DB, error) {
	e, err := backend.open(ctx, opts...)
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
//	id, err := db.InsertRecord(ctx, rec)
func (db *DB) InsertRecord(ctx context.Context, rec Record) (ID, error) {
	return db.Insert(ctx, rec.Vector, rec.Metadata, rec.Payload)
}

// BatchInsertRecords inserts multiple records in a single batch.
//
// Example:
//
//	records := []vecgo.Record{rec1, rec2, rec3}
//	ids, err := db.BatchInsertRecords(ctx, records)
func (db *DB) BatchInsertRecords(ctx context.Context, records []Record) ([]ID, error) {
	vectors := make([][]float32, len(records))
	mds := make([]metadata.Document, len(records))
	payloads := make([][]byte, len(records))

	for i, rec := range records {
		vectors[i] = rec.Vector
		mds[i] = rec.Metadata
		payloads[i] = rec.Payload
	}

	return db.BatchInsert(ctx, vectors, mds, payloads)
}

// Vacuum removes old manifest versions and their orphaned segments based on
// the RetentionPolicy configured via WithRetentionPolicy().
//
// This reclaims disk space from:
//   - Old manifest versions (beyond KeepVersions or KeepDuration)
//   - Segments no longer referenced by any retained manifest
//
// Time-travel queries to vacuumed versions will fail.
//
// Safe to call periodically (e.g., daily cron job). No-op if no retention policy is set.
//
// Example:
//
//	// Configure retention when opening
//	policy := vecgo.RetentionPolicy{KeepVersions: 10}
//	db, _ := vecgo.Open(ctx, vecgo.Local("./data"), vecgo.WithRetentionPolicy(policy))
//
//	// Later, reclaim space from expired versions
//	err := db.Vacuum(ctx)
func (db *DB) Vacuum(ctx context.Context) error {
	return db.Engine.Vacuum(ctx)
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
//	db, _ := vecgo.Open(ctx, vecgo.Remote(s3Store), vecgo.ReadOnly())
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
	return engine.WithRefineFactor(factor)
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

// WithStats enables collection of detailed query execution statistics.
// Pass a pointer to a QueryStats struct that will be populated after the query.
// Use this for query debugging, performance analysis, and cost estimation.
//
// Example:
//
//	stats := &vecgo.QueryStats{}
//	results, _ := db.Search(ctx, query, 10, vecgo.WithStats(stats))
//	fmt.Println(stats.Explain()) // Human-readable explanation
//	fmt.Printf("Distance computations: %d\n", stats.DistanceComputations)
func WithStats(stats *QueryStats) SearchOption {
	return func(o *model.SearchOptions) {
		o.Stats = stats
	}
}

// QueryStats provides detailed execution statistics for a search query.
// Use WithStats to collect these during a search.
type QueryStats = model.QueryStats

// SegmentQueryStats contains execution statistics for a single segment.
type SegmentQueryStats = model.SegmentQueryStats

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
// Used with Vacuum() to control time-travel storage overhead.
type RetentionPolicy = engine.RetentionPolicy

// VacuumStats holds results of the vacuum operation.
type VacuumStats = engine.VacuumStats

// WithRetentionPolicy sets the retention policy for time-travel versioning.
// Old manifests and their orphaned segments are removed by Vacuum() based on this policy.
//
// Example:
//
//	policy := vecgo.RetentionPolicy{
//	    KeepVersions: 10,                    // Keep last 10 versions
//	    KeepDuration: 7 * 24 * time.Hour,    // OR keep 7 days of history
//	}
//	db, _ := vecgo.Open(ctx, vecgo.Local("./data"), vecgo.WithRetentionPolicy(policy))
func WithRetentionPolicy(p RetentionPolicy) Option {
	return engine.WithRetentionPolicy(p)
}

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

// WithTimestamp opens the database at the state closest to the given time (Time-Travel).
// This enables querying historical data states without loading the latest version.
//
// Example:
//
//	// Query the database as it was yesterday
//	yesterday := time.Now().Add(-24 * time.Hour)
//	db, err := vecgo.Open(ctx, vecgo.Local("./data"), vecgo.WithTimestamp(yesterday))
//
// Note: The database must have been committed at least once at or before the given timestamp.
// If no version exists at or before the timestamp, an error is returned.
func WithTimestamp(t time.Time) Option {
	return engine.WithTimestamp(t)
}

// WithVersion opens the database at a specific manifest version ID (Time-Travel).
// Version IDs are monotonically increasing and assigned on each Commit().
//
// Example:
//
//	// Open at version 42
//	db, err := vecgo.Open(ctx, vecgo.Local("./data"), vecgo.WithVersion(42))
//
// Use Stats().ManifestID to get the current version ID after opening.
func WithVersion(v uint64) Option {
	return engine.WithVersion(v)
}
