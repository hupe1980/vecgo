package vecgo

import (
	"log/slog"

	"github.com/hupe1980/vecgo/codec"
	"github.com/hupe1980/vecgo/engine"
	"github.com/hupe1980/vecgo/wal"
)

type options struct {
	codec             codec.Codec
	numShards         int
	metricsCollector  MetricsCollector
	logger            *Logger
	walPath           string
	walOptions        []func(*wal.Options)
	snapshotPath      string                   // Path for auto-checkpoint snapshots
	validationLimits  *engine.ValidationLimits // nil = use defaults, empty = disable validation
	disableValidation bool                     // explicit disable flag
	dimension         int                      // Vector dimension (set by builders)
}

// Option configures Vecgo constructor/load behavior.
//
// Today options primarily exist to avoid exploding the API surface
// (e.g. codec-specific constructor variants).
//
// Breaking changes are expected while Vecgo is pre-release.
type Option func(*options)

// WithCodec configures the codec used for encoding/decoding the user payload (DataStore).
//
// Note: Metadata is always encoded using the internal VecgoBinary format.
// If nil is passed, codec.Default is used.
func WithCodec(c codec.Codec) Option {
	return func(o *options) {
		if c == nil {
			c = codec.Default
		}
		o.codec = c
	}
}

// WithNumShards configures the number of shards for parallel write throughput.
// , numShards: 1
// Sharding eliminates the global lock bottleneck by partitioning writes across
// independent coordinators. Each shard has its own lock, enabling true parallel
// write operations on multi-core systems.
//
// Benefits:
//   - 3-8x write speedup on multi-core systems (measured on 2-8 cores)
//   - Balanced load distribution via hash-based partitioning
//   - Independent per-shard locks eliminate contention
//
// Trade-offs:
//   - Search requires fan-out to all shards (minor overhead)
//   - HNSW graphs are per-shard (no cross-shard edges)
//   - Memory overhead: N metadata stores, N coordinators
//
// Recommended values:
//   - numShards=1: Single-threaded workloads (default, no overhead)
//   - numShards=2-4: Moderate concurrency
//   - numShards=8-16: High concurrency workloads
//
// Performance scales sub-linearly due to search fan-out and merge overhead.
// Benchmark your workload to find optimal value.
//
// If numShards <= 1, sharding is disabled (backward compatible).
func WithNumShards(numShards int) Option {
	return func(o *options) {
		o.numShards = numShards
	}
}

// WithWAL configures Write-Ahead Logging for durability.
// WAL is immutable after database creation - it cannot be enabled/disabled at runtime.
//
// Example:
//
//	vecgo.HNSW[string](128).
//	    SquaredL2().
//	    WAL("./wal", func(o *wal.Options) {
//	        o.DurabilityMode = wal.GroupCommit
//	        o.GroupCommitInterval = 10 * time.Millisecond
//	    }).
//	    Build()
func WithWAL(path string, optFns ...func(*wal.Options)) Option {
	return func(o *options) {
		o.walPath = path
		o.walOptions = optFns
	}
}

// WithSnapshotPath configures the path for automatic snapshots.
// When set along with WAL auto-checkpoint thresholds (AutoCheckpointOps, AutoCheckpointMB),
// the database will automatically save snapshots when thresholds are exceeded.
//
// This enables the "delta-based mmap" architecture:
//   - Writes go to in-memory delta + WAL (fast: 1.4µs)
//   - Auto-flush delta to mmap base periodically
//   - Reads use mmap zero-copy (fast: 3.3ms for 10K vectors)
//
// Example:
//
//	db, _ := vecgo.HNSW[string](128).
//	    SquaredL2().
//	    WAL("./wal", func(o *wal.Options) {
//	        o.AutoCheckpointOps = 10000  // Auto-save every 10k ops
//	        o.AutoCheckpointMB = 100     // Or at 100MB WAL size
//	    }).
//	    SnapshotPath("./data/snapshot.bin").
//	    Build()
func WithSnapshotPath(path string) Option {
	return func(o *options) {
		o.snapshotPath = path
	}
}

// WithMetricsCollector configures a metrics collector for monitoring operations.
// Pass nil to disable metrics collection.
//
// Example with BasicMetricsCollector:
//
//	metrics := &vecgo.BasicMetricsCollector{}
//	vg, _ := vecgo.New(index, vecgo.WithMetricsCollector(metrics))
//	// ... use vg ...
//	stats := metrics.GetStats()
//	fmt.Printf("Inserts: %d, Avg latency: %dns\n", stats.InsertCount, stats.InsertAvgNanos)
func WithMetricsCollector(mc MetricsCollector) Option {
	return func(o *options) {
		o.metricsCollector = mc
	}
}

// WithLogger configures structured logging for operations.
// Pass nil to disable logging.
//
// Example with JSON logging:
//
//	logger := vecgo.NewJSONLogger(slog.LevelInfo)
//	vg, _ := vecgo.New(index, vecgo.WithLogger(logger))
func WithLogger(logger *Logger) Option {
	return func(o *options) {
		o.logger = logger
	}
}

// WithLogLevel creates a text logger with the specified level and sets it.
// Convenience wrapper for WithLogger(NewTextLogger(level)).
func WithLogLevel(level slog.Level) Option {
	return func(o *options) {
		o.logger = NewTextLogger(level)
	}
}

// WithValidationLimits configures input validation with custom limits.
// Validation prevents crashes from nil vectors, NaN/Inf values, oversized batches, etc.
//
// Example with custom limits:
//
//	limits := &engine.ValidationLimits{
//	    MaxDimension:     1024,
//	    MaxVectors:       1_000_000,
//	    MaxK:             100,
//	    MaxMetadataBytes: 4096,
//	    MaxBatchSize:     1000,
//	}
//	vg, _ := vecgo.New(index, vecgo.WithValidationLimits(limits))
//
// Pass nil to use default limits (recommended for most use cases).
func WithValidationLimits(limits *engine.ValidationLimits) Option {
	return func(o *options) {
		o.validationLimits = limits
	}
}

// WithoutValidation disables input validation entirely.
// Use only when you have pre-validated inputs and need maximum performance.
// WARNING: Invalid inputs (nil, NaN, oversized) may cause panics or corruption.
func WithoutValidation() Option {
	return func(o *options) {
		o.disableValidation = true
	}
}

// withDimension is an internal option used by builders to set the vector dimension.
// This is used by the validation layer to check vector dimensions.
func withDimension(dim int) Option {
	return func(o *options) {
		o.dimension = dim
	}
}

func applyOptions(optFns []Option) options {
	o := options{
		codec:            nil,
		metricsCollector: NoopMetricsCollector{},
		logger:           NoopLogger(),
	}
	for _, fn := range optFns {
		if fn != nil {
			fn(&o)
		}
	}
	return o
}
