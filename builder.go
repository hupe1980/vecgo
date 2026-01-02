// Package vecgo provides functionalities for an embedded vector store database.
//
// This file implements index-specific fluent builder APIs for creating and configuring Vecgo instances.
// Builders are immutable - each method returns a new builder with the updated configuration.
package vecgo

import (
	"github.com/hupe1980/vecgo/codec"
	"github.com/hupe1980/vecgo/index"
	"github.com/hupe1980/vecgo/index/diskann"
	"github.com/hupe1980/vecgo/index/hnsw"
	"github.com/hupe1980/vecgo/wal"
)

// =============================================================================
// HNSW Builder (Immutable)
// =============================================================================

// HNSW creates a new HNSW index builder with the specified dimension.
// HNSW provides fast approximate nearest neighbor search in memory.
//
// The builder is immutable - each method returns a new builder with the updated configuration.
// This ensures thread-safety and prevents accidental state sharing.
//
// Example:
//
//	db, err := vecgo.HNSW[string](128).
//	    SquaredL2().
//	    M(32).
//	    EFConstruction(200).
//	    Shards(4).
//	    Build()
func HNSW[T any](dimension int) HNSWBuilder[T] {
	return HNSWBuilder[T]{
		dimension:    dimension,
		distanceType: index.DistanceTypeSquaredL2,
		m:            hnsw.DefaultOptions.M,
		ef:           hnsw.DefaultOptions.EF,
		heuristic:    hnsw.DefaultOptions.Heuristic,
		numShards:    1,
	}
}

// HNSWBuilder is an immutable fluent builder for creating HNSW-based Vecgo instances.
// Each method returns a new builder with the updated configuration.
type HNSWBuilder[T any] struct {
	dimension        int
	distanceType     index.DistanceType
	m                int
	ef               int
	heuristic        bool
	numShards        int
	randomSeed       *int64
	codec            codec.Codec
	logger           *Logger
	metrics          MetricsCollector
	walEnabled       bool
	walPath          string
	walOptions       []func(*wal.Options)
	snapshotPath     string
	syncWrite        bool
	initialArenaSize int
}

// SquaredL2 sets the distance metric to Squared Euclidean distance.
func (b HNSWBuilder[T]) SquaredL2() HNSWBuilder[T] {
	b.distanceType = index.DistanceTypeSquaredL2
	return b
}

// Cosine sets the distance metric to Cosine similarity (normalized vectors).
func (b HNSWBuilder[T]) Cosine() HNSWBuilder[T] {
	b.distanceType = index.DistanceTypeCosine
	return b
}

// DotProduct sets the distance metric to Dot Product (inner product).
func (b HNSWBuilder[T]) DotProduct() HNSWBuilder[T] {
	b.distanceType = index.DistanceTypeDotProduct
	return b
}

// M sets the maximum number of connections per layer.
// Higher values improve recall but increase memory usage.
// Default: 24. Recommended range: 12-64.
func (b HNSWBuilder[T]) M(m int) HNSWBuilder[T] {
	b.m = m
	return b
}

// EFConstruction sets the exploration factor used during index construction.
// Higher values improve index quality but slow down indexing.
// This parameter controls how many candidates are explored when inserting vectors.
// Default: 200. Recommended range: 100-500.
//
// Note: This is different from search-time EF, which is set via Search().EF().
// - EFConstruction: Controls build quality (set once at index creation)
// - Search EF: Controls query accuracy vs speed (can be tuned per-query)
func (b HNSWBuilder[T]) EFConstruction(ef int) HNSWBuilder[T] {
	b.ef = ef
	return b
}

// WithSyncWrite configures whether writes are synchronous (bypassing MemTable).
// This is primarily for benchmarking and testing.
func (b HNSWBuilder[T]) WithSyncWrite(sync bool) HNSWBuilder[T] {
	b.syncWrite = sync
	return b
}

// Heuristic enables or disables heuristic pruning.
// Default: true.
func (b HNSWBuilder[T]) Heuristic(enabled bool) HNSWBuilder[T] {
	b.heuristic = enabled
	return b
}

// Shards sets the number of shards for parallel write throughput.
// Default: 1 (no sharding). Recommended: 2-8 for high-concurrency workloads.
func (b HNSWBuilder[T]) Shards(n int) HNSWBuilder[T] {
	b.numShards = n
	return b
}

// RandomSeed sets the seed for deterministic index construction.
// If not set, a random seed (time-based) is used.
func (b HNSWBuilder[T]) RandomSeed(seed int64) HNSWBuilder[T] {
	b.randomSeed = &seed
	return b
}

// Logger sets the structured logger for operation tracing.
func (b HNSWBuilder[T]) Logger(l *Logger) HNSWBuilder[T] {
	b.logger = l
	return b
}

// Metrics sets the metrics collector for monitoring.
func (b HNSWBuilder[T]) Metrics(mc MetricsCollector) HNSWBuilder[T] {
	b.metrics = mc
	return b
}

// Codec sets the metadata codec for serialization.
func (b HNSWBuilder[T]) Codec(c codec.Codec) HNSWBuilder[T] {
	b.codec = c
	return b
}

// WAL enables Write-Ahead Logging for durability.
func (b HNSWBuilder[T]) WAL(path string, optFns ...func(*wal.Options)) HNSWBuilder[T] {
	b.walEnabled = true
	b.walPath = path
	b.walOptions = optFns
	return b
}

// SnapshotPath sets the path for automatic snapshots during WAL auto-checkpoint.
// When set, the database automatically saves snapshots when WAL thresholds are exceeded.
// This enables the delta-based mmap architecture for optimal read/write performance.
func (b HNSWBuilder[T]) SnapshotPath(path string) HNSWBuilder[T] {
	b.snapshotPath = path
	return b
}

// InitialArenaSize sets the initial size of the memory arena in bytes.
// HNSW uses a custom memory allocator (Arena) for graph nodes to improve cache locality
// and reduce GC pressure.
// Default: 64MB.
func (b HNSWBuilder[T]) InitialArenaSize(size int) HNSWBuilder[T] {
	b.initialArenaSize = size
	return b
}

// Build creates the HNSW-based Vecgo instance.
func (b HNSWBuilder[T]) Build() (*Vecgo[T], error) {
	hnswOpts := func(o *hnsw.Options) {
		o.M = b.m
		o.EF = b.ef
		o.Heuristic = b.heuristic
		o.RandomSeed = b.randomSeed
		if b.initialArenaSize > 0 {
			o.InitialArenaSize = b.initialArenaSize
		}
	}

	var vecgoOpts []Option
	if b.codec != nil {
		vecgoOpts = append(vecgoOpts, WithCodec(b.codec))
	}
	if b.numShards > 1 {
		vecgoOpts = append(vecgoOpts, WithNumShards(b.numShards))
	}
	if b.logger != nil {
		vecgoOpts = append(vecgoOpts, WithLogger(b.logger))
	}
	if b.metrics != nil {
		vecgoOpts = append(vecgoOpts, WithMetricsCollector(b.metrics))
	}
	if b.walEnabled {
		vecgoOpts = append(vecgoOpts, WithWAL(b.walPath, b.walOptions...))
	}
	if b.snapshotPath != "" {
		vecgoOpts = append(vecgoOpts, WithSnapshotPath(b.snapshotPath))
	}
	if b.syncWrite {
		vecgoOpts = append(vecgoOpts, WithSyncWrite(true))
	}

	return newHNSW[T](b.dimension, b.distanceType, []func(o *hnsw.Options){hnswOpts}, vecgoOpts)
}

// MustBuild creates the Vecgo instance, panicking on error.
func (b HNSWBuilder[T]) MustBuild() *Vecgo[T] {
	vg, err := b.Build()
	if err != nil {
		panic(err)
	}
	return vg
}

// =============================================================================
// Flat Builder (Immutable)
// =============================================================================

// Flat creates a new Flat index builder with the specified dimension.
// Flat provides exact nearest neighbor search by exhaustive comparison.
//
// The builder is immutable - each method returns a new builder with the updated configuration.
//
// Example:
//
//	db, err := vecgo.Flat[string](128).
//	    Cosine().
//	    Shards(2).
//	    Build()
func Flat[T any](dimension int) FlatBuilder[T] {
	return FlatBuilder[T]{
		dimension:    dimension,
		distanceType: index.DistanceTypeSquaredL2,
		numShards:    1,
	}
}

// FlatBuilder is an immutable fluent builder for creating Flat-based Vecgo instances.
// Each method returns a new builder with the updated configuration.
type FlatBuilder[T any] struct {
	dimension    int
	distanceType index.DistanceType
	numShards    int
	codec        codec.Codec
	logger       *Logger
	metrics      MetricsCollector
	walEnabled   bool
	walPath      string
	walOptions   []func(*wal.Options)
	snapshotPath string
	syncWrite    bool
}

// SquaredL2 sets the distance metric to Squared Euclidean distance.
func (b FlatBuilder[T]) SquaredL2() FlatBuilder[T] {
	b.distanceType = index.DistanceTypeSquaredL2
	return b
}

// Cosine sets the distance metric to Cosine similarity (normalized vectors).
func (b FlatBuilder[T]) Cosine() FlatBuilder[T] {
	b.distanceType = index.DistanceTypeCosine
	return b
}

// DotProduct sets the distance metric to Dot Product (inner product).
func (b FlatBuilder[T]) DotProduct() FlatBuilder[T] {
	b.distanceType = index.DistanceTypeDotProduct
	return b
}

// Shards sets the number of shards for parallel write throughput.
// Default: 1 (no sharding). Recommended: 2-8 for high-concurrency workloads.
func (b FlatBuilder[T]) Shards(n int) FlatBuilder[T] {
	b.numShards = n
	return b
}

// Logger sets the structured logger for operation tracing.
func (b FlatBuilder[T]) Logger(l *Logger) FlatBuilder[T] {
	b.logger = l
	return b
}

// Metrics sets the metrics collector for monitoring.
func (b FlatBuilder[T]) Metrics(mc MetricsCollector) FlatBuilder[T] {
	b.metrics = mc
	return b
}

// Codec sets the metadata codec for serialization.
func (b FlatBuilder[T]) Codec(c codec.Codec) FlatBuilder[T] {
	b.codec = c
	return b
}

// WAL enables Write-Ahead Logging for durability.
func (b FlatBuilder[T]) WAL(path string, optFns ...func(*wal.Options)) FlatBuilder[T] {
	b.walEnabled = true
	b.walPath = path
	b.walOptions = optFns
	return b
}

// SnapshotPath sets the path for automatic snapshots during WAL auto-checkpoint.
// When set, the database automatically saves snapshots when WAL thresholds are exceeded.
// This enables the delta-based mmap architecture for optimal read/write performance.
func (b FlatBuilder[T]) SnapshotPath(path string) FlatBuilder[T] {
	b.snapshotPath = path
	return b
}

// SyncWrite configures whether writes are synchronous (bypassing MemTable).
// This is primarily for benchmarking and testing.
func (b FlatBuilder[T]) SyncWrite(sync bool) FlatBuilder[T] {
	b.syncWrite = sync
	return b
}

// Build creates the Flat-based Vecgo instance.
func (b FlatBuilder[T]) Build() (*Vecgo[T], error) {
	var vecgoOpts []Option

	if b.codec != nil {
		vecgoOpts = append(vecgoOpts, WithCodec(b.codec))
	}
	if b.numShards > 1 {
		vecgoOpts = append(vecgoOpts, WithNumShards(b.numShards))
	}
	if b.logger != nil {
		vecgoOpts = append(vecgoOpts, WithLogger(b.logger))
	}
	if b.metrics != nil {
		vecgoOpts = append(vecgoOpts, WithMetricsCollector(b.metrics))
	}
	if b.walEnabled {
		vecgoOpts = append(vecgoOpts, WithWAL(b.walPath, b.walOptions...))
	}
	if b.snapshotPath != "" {
		vecgoOpts = append(vecgoOpts, WithSnapshotPath(b.snapshotPath))
	}
	if b.syncWrite {
		vecgoOpts = append(vecgoOpts, WithSyncWrite(true))
	}

	return newFlat[T](b.dimension, b.distanceType, nil, vecgoOpts)
}

// MustBuild creates the Vecgo instance, panicking on error.
func (b FlatBuilder[T]) MustBuild() *Vecgo[T] {
	vg, err := b.Build()
	if err != nil {
		panic(err)
	}
	return vg
}

// =============================================================================
// DiskANN Builder (Immutable)
// =============================================================================

// DiskANN creates a new DiskANN index builder with the specified path and dimension.
// DiskANN provides billion-scale approximate nearest neighbor search with disk-resident storage.
//
// The builder is immutable - each method returns a new builder with the updated configuration.
//
// Example:
//
//	db, err := vecgo.DiskANN[string]("./data", 128).
//	    SquaredL2().
//	    R(64).
//	    L(100).
//	    BeamWidth(4).
//	    Build()
func DiskANN[T any](path string, dimension int) DiskANNBuilder[T] {
	defaults := diskann.DefaultOptions()
	return DiskANNBuilder[T]{
		path:               path,
		dimension:          dimension,
		distanceType:       index.DistanceTypeSquaredL2,
		r:                  defaults.R,
		l:                  defaults.L,
		alpha:              defaults.Alpha,
		pqSubvectors:       defaults.PQSubvectors,
		pqCentroids:        defaults.PQCentroids,
		beamWidth:          defaults.BeamWidth,
		rerankK:            defaults.RerankK,
		binaryPrefilter:    defaults.EnableBinaryPrefilter,
		binaryPrefilterMax: defaults.BinaryPrefilterMaxNormalizedDistance,
		autoCompaction:     defaults.EnableAutoCompaction,
		compactionThresh:   defaults.CompactionThreshold,
		compactionInterval: defaults.CompactionInterval,
		compactionMinVecs:  defaults.CompactionMinVectors,
	}
}

// DiskANNBuilder is an immutable fluent builder for creating DiskANN-based Vecgo instances.
// Each method returns a new builder with the updated configuration.
type DiskANNBuilder[T any] struct {
	path               string
	dimension          int
	distanceType       index.DistanceType
	r                  int
	l                  int
	alpha              float32
	pqSubvectors       int
	pqCentroids        int
	beamWidth          int
	rerankK            int
	binaryPrefilter    bool
	binaryPrefilterMax float32
	autoCompaction     bool
	compactionThresh   float32
	compactionInterval int
	compactionMinVecs  int
	codec              codec.Codec
	logger             *Logger
	metrics            MetricsCollector
	walEnabled         bool
	walPath            string
	walOptions         []func(*wal.Options)
	snapshotPath       string
}

// BinaryPrefilter enables an optional Binary Quantization (BQ) prefilter during DiskANN search.
// This is a coarse, search-only filter; it is NOT used for graph construction or final reranking.
//
// maxNormalizedHamming must be in [0, 1]. Smaller values filter more aggressively (higher recall risk).
func (b DiskANNBuilder[T]) BinaryPrefilter(maxNormalizedHamming float32) DiskANNBuilder[T] {
	b.binaryPrefilter = true
	b.binaryPrefilterMax = maxNormalizedHamming
	return b
}

// SquaredL2 sets the distance metric to Squared Euclidean distance.
func (b DiskANNBuilder[T]) SquaredL2() DiskANNBuilder[T] {
	b.distanceType = index.DistanceTypeSquaredL2
	return b
}

// Cosine sets the distance metric to Cosine similarity (normalized vectors).
func (b DiskANNBuilder[T]) Cosine() DiskANNBuilder[T] {
	b.distanceType = index.DistanceTypeCosine
	return b
}

// DotProduct sets the distance metric to Dot Product (inner product).
func (b DiskANNBuilder[T]) DotProduct() DiskANNBuilder[T] {
	b.distanceType = index.DistanceTypeDotProduct
	return b
}

// R sets the maximum out-degree for graph nodes.
// Higher values improve recall but increase memory.
// Default: 64. Recommended range: 32-128.
func (b DiskANNBuilder[T]) R(r int) DiskANNBuilder[T] {
	b.r = r
	return b
}

// L sets the search list size during construction.
// Higher values improve quality but slow down indexing.
// Default: 100. Recommended range: 75-200.
func (b DiskANNBuilder[T]) L(l int) DiskANNBuilder[T] {
	b.l = l
	return b
}

// Alpha sets the pruning parameter for edge selection.
// Higher values create denser graphs with better recall.
// Default: 1.2. Recommended range: 1.0-1.5.
func (b DiskANNBuilder[T]) Alpha(alpha float32) DiskANNBuilder[T] {
	b.alpha = alpha
	return b
}

// PQSubvectors sets the number of subvectors for Product Quantization.
// Must divide dimension evenly. Higher values improve accuracy.
// Default: dimension/4. Recommended range: dimension/8 to dimension/2.
func (b DiskANNBuilder[T]) PQSubvectors(n int) DiskANNBuilder[T] {
	b.pqSubvectors = n
	return b
}

// PQCentroids sets the number of centroids per subspace.
// Default: 256 (for uint8 codes).
func (b DiskANNBuilder[T]) PQCentroids(n int) DiskANNBuilder[T] {
	b.pqCentroids = n
	return b
}

// BeamWidth sets the beam width for search.
// Higher values improve recall but slow down search.
// Default: 4. Recommended range: 2-8.
func (b DiskANNBuilder[T]) BeamWidth(w int) DiskANNBuilder[T] {
	b.beamWidth = w
	return b
}

// RerankK sets the number of candidates for disk-based reranking.
// Must be >= k in search queries.
// Default: 50. Recommended range: 20-200.
func (b DiskANNBuilder[T]) RerankK(k int) DiskANNBuilder[T] {
	b.rerankK = k
	return b
}

// EnableAutoCompaction enables or disables background compaction.
// Compaction removes deleted vectors and rebuilds the graph.
// Default: true.
func (b DiskANNBuilder[T]) EnableAutoCompaction(enabled bool) DiskANNBuilder[T] {
	b.autoCompaction = enabled
	return b
}

// CompactionThreshold sets the deletion ratio that triggers compaction.
// For example, 0.2 means compact when 20% of vectors are deleted.
// Default: 0.2 (20%). Recommended range: 0.1-0.3.
func (b DiskANNBuilder[T]) CompactionThreshold(threshold float32) DiskANNBuilder[T] {
	b.compactionThresh = threshold
	return b
}

// CompactionInterval sets the interval in seconds between compaction checks.
// Default: 300 (5 minutes). Recommended range: 60-600.
func (b DiskANNBuilder[T]) CompactionInterval(seconds int) DiskANNBuilder[T] {
	b.compactionInterval = seconds
	return b
}

// CompactionMinVectors sets the minimum vectors before compaction is considered.
// Prevents compacting tiny indexes.
// Default: 1000. Recommended range: 100-10000.
func (b DiskANNBuilder[T]) CompactionMinVectors(n int) DiskANNBuilder[T] {
	b.compactionMinVecs = n
	return b
}

// Logger sets the structured logger for operation tracing.
func (b DiskANNBuilder[T]) Logger(l *Logger) DiskANNBuilder[T] {
	b.logger = l
	return b
}

// Metrics sets the metrics collector for monitoring.
func (b DiskANNBuilder[T]) Metrics(mc MetricsCollector) DiskANNBuilder[T] {
	b.metrics = mc
	return b
}

// Codec sets the metadata codec for serialization.
func (b DiskANNBuilder[T]) Codec(c codec.Codec) DiskANNBuilder[T] {
	b.codec = c
	return b
}

// WAL enables Write-Ahead Logging for durability.
func (b DiskANNBuilder[T]) WAL(path string, optFns ...func(*wal.Options)) DiskANNBuilder[T] {
	b.walEnabled = true
	b.walPath = path
	b.walOptions = optFns
	return b
}

// SnapshotPath sets the path for automatic checkpoint snapshots.
// When WAL auto-checkpoint triggers (based on ops or size thresholds),
// the snapshot will be saved to this path for delta-based mmap architecture.
func (b DiskANNBuilder[T]) SnapshotPath(path string) DiskANNBuilder[T] {
	b.snapshotPath = path
	return b
}

// Build creates the DiskANN-based Vecgo instance.
func (b DiskANNBuilder[T]) Build() (*Vecgo[T], error) {
	diskannOpts := func(o *diskann.Options) {
		o.R = b.r
		o.L = b.l
		o.Alpha = b.alpha
		o.PQSubvectors = b.pqSubvectors
		o.PQCentroids = b.pqCentroids
		o.BeamWidth = b.beamWidth
		o.RerankK = b.rerankK
		o.EnableBinaryPrefilter = b.binaryPrefilter
		o.BinaryPrefilterMaxNormalizedDistance = b.binaryPrefilterMax
		o.EnableAutoCompaction = b.autoCompaction
		o.CompactionThreshold = b.compactionThresh
		o.CompactionInterval = b.compactionInterval
		o.CompactionMinVectors = b.compactionMinVecs
	}

	var vecgoOpts []Option
	if b.codec != nil {
		vecgoOpts = append(vecgoOpts, WithCodec(b.codec))
	}
	if b.logger != nil {
		vecgoOpts = append(vecgoOpts, WithLogger(b.logger))
	}
	if b.metrics != nil {
		vecgoOpts = append(vecgoOpts, WithMetricsCollector(b.metrics))
	}
	if b.walEnabled {
		vecgoOpts = append(vecgoOpts, WithWAL(b.walPath, b.walOptions...))
	}
	if b.snapshotPath != "" {
		vecgoOpts = append(vecgoOpts, WithSnapshotPath(b.snapshotPath))
	}

	return newDiskANN[T](b.path, b.dimension, b.distanceType, []func(o *diskann.Options){diskannOpts}, vecgoOpts)
}

// MustBuild creates the Vecgo instance, panicking on error.
func (b DiskANNBuilder[T]) MustBuild() *Vecgo[T] {
	vg, err := b.Build()
	if err != nil {
		panic(err)
	}
	return vg
}
