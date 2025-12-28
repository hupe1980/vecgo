// Package vecgo provides a high-performance embedded vector database for Go.
//
// Vecgo supports efficient vector indexing and approximate nearest neighbor (ANN) search
// with production-ready features including:
//
//   - Multiple index types: Flat (exact), HNSW (in-memory ANN), DiskANN (disk-resident)
//   - Type-safe fluent builders: HNSW[T](), Flat[T](), DiskANN[T]()
//   - Thread-safe CRUD operations with optional sharding for multi-core scaling
//   - Metadata filtering with Roaring Bitmap-based inverted index
//   - SIMD-optimized distance kernels (AVX/AVX512 on x86_64, NEON on ARM64)
//   - Zero-allocation search with pooled buffers
//   - Columnar vector storage (SOA layout) with mmap support
//   - Write-Ahead Logging (WAL) with group commit for durability
//   - Streaming search API for memory-efficient iteration
//   - Quantization (Binary, PQ, OPQ) for memory reduction
//   - Background compaction for deleted vectors
//
// # Index Selection
//
// Choose the right index for your dataset:
//   - Flat: <100K vectors, 100% recall (exact search)
//   - HNSW: 100K-10M vectors, 95-99% recall (fast in-memory)
//   - DiskANN: 10M+ vectors, 90-95% recall (disk-resident, scales to billions)
//
// # Quick Start (Fluent API)
//
// Create an HNSW index with type-safe builder:
//
//	ctx := context.Background()
//	db, err := vecgo.HNSW[string](128).  // 128-dimensional vectors
//	    SquaredL2().                      // Distance function
//	    M(32).                            // Graph connectivity
//	    EF(200).                          // Search quality
//	    Shards(4).                        // Multi-core scaling
//	    WAL("./wal", func(o *wal.Options) {
//	        o.DurabilityMode = wal.GroupCommit
//	        o.GroupCommitInterval = 10 * time.Millisecond
//	    }).
//	    Build()
//	if err != nil {
//	    panic(err)
//	}
//	defer db.Close()
//
// Insert vectors with metadata:
//
//	id, err := db.Insert(ctx, vecgo.VectorWithData[string]{
//	    Vector: []float32{1.0, 2.0, 3.0, ...},
//	    Data:   "my document",
//	    Metadata: metadata.Metadata{
//	        "category": "tech",
//	        "year": 2024,
//	    },
//	})
//
// Batch insert for better performance:
//
//	items := []vecgo.VectorWithData[string]{
//	    {Vector: vec1, Data: "doc1", Metadata: meta1},
//	    {Vector: vec2, Data: "doc2", Metadata: meta2},
//	}
//	results := db.BatchInsert(ctx, items)
//
// Search with fluent API:
//
//	results, err := db.Search(query).
//	    KNN(10).                                  // Top 10 results
//	    EF(400).                                  // Override search quality
//	    Filter(metadata.Eq("category", "tech")). // Metadata filter
//	    Execute(ctx)
//
// Streaming search for memory efficiency:
//
//	for result, err := range db.Search(query).KNN(100).Stream(ctx) {
//	    if err != nil {
//	        log.Fatal(err)
//	    }
//	    if result.Distance > threshold {
//	        break  // Early termination
//	    }
//	    process(result)
//	}
//
// # Other Index Types
//
// Flat index (exact search):
//
//	db, err := vecgo.Flat[string](128).
//	    Cosine().
//	    Build()
//
// DiskANN index (billion-scale):
//
//	db, err := vecgo.DiskANN[string]("./data", 128).
//	    SquaredL2().
//	    R(64).                            // Graph degree
//	    L(100).                           // Search list size
//	    EnableAutoCompaction(true).       // Background cleanup
//	    CompactionThreshold(0.2).         // Compact at 20% deleted
//	    Build()
//
// # Performance Tuning
//
// See docs/tuning.md for detailed guidance on:
//   - HNSW parameters (M, EF, efConstruction)
//   - Sharding for multi-core scaling (2.7x-3.4x speedup)
//   - WAL durability modes (Async/GroupCommit/Sync)
//   - Quantization (Binary/PQ/OPQ) for memory reduction
//   - DiskANN configuration (R, L, BeamWidth)
//
// For architecture details, see docs/architecture.md.
package vecgo

import (
	"context"
	"errors"
	"fmt"
	"io"
	"iter"
	"time"

	"github.com/hupe1980/vecgo/codec"
	"github.com/hupe1980/vecgo/engine"
	"github.com/hupe1980/vecgo/index"
	"github.com/hupe1980/vecgo/index/diskann"
	"github.com/hupe1980/vecgo/index/flat"
	"github.com/hupe1980/vecgo/index/hnsw"
	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/wal"
)

var (
	// ErrNotFound is returned when an item is not found.
	ErrNotFound = errors.New("not found")

	// ErrInvalidEFValue is returned when the explore factor (ef) is less than the value of k.
	ErrInvalidEFValue = errors.New("explore factor (ef) must be at least the value of k")
)

// Vecgo is a vector store database with support for metadata filtering and write-ahead logging.
type Vecgo[T any] struct {
	index        index.Index
	store        engine.Store[T]
	metaStore    *metadata.UnifiedIndex
	mmapCloser   io.Closer
	coordinator  coordinator[T] // interface: *engine.Coordinator[T] or *engine.ShardedCoordinator[T]
	wal          *wal.WAL
	codec        codec.Codec
	metrics      MetricsCollector
	logger       *Logger
	snapshotPath string // Path for auto-checkpoint snapshots (if set, enables delta-based mmap)
}

// coordinator is an internal interface satisfied by both
// engine.Coordinator[T] and engine.ShardedCoordinator[T].
type coordinator[T any] interface {
	Insert(ctx context.Context, vector []float32, data T, metadata metadata.Metadata) (uint32, error)
	BatchInsert(ctx context.Context, vectors [][]float32, dataSlice []T, metadataSlice []metadata.Metadata) ([]uint32, error)
	Update(ctx context.Context, id uint32, vector []float32, data T, metadata metadata.Metadata) error
	Delete(ctx context.Context, id uint32) error
	KNNSearch(ctx context.Context, query []float32, k int, opts *index.SearchOptions) ([]index.SearchResult, error)
	BruteSearch(ctx context.Context, query []float32, k int, filter func(id uint32) bool) ([]index.SearchResult, error)
}

// EnableProductQuantization enables Product Quantization (PQ) on the underlying index,
// if the index supports it.
//
// PQ is optional/explicit and can accelerate distance computations by using compact
// codes for query->vector distance approximations.
func (vg *Vecgo[T]) EnableProductQuantization(cfg index.ProductQuantizationConfig) error {
	if vg.index == nil {
		return fmt.Errorf("vecgo: index not initialized")
	}
	cap, ok := vg.index.(index.ProductQuantizationEnabler)
	if !ok {
		return fmt.Errorf("vecgo: index type %T does not support product quantization", vg.index)
	}
	return cap.EnableProductQuantization(cfg)
}

// DisableProductQuantization disables Product Quantization (PQ) on the underlying index.
func (vg *Vecgo[T]) DisableProductQuantization() {
	if vg.index == nil {
		return
	}
	if cap, ok := vg.index.(index.ProductQuantizationEnabler); ok {
		cap.DisableProductQuantization()
	}
}

// newFlat is an internal constructor used by the Flat builder.
// External users should use vecgo.Flat[T](dimension).Build() instead.
func newFlat[T any](dimension int, distanceType index.DistanceType, indexOptFns []func(o *flat.Options), vecgoOptFns []Option) (*Vecgo[T], error) {
	opts := flat.DefaultOptions
	for _, fn := range indexOptFns {
		fn(&opts)
	}
	// Arguments are authoritative.
	opts.Dimension = dimension
	opts.DistanceType = distanceType

	vecOpts := applyOptions(vecgoOptFns)

	// Non-sharded mode (numShards <= 1)
	if vecOpts.numShards <= 1 {
		i, err := flat.New(func(o *flat.Options) {
			*o = opts
		})
		if err != nil {
			return nil, translateError(err)
		}

		vg, err := new(i, engine.NewMapStore[T](), metadata.NewUnifiedIndex(), vecgoOptFns...)
		return vg, translateError(err)
	}

	// Sharded mode (numShards > 1)
	numShards := vecOpts.numShards
	indexes := make([]index.Index, numShards)
	dataStores := make([]engine.Store[T], numShards)
	metaStores := make([]*metadata.UnifiedIndex, numShards)

	for i := range numShards {
		idx, err := flat.NewSharded(i, numShards, func(o *flat.Options) {
			*o = opts
		})
		if err != nil {
			return nil, translateError(err)
		}
		indexes[i] = idx
		dataStores[i] = engine.NewMapStore[T]()
		metaStores[i] = metadata.NewUnifiedIndex()
	}

	vg, err := newSharded(indexes, dataStores, metaStores, vecgoOptFns...)
	return vg, translateError(err)
}

// newHNSW is an internal constructor used by the HNSW builder.
// External users should use vecgo.HNSW[T](dimension).Build() instead.
func newHNSW[T any](dimension int, distanceType index.DistanceType, indexOptFns []func(o *hnsw.Options), vecgoOptFns []Option) (*Vecgo[T], error) {
	opts := hnsw.DefaultOptions
	for _, fn := range indexOptFns {
		fn(&opts)
	}
	// Arguments are authoritative.
	opts.Dimension = dimension
	opts.DistanceType = distanceType

	vecOpts := applyOptions(vecgoOptFns)

	// Non-sharded mode (numShards <= 1)
	if vecOpts.numShards <= 1 {
		i, err := hnsw.New(func(o *hnsw.Options) {
			*o = opts
		})
		if err != nil {
			return nil, translateError(err)
		}

		vg, err := new(i, engine.NewMapStore[T](), metadata.NewUnifiedIndex(), vecgoOptFns...)
		return vg, translateError(err)
	}

	// Sharded mode (numShards > 1)
	numShards := vecOpts.numShards
	indexes := make([]index.Index, numShards)
	dataStores := make([]engine.Store[T], numShards)
	metaStores := make([]*metadata.UnifiedIndex, numShards)

	for i := 0; i < numShards; i++ {
		idx, err := hnsw.NewSharded(i, numShards, func(o *hnsw.Options) {
			*o = opts
		})
		if err != nil {
			return nil, translateError(err)
		}
		indexes[i] = idx
		dataStores[i] = engine.NewMapStore[T]()
		metaStores[i] = metadata.NewUnifiedIndex()
	}

	vg, err := newSharded(indexes, dataStores, metaStores, vecgoOptFns...)
	return vg, translateError(err)
}

// newDiskANN is an internal constructor used by the DiskANN builder.
// External users should use vecgo.DiskANN[T](path, dimension).Build() instead.
func newDiskANN[T any](path string, dimension int, distanceType index.DistanceType, indexOptFns []func(o *diskann.Options), vecgoOptFns []Option) (*Vecgo[T], error) {
	opts := diskann.DefaultOptions()
	for _, fn := range indexOptFns {
		fn(opts)
	}

	// Create DiskANN index (always mutable via New)
	i, err := diskann.New(dimension, distanceType, path, opts)
	if err != nil {
		return nil, translateError(err)
	}

	vg, err := new(i, engine.NewMapStore[T](), metadata.NewUnifiedIndex(), vecgoOptFns...)
	return vg, translateError(err)
}

// new creates a new Vecgo instance with the given index, data store, and metadata store.
// This is an internal constructor - external users should use builders (HNSW[T](), Flat[T](), DiskANN[T]()).
func new[T any](i index.Index, s engine.Store[T], ms *metadata.UnifiedIndex, optFns ...Option) (*Vecgo[T], error) {
	opts := applyOptions(optFns)

	// Set codec (default if not specified)
	c := opts.codec
	if c == nil {
		c = codec.Default
	}

	// Create WAL if path is specified
	var w *wal.WAL
	if opts.walPath != "" {
		// Prepare WAL options with codec
		walOptFns := append([]func(*wal.Options){
			func(o *wal.Options) {
				o.Path = opts.walPath
				o.MetadataCodec = c // Use unified codec
			},
		}, opts.walOptions...)

		var err error
		w, err = wal.New(walOptFns...)
		if err != nil {
			return nil, fmt.Errorf("vecgo: failed to create WAL: %w", err)
		}
	}

	vg := &Vecgo[T]{
		index:        i,
		store:        s,
		metaStore:    ms,
		mmapCloser:   nil,
		coordinator:  nil, // set below
		wal:          w,
		codec:        c,
		metrics:      opts.metricsCollector,
		logger:       opts.logger,
		snapshotPath: opts.snapshotPath,
	}

	// Convert typed nil to untyped nil for interface
	var durability engine.Durability
	if w != nil {
		durability = w
	}

	coord, err := engine.New(i, s, ms, durability, c)
	if err != nil {
		if w != nil {
			_ = w.Close()
		}
		return nil, translateError(err)
	}
	vg.coordinator = coord

	// Set auto-checkpoint callback if WAL is enabled
	if w != nil {
		w.SetCheckpointCallback(vg.autoCheckpoint)
	}

	return vg, nil
}

// newSharded creates a new sharded Vecgo instance for parallel write throughput.
// All slices (indexes, dataStores, metaStores) must have the same length (numShards).
// This is an internal constructor - external users should use builders with .Shards(n) option.
func newSharded[T any](indexes []index.Index, dataStores []engine.Store[T], metaStores []*metadata.UnifiedIndex, optFns ...Option) (*Vecgo[T], error) {
	if len(indexes) == 0 {
		return nil, fmt.Errorf("vecgo: at least one shard required, got 0")
	}
	if len(indexes) != len(dataStores) || len(indexes) != len(metaStores) {
		return nil, fmt.Errorf("vecgo: shard count mismatch: indexes=%d, dataStores=%d, metaStores=%d", len(indexes), len(dataStores), len(metaStores))
	}

	opts := applyOptions(optFns)

	// Set codec (default if not specified)
	c := opts.codec
	if c == nil {
		c = codec.Default
	}

	// Create WAL if path is specified
	var w *wal.WAL
	if opts.walPath != "" {
		// Prepare WAL options with codec
		walOptFns := append([]func(*wal.Options){
			func(o *wal.Options) {
				o.Path = opts.walPath
				o.MetadataCodec = c // Use unified codec
			},
		}, opts.walOptions...)

		var err error
		w, err = wal.New(walOptFns...)
		if err != nil {
			return nil, fmt.Errorf("vecgo: failed to create WAL: %w", err)
		}
	}

	// Create durability slice (all shards share the same WAL)
	var durabilities []engine.Durability
	if w != nil {
		durabilities = make([]engine.Durability, len(indexes))
		for i := range durabilities {
			durabilities[i] = w
		}
	}

	// Create sharded coordinator
	shardedCoord, err := engine.NewSharded(indexes, dataStores, metaStores, durabilities, c)
	if err != nil {
		if w != nil {
			_ = w.Close()
		}
		return nil, translateError(err)
	}

	// Use first shard's index as primary (for Stats, etc.)
	vg := &Vecgo[T]{
		index:        indexes[0],
		store:        dataStores[0],
		metaStore:    metaStores[0],
		mmapCloser:   nil,
		coordinator:  shardedCoord,
		wal:          w,
		codec:        c,
		metrics:      opts.metricsCollector,
		logger:       opts.logger,
		snapshotPath: opts.snapshotPath,
	}

	// Set auto-checkpoint callback if WAL is enabled
	if w != nil {
		w.SetCheckpointCallback(vg.autoCheckpoint)
	}

	return vg, nil
}

// autoCheckpoint is called by WAL when auto-checkpoint thresholds are exceeded.
// If snapshotPath is configured, it saves a snapshot and checkpoints the WAL.
// This implements the "delta-based mmap" architecture: in-memory delta is flushed
// to a new mmap base periodically.
func (vg *Vecgo[T]) autoCheckpoint() error {
	if vg.snapshotPath == "" {
		// No snapshot path configured - user must handle checkpointing manually
		return nil
	}

	// Save snapshot to disk (flushes in-memory delta to mmap base)
	if err := vg.SaveToFile(vg.snapshotPath); err != nil {
		return fmt.Errorf("auto-checkpoint: failed to save snapshot: %w", err)
	}

	return nil
}

// Checkpoint creates a checkpoint in the WAL and truncates the log.
// This should be called after saving the index to disk.
func (vg *Vecgo[T]) Checkpoint() error {
	if vg.wal != nil {
		return vg.wal.Checkpoint()
	}
	return nil
}

// NewFromFile loads a snapshot using zero-copy mmap.
// This is the only supported way to load snapshots (regular loading allocates 153GB for 10K vectors).
//
// Options:
//   - WithCodec: choose the codec used to decode snapshot sections.
//   - WithWAL: enable Write-Ahead Logging for durability.
//
// IMPORTANT: The returned Vecgo must be Close()'d when no longer needed to unmap the file.
//
// Performance: 760x faster than regular loading (3.4ms vs 2.5s for 10K vectors),
// 53,000x less memory (2.9MB vs 153GB).
func NewFromFile[T any](filename string, optFns ...Option) (*Vecgo[T], error) {
	opts := applyOptions(optFns)
	snap, err := engine.LoadFromFileMmapWithCodec[T](filename, opts.codec)
	if err != nil {
		return nil, translateError(err)
	}

	// Convert map store to UnifiedIndex
	metaStore := metadata.NewUnifiedIndex()
	for id, doc := range snap.MetadataStore.ToMap() {
		if doc != nil {
			metaStore.Set(id, doc)
		}
	}

	// Set codec (default if not specified)
	c := opts.codec
	if c == nil {
		c = codec.Default
	}

	// Create WAL if path is specified
	var w *wal.WAL
	if opts.walPath != "" {
		// Prepare WAL options with codec
		walOptFns := append([]func(*wal.Options){
			func(o *wal.Options) {
				o.Path = opts.walPath
				o.MetadataCodec = c // Use unified codec
			},
		}, opts.walOptions...)

		w, err = wal.New(walOptFns...)
		if err != nil {
			_ = snap.MappedFile.Close()
			return nil, fmt.Errorf("vecgo: failed to create WAL: %w", err)
		}
	}

	// Use snapshotPath from options, defaulting to the loaded filename for auto-checkpoint
	snapshotPath := opts.snapshotPath
	if snapshotPath == "" {
		snapshotPath = filename // Default to rewriting the loaded file
	}

	vg := &Vecgo[T]{
		index:        snap.Index,
		store:        snap.DataStore,
		metaStore:    metaStore,
		mmapCloser:   snap.MappedFile,
		coordinator:  nil, // set below
		wal:          w,
		codec:        c,
		logger:       opts.logger,
		metrics:      opts.metricsCollector,
		snapshotPath: snapshotPath,
	}
	coord, err := engine.New(snap.Index, snap.DataStore, metaStore, w, c)
	if err != nil {
		if w != nil {
			_ = w.Close()
		}
		_ = snap.MappedFile.Close()
		return nil, translateError(err)
	}
	vg.coordinator = coord

	// Set auto-checkpoint callback if WAL is enabled
	if w != nil {
		w.SetCheckpointCallback(vg.autoCheckpoint)
	}

	return vg, nil
}

// Get retrieves an item by ID.
func (vg *Vecgo[T]) Get(id uint32) (T, error) {
	data, ok := vg.store.Get(id)
	if !ok {
		var zero T
		return zero, ErrNotFound
	}

	return data, nil
}

// VectorWithData represents a vector along with associated data and optional metadata.
type VectorWithData[T any] struct {
	Vector   []float32
	Data     T
	Metadata metadata.Metadata
}

// Insert inserts a vector along with associated data into the database.
func (vg *Vecgo[T]) Insert(ctx context.Context, item VectorWithData[T]) (uint32, error) {
	start := time.Now()
	if vg.coordinator == nil {
		return 0, fmt.Errorf("vecgo: coordinator not initialized (internal error - use builder API)")
	}
	id, err := vg.coordinator.Insert(ctx, item.Vector, item.Data, item.Metadata)
	duration := time.Since(start)
	err = translateError(err)
	vg.metrics.RecordInsert(duration, err)
	vg.logger.LogInsert(ctx, id, len(item.Vector), err)
	return id, err
}

// BatchInsertResult represents the result of a batch insert as it:
//   - Uses a single coordinated mutation path across index + stores + metadata index
//   - Performs a single fsync for all WAL entries (if WAL is enabled)
//   - Amortizes overhead across all items
//
// When WAL is enabled, successful insertions are logged for crash recovery.
// WAL affects durability, not atomicity.
type BatchInsertResult[T any] struct {
	IDs    []uint32 // IDs of successfully inserted items
	Errors []error  // Errors for failed insertions (nil for successful)
}

// BatchInsert inserts multiple vectors along with associated data into the database.
// This is more efficient than calling Insert multiple times.
func (vg *Vecgo[T]) BatchInsert(ctx context.Context, items []VectorWithData[T]) BatchInsertResult[T] {
	start := time.Now()
	result := BatchInsertResult[T]{
		IDs:    make([]uint32, 0),
		Errors: make([]error, len(items)),
	}
	if vg.coordinator == nil {
		err := fmt.Errorf("vecgo: coordinator not initialized (internal error - use builder API)")
		for i := range result.Errors {
			result.Errors[i] = err
		}
		return result
	}

	ids, err := vg.coordinator.BatchInsert(ctx,
		extractVectors(items),
		extractData(items),
		extractMetadata(items))
	if err != nil {
		err = translateError(err)
		for i := range result.Errors {
			result.Errors[i] = err
		}
		duration := time.Since(start)
		vg.metrics.RecordBatchInsert(len(items), len(items), duration)
		vg.logger.LogBatchInsert(ctx, len(items), len(items))
		return result
	}
	result.IDs = ids

	// Count failures
	failedCount := 0
	for _, e := range result.Errors {
		if e != nil {
			failedCount++
		}
	}
	duration := time.Since(start)
	vg.metrics.RecordBatchInsert(len(items), failedCount, duration)
	vg.logger.LogBatchInsert(ctx, len(items), failedCount)
	return result
}

// Helper functions for extracting data from VectorWithData slice
func extractVectors[T any](items []VectorWithData[T]) [][]float32 {
	vectors := make([][]float32, len(items))
	for i, item := range items {
		vectors[i] = item.Vector
	}
	return vectors
}

func extractData[T any](items []VectorWithData[T]) []T {
	data := make([]T, len(items))
	for i, item := range items {
		data[i] = item.Data
	}
	return data
}

func extractMetadata[T any](items []VectorWithData[T]) []metadata.Metadata {
	meta := make([]metadata.Metadata, len(items))
	for i, item := range items {
		meta[i] = item.Metadata
	}
	return meta
}

// Delete removes a vector and associated data from the database.
func (vg *Vecgo[T]) Delete(ctx context.Context, id uint32) error {
	start := time.Now()
	if vg.coordinator == nil {
		return fmt.Errorf("vecgo: coordinator not initialized (ensure vecgo.New was called correctly)")
	}
	err := translateError(vg.coordinator.Delete(ctx, id))
	duration := time.Since(start)
	vg.metrics.RecordDelete(duration, err)
	vg.logger.LogDelete(ctx, id, err)
	return err
}

// Update updates a vector and associated data in the database.
func (vg *Vecgo[T]) Update(ctx context.Context, id uint32, item VectorWithData[T]) error {
	start := time.Now()
	if vg.coordinator == nil {
		return fmt.Errorf("vecgo: coordinator not initialized (internal error - use builder API)")
	}
	err := translateError(vg.coordinator.Update(ctx, id, item.Vector, item.Data, item.Metadata))
	duration := time.Since(start)
	vg.metrics.RecordUpdate(duration, err)
	vg.logger.LogUpdate(ctx, id, err)
	return err
}

// SearchResult represents a search result.
type SearchResult[T any] struct {
	index.SearchResult

	// Data is the associated data of the search result.
	Data T

	// Metadata is the associated metadata of the search result (may be nil).
	Metadata metadata.Metadata
}

// FilterFunc is a function type used for filtering search results.
type FilterFunc func(id uint32) bool

// KNNSearchOptions contains options for KNN search.
type KNNSearchOptions struct {
	// EF (Explore Factor) specifies the size of the dynamic list for the nearest neighbors during the search.
	// Higher EF leads to more accurate but slower search.
	// EF cannot be set lower than the number of queried nearest neighbors (k).
	// The value of EF can be anything between k and the size of the dataset.
	// Default: 0 (use index default, typically HNSW.EF=200 for >95% recall)
	EF int

	// FilterFunc is a function used to filter search results.
	FilterFunc FilterFunc
}

// KNNSearch performs a K-nearest neighbor search.
func (vg *Vecgo[T]) KNNSearch(ctx context.Context, query []float32, k int, optFns ...func(o *KNNSearchOptions)) ([]SearchResult[T], error) {
	start := time.Now()
	opts := KNNSearchOptions{
		EF:         0, // 0 means use index default (HNSW.EF)
		FilterFunc: func(id uint32) bool { return true },
	}

	for _, fn := range optFns {
		fn(&opts)
	}

	// EF=0 means use index default, only validate if explicitly set
	if opts.EF > 0 && opts.EF < k {
		err := ErrInvalidEFValue
		vg.metrics.RecordSearch(k, time.Since(start), err)
		vg.logger.LogSearch(ctx, k, 0, err)
		return nil, err
	}

	// Create SearchOptions for the coordinator
	searchOpts := &index.SearchOptions{
		EFSearch: opts.EF,
		Filter: func(id uint32) bool {
			return opts.FilterFunc(id)
		},
	}

	bestCandidates, err := vg.coordinator.KNNSearch(ctx, query, k, searchOpts)
	if err != nil {
		err = translateError(err)
		vg.metrics.RecordSearch(k, time.Since(start), err)
		vg.logger.LogSearch(ctx, k, 0, err)
		return nil, err
	}

	results := vg.extractSearchResults(bestCandidates)
	duration := time.Since(start)
	vg.metrics.RecordSearch(k, duration, nil)
	vg.logger.LogSearch(ctx, k, len(results), nil)
	return results, nil
}

// KNNSearchStream returns an iterator over K-nearest neighbor search results.
// Results are yielded in order from nearest to farthest.
// The iterator supports early termination - stop iterating to cancel.
//
// This is more memory-efficient than KNNSearch for large result sets and
// allows processing results as they become available.
//
// Example:
//
//	for result, err := range db.KNNSearchStream(ctx, query, 100) {
//	    if err != nil {
//	        log.Fatal(err)
//	    }
//	    if result.Distance > threshold {
//	        break // Early termination
//	    }
//	    process(result)
//	}
func (vg *Vecgo[T]) KNNSearchStream(ctx context.Context, query []float32, k int, optFns ...func(o *KNNSearchOptions)) iter.Seq2[SearchResult[T], error] {
	return func(yield func(SearchResult[T], error) bool) {
		start := time.Now()
		opts := KNNSearchOptions{
			EF:         0, // 0 means use index default (HNSW.EF)
			FilterFunc: func(id uint32) bool { return true },
		}

		for _, fn := range optFns {
			fn(&opts)
		}

		// EF=0 means use index default, only validate if explicitly set
		if opts.EF > 0 && opts.EF < k {
			yield(SearchResult[T]{}, ErrInvalidEFValue)
			vg.metrics.RecordSearch(k, time.Since(start), ErrInvalidEFValue)
			vg.logger.LogSearch(ctx, k, 0, ErrInvalidEFValue)
			return
		}

		// Create SearchOptions for the coordinator
		searchOpts := &index.SearchOptions{
			EFSearch: opts.EF,
			Filter: func(id uint32) bool {
				return opts.FilterFunc(id)
			},
		}

		// Use the streaming interface from the index
		var count int
		for indexResult, err := range vg.index.KNNSearchStream(ctx, query, k, searchOpts) {
			if err != nil {
				err = translateError(err)
				vg.metrics.RecordSearch(k, time.Since(start), err)
				vg.logger.LogSearch(ctx, k, count, err)
				yield(SearchResult[T]{}, err)
				return
			}

			// Enrich with data and metadata
			result := SearchResult[T]{
				SearchResult: indexResult,
			}

			if data, ok := vg.store.Get(indexResult.ID); ok {
				result.Data = data
			}

			if meta, ok := vg.metaStore.Get(indexResult.ID); ok {
				result.Metadata = meta
			}

			count++
			if !yield(result, nil) {
				// Early termination
				vg.metrics.RecordSearch(k, time.Since(start), nil)
				vg.logger.LogSearch(ctx, k, count, nil)
				return
			}
		}

		vg.metrics.RecordSearch(k, time.Since(start), nil)
		vg.logger.LogSearch(ctx, k, count, nil)
	}
}

// BruteSearchOptions contains options for brute-force search.
type BruteSearchOptions struct {
	// FilterFunc is a function used to filter search results.
	FilterFunc FilterFunc
}

// BruteSearch performs a brute-force search.
func (vg *Vecgo[T]) BruteSearch(ctx context.Context, query []float32, k int, optFns ...func(o *BruteSearchOptions)) ([]SearchResult[T], error) {
	opts := BruteSearchOptions{
		FilterFunc: func(id uint32) bool { return true },
	}

	for _, fn := range optFns {
		fn(&opts)
	}

	bestCandidates, err := vg.coordinator.BruteSearch(ctx, query, k, func(id uint32) bool {
		return opts.FilterFunc(id)
	})
	if err != nil {
		return nil, translateError(err)
	}

	return vg.extractSearchResults(bestCandidates), nil
}

// HybridSearchOptions contains options for hybrid search (vector + metadata).
type HybridSearchOptions struct {
	// EF (Explore Factor) for HNSW search
	EF int

	// MetadataFilters are metadata conditions that must all match (AND logic)
	MetadataFilters *metadata.FilterSet

	// PreFilter applies metadata filtering before vector search (more efficient but may reduce recall)
	// PostFilter applies metadata filtering after vector search (maintains recall but less efficient)
	PreFilter bool
}

// HybridSearch performs a hybrid search combining vector similarity and metadata filtering.
// This enables queries like "find similar vectors where category=electronics AND price>100".
func (vg *Vecgo[T]) HybridSearch(ctx context.Context, query []float32, k int, optFns ...func(o *HybridSearchOptions)) ([]SearchResult[T], error) {
	opts := HybridSearchOptions{
		EF:              50,
		MetadataFilters: nil,
		PreFilter:       false, // Default to post-filtering for correctness/recall
	}

	for _, fn := range optFns {
		fn(&opts)
	}

	if opts.EF < k {
		return nil, ErrInvalidEFValue
	}

	// If no metadata filters, fall back to regular KNN search
	if opts.MetadataFilters == nil || len(opts.MetadataFilters.Filters) == 0 {
		return vg.KNNSearch(ctx, query, k, func(o *KNNSearchOptions) {
			o.EF = opts.EF
		})
	}

	// Create metadata filter function (may be accelerated by the inverted index).
	metadataFilter := func(id uint32) bool {
		meta, ok := vg.metaStore.Get(id)
		if !ok {
			return false // No metadata = doesn't match
		}
		return opts.MetadataFilters.Matches(meta)
	}
	if vg.metaStore != nil {
		// Try to compile filter for fast bitmap-based filtering
		metadataFilter = vg.metaStore.CreateFilterFunc(opts.MetadataFilters)
	}

	if opts.PreFilter {
		// Pre-filtering: apply metadata filter during vector search
		searchOpts := &index.SearchOptions{EFSearch: opts.EF, Filter: metadataFilter}
		bestCandidates, err := vg.index.KNNSearch(ctx, query, k, searchOpts)
		if err != nil {
			return nil, translateError(err)
		}
		return vg.extractSearchResults(bestCandidates), nil
	}

	// Post-filtering: get more results, then filter
	// Request more results to account for filtering
	oversampleK := min(k*3, 1000)

	searchOpts := &index.SearchOptions{EFSearch: opts.EF, Filter: func(id uint32) bool {
		return true // No pre-filtering
	}}
	bestCandidates, err := vg.index.KNNSearch(ctx, query, oversampleK, searchOpts)
	if err != nil {
		return nil, translateError(err)
	}

	// Apply metadata filtering and take top k
	results := make([]SearchResult[T], 0, k)
	for _, item := range bestCandidates {
		if len(results) >= k {
			break
		}

		data, ok := vg.store.Get(item.ID)
		if !ok {
			continue
		}

		// Apply metadata filter
		if metadataFilter(item.ID) {
			// Get metadata
			meta, _ := vg.metaStore.Get(item.ID)

			results = append(results, SearchResult[T]{
				SearchResult: index.SearchResult{
					ID:       item.ID,
					Distance: item.Distance,
				},
				Data:     data,
				Metadata: meta,
			})
		}
	}

	return results, nil
}

// extractSearchResults extracts search results from a priority queue.
func (vg *Vecgo[T]) extractSearchResults(bestCandidates []index.SearchResult) []SearchResult[T] {
	result := make([]SearchResult[T], 0, len(bestCandidates))

	for _, item := range bestCandidates {
		data, ok := vg.store.Get(item.ID)
		if !ok {
			// Skip if data not found (index and store might be out of sync)
			continue
		}

		// Get metadata if available
		meta, _ := vg.metaStore.Get(item.ID)

		result = append(result, SearchResult[T]{
			SearchResult: index.SearchResult{
				ID:       item.ID,
				Distance: item.Distance,
			},
			Data:     data,
			Metadata: meta,
		})
	}

	return result
}

// SaveToWriter saves the Vecgo database to an io.Writer.
// Uses a sectioned snapshot container: header + sections + directory + footer.
func (vg *Vecgo[T]) SaveToWriter(w io.Writer) error {
	// Convert UnifiedIndex to map store for snapshot saving
	metadataStore := engine.NewMapStore[metadata.Metadata]()
	for id, doc := range vg.metaStore.ToMap() {
		_ = metadataStore.Set(id, doc)
	}
	return translateError(engine.SaveToWriter(w, vg.index, vg.store, metadataStore, vg.codecOrDefault()))
}

// SaveToFile saves the Vecgo database to a file.
// If WAL is enabled, this will also create a checkpoint.
func (vg *Vecgo[T]) SaveToFile(filename string) error {
	// Convert UnifiedIndex to map store for snapshot saving
	metadataStore := engine.NewMapStore[metadata.Metadata]()
	for id, doc := range vg.metaStore.ToMap() {
		_ = metadataStore.Set(id, doc)
	}
	return translateError(engine.SaveToFile(filename, vg.index, vg.store, metadataStore, vg.wal, vg.codecOrDefault()))
}

// Stats returns statistics about the underlying index.
func (vg *Vecgo[T]) Stats() index.Stats {
	return vg.index.Stats()
}

func (vg *Vecgo[T]) codecOrDefault() codec.Codec {
	if vg.codec == nil {
		return codec.Default
	}
	return vg.codec
}

// RecoverFromWAL replays the write-ahead log to recover from a crash.
// This should be called after creating a Vecgo instance and enabling WAL
// but before any other operations.
func (vg *Vecgo[T]) RecoverFromWAL(ctx context.Context) error {
	if vg.wal == nil {
		return fmt.Errorf("vecgo: WAL not enabled (use WithWAL option)")
	}
	return engine.RecoverFromWAL(ctx, vg.index, vg.store, vg.metaStore, vg.wal, vg.codecOrDefault())
}
