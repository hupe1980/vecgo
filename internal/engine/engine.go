package engine

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"iter"
	"log/slog"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/hupe1980/vecgo/blobstore"
	"github.com/hupe1980/vecgo/distance"
	"github.com/hupe1980/vecgo/internal/cache"
	"github.com/hupe1980/vecgo/internal/fs"
	"github.com/hupe1980/vecgo/internal/manifest"
	imetadata "github.com/hupe1980/vecgo/internal/metadata"
	"github.com/hupe1980/vecgo/internal/pk"
	"github.com/hupe1980/vecgo/internal/quantization"
	"github.com/hupe1980/vecgo/internal/resource"
	"github.com/hupe1980/vecgo/internal/searcher"
	"github.com/hupe1980/vecgo/internal/segment/diskann"
	"github.com/hupe1980/vecgo/internal/segment/flat"
	"github.com/hupe1980/vecgo/internal/segment/memtable"
	"github.com/hupe1980/vecgo/lexical"
	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/model"
)

// VacuumStats holds results of the vacuum operation.
type VacuumStats struct {
	ManifestsDeleted int
	SegmentsDeleted  int
	BytesReclaimed   int64
}

// RetentionPolicy defines rules for retaining old versions.
type RetentionPolicy struct {
	// KeepVersions is the minimum number of recent versions to keep.
	KeepVersions int
	// KeepDuration is the minimum duration of history to keep.
	KeepDuration time.Duration
}

// Engine is the main entry point for the vector database.
type Engine struct {
	mu               sync.RWMutex
	dir              string
	store            blobstore.BlobStore
	manifestStore    blobstore.BlobStore
	dim              int
	metric           distance.Metric
	quantizationType quantization.Type

	manifest *manifest.Manifest
	pkIndex  *pk.Index
	lsn      atomic.Uint64
	// nextID is the next auto-increment ID to assign.
	nextID atomic.Uint64

	current atomic.Pointer[Snapshot]

	targetVersion   uint64    // If > 0, engine is read-only at this version
	targetTimestamp time.Time // If !IsZero, engine resolves closest version (read-only)
	retentionPolicy RetentionPolicy

	tombstonesMu sync.RWMutex
	tombstones   map[model.SegmentID]*VersionedTombstones

	policy       CompactionPolicy
	compactionCh chan struct{}
	closeCh      chan struct{}
	wg           sync.WaitGroup

	metrics MetricsObserver

	resourceController *resource.Controller
	blockCache         cache.BlockCache
	blockCacheSize     int64
	// blockCacheBlockSize is the size of blocks in the block cache.
	// Larger blocks (e.g. 1MB) are better for high-latency stores like S3.
	blockCacheBlockSize int64

	diskCache          cache.BlockCache
	diskCacheDir       string
	diskCacheSize      int64
	diskCacheBlockSize int64

	compactionConfig CompactionConfig
	flushConfig      FlushConfig
	flushCh          chan struct{}
	closed           atomic.Bool

	lexicalIndex lexical.Index
	lexicalField string

	isCloud   bool
	readOnly  bool // Read-only mode: no writes, purely stateless
	dimSet    bool
	metricSet bool

	schema metadata.Schema

	fs     fs.FileSystem
	logger *slog.Logger

	watermarkFilterPool sync.Pool
	tombstoneFilterPool sync.Pool

	ctx    context.Context
	cancel context.CancelFunc
}

// FlushConfig holds configuration for automatic flushing.
type FlushConfig struct {
	// MaxMemTableSize is the maximum size of the MemTable in bytes before a flush is triggered.
	// If 0, defaults to 64MB.
	MaxMemTableSize int64
}

// CompactionConfig holds configuration for compaction.
type CompactionConfig struct {
	// DiskANNThreshold is the number of rows above which DiskANN is used.
	// If 0, defaults to 10000.
	DiskANNThreshold int

	// FlatQuantizationType is the quantization type for Flat segments.
	// 0=None, 1=SQ8, 2=PQ.
	FlatQuantizationType int

	// DiskANNOptions are the options for DiskANN segments.
	DiskANNOptions diskann.Options
}

// Option defines a configuration option for the Engine.
type Option func(*Engine)

// WithLogger sets the logger for the engine.
func WithLogger(l *slog.Logger) Option {
	return func(e *Engine) {
		e.logger = l
	}
}

// WithResourceController sets the resource controller for the engine.
func WithResourceController(rc *resource.Controller) Option {
	return func(e *Engine) {
		e.resourceController = rc
	}
}

// WithMemoryLimit sets the memory limit for the engine in bytes.
// If set to 0, memory is unlimited.
func WithMemoryLimit(bytes int64) Option {
	return func(e *Engine) {
		e.resourceController = resource.NewController(resource.Config{
			MemoryLimitBytes: bytes,
		})
	}
}

// WithMetricsObserver sets the metrics observer for the engine.
func WithMetricsObserver(observer MetricsObserver) Option {
	return func(e *Engine) {
		e.metrics = observer
	}
}

// WithCompactionPolicy sets the compaction policy used by the background loop.
// If unset, the engine uses a default size-tiered policy.
func WithCompactionPolicy(policy CompactionPolicy) Option {
	return func(e *Engine) {
		if policy != nil {
			e.policy = policy
		}
	}
}

// WithCompactionThreshold sets the segment-count threshold for the default
// size-tiered compaction policy.
func WithCompactionThreshold(threshold int) Option {
	return func(e *Engine) {
		if threshold > 0 {
			e.policy = &BoundedSizeTieredPolicy{Threshold: threshold}
		}
	}
}

// WithDiskANNThreshold sets the number of vectors required to build a DiskANN segment.
// Segments smaller than this will use a Flat (IVF) index.
func WithDiskANNThreshold(threshold int) Option {
	return func(e *Engine) {
		e.compactionConfig.DiskANNThreshold = threshold
	}
}

// WithLexicalIndex sets the lexical index and the metadata field to index.
func WithLexicalIndex(idx lexical.Index, field string) Option {
	return func(e *Engine) {
		e.lexicalIndex = idx
		e.lexicalField = field
	}
}

// WithSchema sets the metadata schema for the engine.
func WithSchema(schema metadata.Schema) Option {
	return func(e *Engine) {
		e.schema = schema
	}
}

// WithCompactionConfig sets the compaction configuration.
func WithCompactionConfig(cfg CompactionConfig) Option {
	return func(e *Engine) {
		e.compactionConfig = cfg
	}
}

// WithFlushConfig sets the flush configuration.
func WithFlushConfig(cfg FlushConfig) Option {
	return func(e *Engine) {
		e.flushConfig = cfg
	}
}

// WithFileSystem sets the file system for the engine.
// This is primarily used for testing and fault injection.
func WithFileSystem(fs fs.FileSystem) Option {
	return func(e *Engine) {
		e.fs = fs
	}
}

// WithBlobStore is deprecated. Use Open with WithRemoteStore instead.
// To keep backward compatibility for now, it behaves as before but
// it is better to let the engine manage the store stack (Remote -> Disk Cache -> Mem Cache).
func WithBlobStore(st blobstore.BlobStore) Option {
	return func(e *Engine) {
		if st != nil {
			e.store = st
		}
	}
}

// WithManifestStore sets the store used for the manifest file.
// If unset, defaults to local file system.
// This is useful for stateless compute nodes that need to read the manifest from S3.
func WithManifestStore(st blobstore.BlobStore) Option {
	return func(e *Engine) {
		e.manifestStore = st
	}
}

// WithBlockCacheSize sets the size of the block cache in bytes.
// If 0, defaults to 256MB.
func WithBlockCacheSize(size int64) Option {
	return func(e *Engine) {
		e.blockCacheSize = size
	}
}

// WithBlockCacheBlockSize sets the size of blocks in the block cache.
// Defaults to 4KB (4096 bytes). For S3/Cloud stores, use 1MB+.
func WithBlockCacheBlockSize(size int64) Option {
	return func(e *Engine) {
		e.blockCacheBlockSize = size
	}
}

// WithDiskCache enables a secondary disk-based cache.
// dir: directory to store cache files.
// size: maximum size in bytes.
// blockSize: size of blocks on disk (e.g. 1MB or 4MB).
func WithDiskCache(dir string, size, blockSize int64) Option {
	return func(e *Engine) {
		e.diskCacheDir = dir
		e.diskCacheSize = size
		e.diskCacheBlockSize = blockSize
	}
}

// WithRemoteStore configures the engine to run in "Stateless/Cloud" mode.
// The provided store is treated as the source of truth for all persistent data (Manifest, Segments, etc.).
// The local directory passed to Open() is used purely for caching and temporary files.
//
// Deprecated: Use OpenRemote(store, opts...) instead.
func WithRemoteStore(st blobstore.BlobStore) Option {
	return func(e *Engine) {
		// In Cloud Mode:
		// 1. The remote store is the source of truth (Manifest + Segments)
		e.manifestStore = st

		// 2. The main data store starts as the remote store.
		//    Open() will wrap this with DiskCache(localDir) and BlockCache(RAM).
		e.store = st

		// 3. Mark the engine as stateless/cloud-backed so we can apply defaults later
		e.isCloud = true
	}
}

// WithCacheDir sets the local directory for caching remote data.
// Only applicable when opening a remote store.
func WithCacheDir(dir string) Option {
	return func(e *Engine) {
		e.dir = dir
	}
}

// WithVersion opens the database at a specific version ID (Time Travel).
// The engine will be in Read-Only mode.
func WithVersion(v uint64) Option {
	return func(e *Engine) {
		e.targetVersion = v
		e.readOnly = true
	}
}

// WithQuantization sets the vector quantization method (e.g. PQ, SQ8, INT4).
// This applies to new segments created during flush/compaction.
func WithQuantization(t quantization.Type) Option {
	return func(e *Engine) {
		e.quantizationType = t
	}
}

// WithTimestamp opens the database at the state closest to the given time (Time Travel).
// The engine will be in Read-Only mode.
func WithTimestamp(t time.Time) Option {
	return func(e *Engine) {
		e.targetTimestamp = t
		e.readOnly = true
	}
}

// WithRetentionPolicy sets the retention policy for vacuuming/compaction.
func WithRetentionPolicy(p RetentionPolicy) Option {
	return func(e *Engine) {
		e.retentionPolicy = p
	}
}

// ReadOnly puts the engine in read-only mode.
// In this mode:
//   - No writes, purely stateless
//   - Insert/Delete operations return ErrReadOnly
//   - No local state is required (pure memory cache)
//   - Ideal for stateless serverless search nodes
//
// This follows LanceDB's model for cloud deployments where the remote
// store (S3) is the source of truth and search nodes are stateless.
func ReadOnly() Option {
	return func(e *Engine) {
		e.readOnly = true
	}
}

// WithDimension sets the vector dimension for the engine.
// Required when creating a new index.
func WithDimension(dim int) Option {
	return func(e *Engine) {
		e.dim = dim
		e.dimSet = true
	}
}

// WithMetric sets the distance metric for the engine.
// Required when creating a new index.
func WithMetric(m distance.Metric) Option {
	return func(e *Engine) {
		e.metric = m
		e.metricSet = true
	}
}

// Open opens or creates a new Engine (backward-compatible signature).
// For new code, prefer OpenLocal or the unified vecgo.Open().
func Open(dir string, dim int, metric distance.Metric, opts ...Option) (*Engine, error) {
	// Combine explicit dim/metric with opts
	allOpts := append([]Option{WithDimension(dim), WithMetric(metric)}, opts...)
	return OpenLocal(dir, allOpts...)
}

// OpenLocal opens or creates an Engine using local storage.
// If dim/metric are not provided via options, they are loaded from an existing manifest.
func OpenLocal(dir string, opts ...Option) (*Engine, error) {
	ctx, cancel := context.WithCancel(context.Background())

	e := &Engine{
		ctx:          ctx,
		cancel:       cancel,
		dir:          dir,
		policy:       &BoundedSizeTieredPolicy{Threshold: 4},
		compactionCh: make(chan struct{}, 1),
		flushCh:      make(chan struct{}, 1),
		closeCh:      make(chan struct{}),
		metrics:      &NoopMetricsObserver{},
	}

	for _, opt := range opts {
		opt(e)
	}

	// Local mode: use local store if not overridden
	if e.store == nil {
		e.store = blobstore.NewLocalStore(e.dir)
	}

	return e.init()
}

// OpenRemote opens an Engine backed by a remote store (e.g., S3, GCS).
// The store is the source of truth. A local cache directory is auto-created if not specified.
// Dim/metric are loaded from the remote manifest (use WithDimension/WithMetric for new indexes).
func OpenRemote(store blobstore.BlobStore, opts ...Option) (*Engine, error) {
	ctx, cancel := context.WithCancel(context.Background())

	e := &Engine{
		ctx:           ctx,
		cancel:        cancel,
		policy:        &BoundedSizeTieredPolicy{Threshold: 4},
		compactionCh:  make(chan struct{}, 1),
		flushCh:       make(chan struct{}, 1),
		closeCh:       make(chan struct{}),
		metrics:       &NoopMetricsObserver{},
		isCloud:       true,
		store:         store,
		manifestStore: store,
	}

	for _, opt := range opts {
		opt(e)
	}

	// Auto-create cache directory if not specified
	if e.dir == "" {
		tmpDir, err := os.MkdirTemp("", "vecgo-cache-*")
		if err != nil {
			cancel()
			return nil, fmt.Errorf("failed to create cache directory: %w", err)
		}
		e.dir = tmpDir
	}

	return e.init()
}

// OpenCloud opens an Engine pointing to a remote store (e.g., S3).
// The local `dir` is used for caching. Dimension and Metric are read from the manifest.
//
// Deprecated: Use OpenRemote(store, WithCacheDir(dir)) instead.
func OpenCloud(dir string, opts ...Option) (*Engine, error) {
	ctx, cancel := context.WithCancel(context.Background())

	e := &Engine{
		ctx:          ctx,
		cancel:       cancel,
		dir:          dir,
		policy:       &BoundedSizeTieredPolicy{Threshold: 4},
		compactionCh: make(chan struct{}, 1),
		flushCh:      make(chan struct{}, 1),
		closeCh:      make(chan struct{}),
		metrics:      &NoopMetricsObserver{},
	}

	for _, opt := range opts {
		opt(e)
	}

	// Cloud mode requires WithRemoteStore to have been set
	if e.store == nil {
		return nil, fmt.Errorf("OpenCloud requires WithRemoteStore option")
	}

	return e.init()
}

// init completes engine initialization (shared between Open and OpenCloud)
func (e *Engine) init() (*Engine, error) {
	if e.isCloud {
		// Cloud Mode Defaults
		if e.blockCacheBlockSize == 0 {
			e.blockCacheBlockSize = 4 * 1024 * 1024 // 4MB for Cloud
		}

		// Auto-configure Disk Cache at 'dir' if not explicitly set
		if e.diskCacheSize == 0 {
			e.diskCacheSize = 10 * 1024 * 1024 * 1024 // 10GB default
		}
		e.diskCacheDir = e.dir
		if e.diskCacheBlockSize == 0 {
			e.diskCacheBlockSize = e.blockCacheBlockSize // Match fetch size
		}
	} else {
		// Local Mode Defaults
		if e.blockCacheBlockSize == 0 {
			e.blockCacheBlockSize = 4 * 1024 // 4KB for Local
		}
	}

	if e.resourceController == nil {
		e.resourceController = resource.NewController(resource.Config{
			MemoryLimitBytes: 1 << 30, // 1GB default
		})
	}
	if e.blockCacheSize == 0 {
		e.blockCacheSize = 256 << 20 // 256MB default
	}

	if e.diskCacheDir != "" && e.diskCacheSize > 0 {
		dcConf := cache.DiskCacheConfig{
			RootDir:      e.diskCacheDir,
			MaxSizeBytes: e.diskCacheSize,
		}
		dc, err := cache.NewDiskBlockCache(dcConf)
		if err != nil {
			return nil, fmt.Errorf("failed to create disk cache: %w", err)
		}
		e.diskCache = dc
		// Wrap the base store with Disk Cache
		// Logic: Reads hit DiskCache first. If miss, fetch from e.store (which is Remote in Cloud mode, or Local in Local mode)
		// Wait, in local mode, the base store IS the disk. Caching disk on disk?
		// No, in local mode, e.store is LocalStore. Cloning files to a cache dir is redundant.
		// DiskCache usually only makes sense for RemoteStore.
		if e.isCloud {
			e.store = blobstore.NewCachingStore(e.store, dc, e.diskCacheBlockSize)
		}
	}

	e.blockCache = cache.NewLRUBlockCache(e.blockCacheSize, e.resourceController)
	e.store = blobstore.NewCachingStore(e.store, e.blockCache, e.blockCacheBlockSize)

	if e.flushConfig.MaxMemTableSize == 0 {
		e.flushConfig.MaxMemTableSize = 64 * 1024 * 1024 // 64MB
	}

	if e.fs == nil {
		e.fs = fs.LocalFS{}
	}

	if e.manifestStore == nil {
		// Default to LocalStore in data directory
		e.manifestStore = blobstore.NewLocalStore(e.dir)
	}

	// Ensure dir exists
	if err := e.fs.MkdirAll(e.dir, 0755); err != nil {
		return nil, err
	}

	// 1. Load Manifest
	mStore := manifest.NewStore(e.manifestStore)
	var m *manifest.Manifest
	var err error

	if e.targetVersion > 0 {
		m, err = mStore.LoadVersion(e.targetVersion)
		if err != nil {
			return nil, fmt.Errorf("failed to load version %d: %w", e.targetVersion, err)
		}
	} else if !e.targetTimestamp.IsZero() {
		// Time Travel by Timestamp
		versions, lErr := mStore.ListVersions(e.ctx)
		if lErr != nil {
			return nil, fmt.Errorf("failed to list versions for timestamp lookup: %w", lErr)
		}
		slices.SortFunc(versions, func(a, b *manifest.Manifest) int {
			// Sort ID Descending (Newest First)
			return cmp.Compare(b.ID, a.ID)
		})

		if e.logger != nil {
			e.logger.Info("TimeTravel resolving version", "target", e.targetTimestamp.Format(time.RFC3339Nano), "candidates", len(versions))
		}

		var best *manifest.Manifest
		for _, v := range versions {
			// Find the first (newest) version that is created at or before target timestamp
			if !v.CreatedAt.After(e.targetTimestamp) {
				best = v
				break
			}
		}
		if best == nil {
			return nil, fmt.Errorf("no version found at or before %v", e.targetTimestamp)
		}
		m = best
		if e.logger != nil {
			e.logger.Info("TimeTravel selected version", "id", m.ID, "created", m.CreatedAt.Format(time.RFC3339Nano), "segments", len(m.Segments))
		}
	} else {
		m, err = mStore.Load()
	}

	if err != nil {
		if errors.Is(err, manifest.ErrNotFound) {
			// Initialize new Manifest
			if !e.dimSet || !e.metricSet {
				return nil, fmt.Errorf("creating new index requires WithDimension and WithMetric options")
			}
			m = manifest.New(e.dim, e.metric.String())
		} else if errors.Is(err, manifest.ErrIncompatibleVersion) {
			return nil, fmt.Errorf("%w: %v", ErrIncompatibleFormat, err)
		} else {
			return nil, fmt.Errorf("failed to load manifest: %w", err)
		}
	} else {
		// Loaded existing manifest, use its properties
		if !e.dimSet {
			e.dim = m.Dim
		} else if e.dim != m.Dim {
			return nil, fmt.Errorf("dimension mismatch: requested %d, found %d", e.dim, m.Dim)
		}

		loadedMetricString := m.Metric
		// Convert string metric to distance.Metric enum logic if needed,
		// but currently distance.Metric is int.
		// If manifest stores string "L2", "Cosine", etc. we need parsing.
		// Since I changed manifest.New to take string, I should parse it back.
		// For now simple check:
		if !e.metricSet {
			// Parse metric string to enum
			switch loadedMetricString {
			case "L2":
				e.metric = distance.MetricL2
			case "Cosine":
				e.metric = distance.MetricCosine
			case "Dot":
				e.metric = distance.MetricDot
			default:
				// Fallback or error? Assume L2 if unknown/empty for now or handle better
				if loadedMetricString == "" {
					e.metric = distance.MetricL2
				} else {
					return nil, fmt.Errorf("unknown metric in manifest: %s", loadedMetricString)
				}
			}
		} else {
			// Verify match
			if e.metric.String() != loadedMetricString {
				return nil, fmt.Errorf("metric mismatch: requested %s, found %s", e.metric.String(), loadedMetricString)
			}
		}
	}

	// Clean up orphan segments (crash recovery)
	// Skip for read-only engines - they're loading a historical snapshot and
	// must not delete segments that exist in newer versions
	if !e.readOnly {
		validSegIDs := make(map[uint64]struct{})
		for _, segMeta := range m.Segments {
			validSegIDs[uint64(segMeta.ID)] = struct{}{}
		}
		if entries, err := e.fs.ReadDir(e.dir); err == nil {
			for _, entry := range entries {
				name := entry.Name()
				if strings.HasPrefix(name, "segment_") {
					parts := strings.Split(name, ".")
					if len(parts) >= 2 {
						var id uint64
						if n, _ := fmt.Sscanf(parts[0], "segment_%d", &id); n == 1 {
							if _, ok := validSegIDs[id]; !ok {
								_ = e.fs.Remove(filepath.Join(e.dir, name))
							}
						}
					}
				}
			}
		}
	}

	// 2. Open Segments and Tombstones (and Rebuild PK Index)
	segments := make(map[model.SegmentID]*RefCountedSegment)
	e.tombstones = make(map[model.SegmentID]*VersionedTombstones)

	// Recover PK Index
	pkIdx := pk.New()
	var maxID uint64 = 0

	// Try loading valid PK checkpoint
	pkLoaded := false
	if m.PKIndex.Path != "" {
		pkPath := filepath.Join(e.dir, m.PKIndex.Path)
		if f, err := e.fs.OpenFile(pkPath, os.O_RDONLY, 0); err == nil {
			if err := pkIdx.Load(f); err == nil {
				pkLoaded = true
				if e.logger != nil {
					e.logger.Info("Loaded PK Index from checkpoint", "path", m.PKIndex.Path)
				}
				maxID = pkIdx.MaxID()
			} else {
				if e.logger != nil {
					e.logger.Warn("Failed to load PK Index checkpoint, rebuilding", "path", m.PKIndex.Path, "error", err)
				}
			}
			_ = f.Close() // Intentionally ignore: cleanup path
		}
	}

	for _, segMeta := range m.Segments {
		// Optional payload file.
		payloadPath := fmt.Sprintf("segment_%d.payload", segMeta.ID)
		var payloadBlob blobstore.Blob
		if b, err := e.store.Open(context.Background(), payloadPath); err == nil {
			payloadBlob = b
		} else if !errors.Is(err, blobstore.ErrNotFound) {
			return nil, fmt.Errorf("failed to open payload blob for segment %d: %w", segMeta.ID, err)
		}

		seg, err := openSegment(context.Background(), e.store, segMeta.Path, e.blockCache, payloadBlob)
		if err != nil {
			if errors.Is(err, flat.ErrInvalidVersion) || errors.Is(err, flat.ErrInvalidMagic) {
				return nil, fmt.Errorf("%w: segment %d: %v", ErrIncompatibleFormat, segMeta.ID, err)
			}
			return nil, fmt.Errorf("failed to open segment %d: %w", segMeta.ID, err)
		}
		segments[segMeta.ID] = NewRefCountedSegment(seg)

		vt := NewVersionedTombstones(int(seg.RowCount()))
		// Only load tombstones for writable engines.
		// Read-only (time-travel) engines show historical state WITHOUT later deletes.
		// The tombstone file format (bitmap) doesn't preserve LSN information,
		// so we can't correctly filter tombstones by version.
		if !e.readOnly {
			tombPath := filepath.Join(e.dir, fmt.Sprintf("segment_%d.tomb", segMeta.ID))
			if f, err := e.fs.OpenFile(tombPath, os.O_RDONLY, 0); err == nil {
				bm := imetadata.NewLocalBitmap()
				if _, err := bm.ReadFrom(f); err == nil {
					vt.LoadFromBitmap(bm)
				}
				_ = f.Close() // Intentionally ignore: cleanup path
			}
		}
		e.tombstones[segMeta.ID] = vt

		// Populate ID Index if not loaded from checkpoint
		if !pkLoaded {
			rowCount := int(seg.RowCount())
			batchSize := 1024
			for i := 0; i < rowCount; i += batchSize {
				end := i + batchSize
				if end > rowCount {
					end = rowCount
				}
				rows := make([]uint32, end-i)
				for j := 0; j < len(rows); j++ {
					rows[j] = uint32(i + j)
				}

				// Fetch IDs
				ids := make([]model.ID, len(rows))
				if err := seg.FetchIDs(context.Background(), rows, ids); err != nil {
					// Fallback or error? If FetchIDs is not implemented by legacy segments, this fails.
					// But we are assuming compatible segments or new segments.
					return nil, fmt.Errorf("failed to fetch ids from segment %d: %w", segMeta.ID, err)
				}

				// Initialize with an LSN < RecoveryLSN. 0 is fine for base state.
				baseLSN := uint64(0)
				for j, id := range ids {
					pkIdx.Upsert(id, model.Location{
						SegmentID: segMeta.ID,
						RowID:     model.RowID(rows[j]),
					}, baseLSN)
					if uint64(id) > maxID {
						maxID = uint64(id)
					}
				}
			}
		}
	}

	// 3. Initialize Active MemTable
	var mt *memtable.MemTable
	if !e.readOnly {
		activeID := m.NextSegmentID
		m.NextSegmentID++

		var err error
		mt, err = memtable.New(activeID, e.dim, e.metric, e.resourceController)
		if err != nil {
			return nil, fmt.Errorf("failed to create memtable: %w", err)
		}
		// Initialize tombstones for active segment if not exists
		e.tombstonesMu.Lock()
		if _, ok := e.tombstones[activeID]; !ok {
			e.tombstones[activeID] = NewVersionedTombstones(1024)
		}
		e.tombstonesMu.Unlock()
	} else {
		// In Read-Only mode (Time Travel), we still need a valid Snapshot structure.
		// Snapshot expects an Active MemTable, but it will be empty and never written to.
		// Wait, if it's read-only, we shouldn't allocate a real MemTable or increment segment ID?
		// Existing logic: `e.readOnly` is set if loading historic version.
		// If we are looking at historic version, we definitely should NOT modify manifest NextSegmentID.
		// Create a dummy MemTable? Or allow nil in Snapshot?
		// `NewSnapshot` calls `active.RowCount()`.
		// Let's check `memtable.New` signature.
		// It creates structures.
		// We can create a dummy MemTable with valid ID (e.g. 0 or max uint64) but ensure it's never flushed.
		// Or handle nil active in Snapshot. Snapshot defines active as `*memtable.MemTable`.
		// Let's create a dummy one for now to satisfy the non-nil invariance,
		// but using a safe ID (e.g. m.NextSegmentID but DO NOT increment manifest).
		var err error
		mt, err = memtable.New(m.NextSegmentID, e.dim, e.metric, e.resourceController)
		if err != nil {
			return nil, fmt.Errorf("failed to create dummy memtable: %w", err)
		}
	}

	// 4. (Merged with 2) Segments are already loaded in 'segments' map.

	// 5. Initial Snapshot
	e.pkIndex = pkIdx
	e.manifest = m
	e.lsn.Store(m.MaxLSN)
	if e.logger != nil {
		e.logger.Info("Init loaded segments", "count", len(segments), "manifest_len", len(m.Segments), "manifest_id", m.ID)
	}

	snap := NewSnapshot(mt, m.MaxLSN)
	snap.segments = segments
	snap.RebuildSorted()

	e.current.Store(snap)

	if !e.readOnly {
		e.nextID.Store(maxID)
		e.wg.Add(2)
		GoSafe(e.runFlushLoop)
		GoSafe(e.runCompactionLoop)
	}

	return e, nil
}

// loadSegment opens a segment from disk.
func (e *Engine) loadSegment(sInfo manifest.SegmentInfo) (*flat.Segment, error) {
	// Open main segment file
	blob, err := e.store.Open(context.Background(), sInfo.Path)
	if err != nil {
		return nil, fmt.Errorf("failed to open segment blob: %w", err)
	}

	// Open optional payload
	var payloadBlob blobstore.Blob
	payloadPath := fmt.Sprintf("segment_%d.payload", sInfo.ID)
	if pb, err := e.store.Open(context.Background(), payloadPath); err == nil {
		payloadBlob = pb
	} else if !errors.Is(err, blobstore.ErrNotFound) {
		return nil, fmt.Errorf("failed to open payload blob: %w", err)
	}

	opts := []flat.Option{
		flat.WithBlockCache(e.blockCache),
		flat.WithPayloadBlob(payloadBlob),
	}

	// In the future: e.blockCache might need to be passed down.
	// The flat package might accept cache options.

	seg, err := flat.Open(blob, opts...)
	if err != nil {
		if errors.Is(err, flat.ErrInvalidVersion) || errors.Is(err, flat.ErrInvalidMagic) {
			return nil, fmt.Errorf("%w: segment %d: %v", ErrIncompatibleFormat, sInfo.ID, err)
		}
		return nil, err
	}
	return seg, nil
}

func (e *Engine) validateVector(vec []float32) error {
	if len(vec) != e.dim {
		return fmt.Errorf("%w: dimension mismatch: expected %d, got %d", ErrInvalidArgument, e.dim, len(vec))
	}
	for i, v := range vec {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return fmt.Errorf("%w: vector contains NaN or Inf at index %d", ErrInvalidArgument, i)
		}
	}
	return nil
}

// loadSnapshot safely loads the current snapshot with an incremented reference count.
// It handles race conditions where the snapshot might be destroyed during load.
func (e *Engine) loadSnapshot() (*Snapshot, error) {
	for {
		// Fast path check if closed
		if e.closed.Load() {
			return nil, ErrClosed
		}

		snap := e.current.Load()
		if snap == nil {
			// Should only happen if not initialized, but check closed again
			if e.closed.Load() {
				return nil, ErrClosed
			}
			return nil, errors.New("engine not initialized")
		}

		if snap.TryIncRef() {
			return snap, nil
		}

		// Snapshot was concurrently destroyed (refs reached 0).
		// Wait for the writer to install the new snapshot (which must happen before destruction).
		runtime.Gosched()
	}
}

func (e *Engine) Insert(vec []float32, md metadata.Document, payload []byte) (id model.ID, err error) {
	start := time.Now()
	defer func() {
		e.metrics.OnInsert(time.Since(start), err)
	}()

	// Read-only mode check (LanceDB-style stateless deployments)
	if e.readOnly {
		return 0, ErrReadOnly
	}

	if err := e.validateVector(vec); err != nil {
		return 0, err
	}

	if e.schema != nil {
		if err := e.schema.Validate(md); err != nil {
			return 0, fmt.Errorf("%w: %v", ErrInvalidArgument, err)
		}
	}

	e.mu.RLock()
	if e.closed.Load() {
		e.mu.RUnlock()
		return 0, ErrClosed
	}

	// Generate ID
	id = model.ID(e.nextID.Add(1))

	snap := e.current.Load()

	// Get LSN for ordering
	lsn := e.lsn.Add(1)

	// WAL write removed: In pure commit-oriented mode, we do NOT write to WAL.
	// Durability is achieved via explicit Commit() or periodic Flush().
	// If the process crashes before flush, data in MemTable is lost.

	// Insert into Active MemTable
	rowID, err := snap.active.InsertWithPayload(id, vec, md, payload)
	if err != nil {
		e.mu.RUnlock()
		return 0, err
	}

	// Update Global PK Index
	e.pkIndex.Upsert(id, model.Location{
		SegmentID: snap.active.ID(),
		RowID:     rowID,
	}, lsn)

	// Update Lexical Index
	if e.lexicalIndex != nil && e.lexicalField != "" && md != nil {
		if val, ok := md[e.lexicalField]; ok {
			if str := val.StringValue(); str != "" {
				if err := e.lexicalIndex.Add(id, str); err != nil {
					if e.logger != nil {
						e.logger.Error("failed to update lexical index", "id", id, "error", err)
					}
				}
			}
		}
	}

	// Check triggers
	memSize := snap.active.Size()
	shouldFlush := memSize > e.flushConfig.MaxMemTableSize

	if e.flushConfig.MaxMemTableSize > 0 {
		e.metrics.OnMemTableStatus(memSize, float64(memSize)/float64(e.flushConfig.MaxMemTableSize))
	}

	e.mu.RUnlock()

	if shouldFlush {
		select {
		case e.flushCh <- struct{}{}:
		default:
		}
	}

	return id, nil
}

// BatchInsert adds multiple vectors to the engine.
// It amortizes locking overhead.
func (e *Engine) BatchInsert(vectors [][]float32, mds []metadata.Document, payloads [][]byte) ([]model.ID, error) {
	n := len(vectors)
	if n == 0 {
		return nil, nil
	}
	if mds != nil && len(mds) != n {
		return nil, fmt.Errorf("metadata count mismatch: expected %d, got %d", n, len(mds))
	}
	if payloads != nil && len(payloads) != n {
		return nil, fmt.Errorf("payload count mismatch: expected %d, got %d", n, len(payloads))
	}

	// Validate all first
	for i, vec := range vectors {
		if len(vec) != e.dim {
			return nil, fmt.Errorf("record %d: %w: expected %d, got %d", i, ErrInvalidArgument, e.dim, len(vec))
		}
		if e.schema != nil && mds != nil && mds[i] != nil {
			if err := e.schema.Validate(mds[i]); err != nil {
				return nil, fmt.Errorf("record %d: %w: %v", i, ErrInvalidArgument, err)
			}
		}
	}

	e.mu.RLock()
	if e.closed.Load() {
		e.mu.RUnlock()
		return nil, ErrClosed
	}

	snap := e.current.Load()
	ids := make([]model.ID, n)

	// Prepare batch
	type batchItem struct {
		id      model.ID
		lsn     uint64
		md      metadata.Document
		payload []byte
	}
	items := make([]batchItem, n)

	// Batch allocate LSNs and IDs (single atomic op each instead of N)
	lsnStart := e.lsn.Add(uint64(n)) - uint64(n) + 1   // Reserve n LSNs, get first one
	idStart := e.nextID.Add(uint64(n)) - uint64(n) + 1 // Reserve n IDs, get first one

	// Phase 1: Prepare batch items (no atomic ops in loop)
	for i := range vectors {
		// Assign pre-allocated ID and LSN
		id := model.ID(idStart + uint64(i))
		ids[i] = id

		// Serialize metadata
		var md metadata.Document
		if mds != nil {
			md = mds[i]
		}

		var payload []byte
		if payloads != nil {
			payload = payloads[i]
		}

		items[i] = batchItem{
			id:      id,
			lsn:     lsnStart + uint64(i),
			md:      md,
			payload: payload,
		}
	}

	// Phase 3: Update MemTable & Indexes in Parallel
	// Parallelize ingestion to maximize CPU utilization for HNSW construction.

	concurrency := runtime.GOMAXPROCS(0)
	if n < concurrency {
		concurrency = n
	}
	sem := make(chan struct{}, concurrency)
	var wg sync.WaitGroup
	var errMu sync.Mutex
	var firstErr error

	for i, item := range items {
		wg.Add(1)
		sem <- struct{}{}
		go func(i int, item batchItem) {
			defer wg.Done()
			defer func() { <-sem }()

			// Stop processing if an error occurred
			errMu.Lock()
			if firstErr != nil {
				errMu.Unlock()
				return
			}
			errMu.Unlock()

			// Insert into Active MemTable
			rowID, err := snap.active.InsertWithPayload(item.id, vectors[i], item.md, item.payload)
			if err != nil {
				errMu.Lock()
				if firstErr == nil {
					firstErr = err
				}
				errMu.Unlock()
				return
			}

			// Update PK Index
			oldLoc, exists := e.pkIndex.Upsert(item.id, model.Location{
				SegmentID: snap.active.ID(),
				RowID:     rowID,
			}, item.lsn)

			if exists {
				if vt, ok := e.tombstones[oldLoc.SegmentID]; ok {
					vt.MarkDeleted(uint32(oldLoc.RowID), item.lsn)
				}
			}

			// Update Lexical Index
			if e.lexicalIndex != nil && e.lexicalField != "" && item.md != nil {
				if val, ok := item.md[e.lexicalField]; ok {
					if str := val.StringValue(); str != "" {
						if err := e.lexicalIndex.Add(item.id, str); err != nil {
							if e.logger != nil {
								e.logger.Error("failed to update lexical index", "id", item.id, "error", err)
							}
						}
					}
				}
			}
		}(i, item)
	}

	wg.Wait()

	if firstErr != nil {
		e.mu.RUnlock()
		return nil, firstErr
	}

	// Check triggers
	memSize := snap.active.Size()
	shouldFlush := memSize > e.flushConfig.MaxMemTableSize

	e.mu.RUnlock()

	if shouldFlush {
		select {
		case e.flushCh <- struct{}{}:
		default:
		}
	}

	return ids, nil
}

// BatchInsertDeferred adds multiple vectors WITHOUT indexing (Bulk Load).
// This is significantly faster (~30x) but vectors will NOT be searchable via HNSW
// until flushed to disk. They are persisted safely on Commit().
//
// OPTIMIZATION: This uses sequential processing to avoid goroutine overhead.
// For bulk load, the bottleneck is memory allocation and data copying, not CPU.
// Parallel goroutines with semaphores add ~20% overhead from channel ops.
func (e *Engine) BatchInsertDeferred(vectors [][]float32, mds []metadata.Document, payloads [][]byte) ([]model.ID, error) {
	n := len(vectors)
	if n == 0 {
		return nil, nil
	}
	if mds != nil && len(mds) != n {
		return nil, fmt.Errorf("metadata count mismatch: expected %d, got %d", n, len(mds))
	}
	if payloads != nil && len(payloads) != n {
		return nil, fmt.Errorf("payload count mismatch: expected %d, got %d", n, len(payloads))
	}

	// Validate all vectors first (fail fast)
	for i, vec := range vectors {
		if len(vec) != e.dim {
			return nil, fmt.Errorf("record %d: %w: expected %d, got %d", i, ErrInvalidArgument, e.dim, len(vec))
		}
	}

	// Validate metadata schema if present
	if e.schema != nil && mds != nil {
		for i, md := range mds {
			if md != nil {
				if err := e.schema.Validate(md); err != nil {
					return nil, fmt.Errorf("record %d: %w: %v", i, ErrInvalidArgument, err)
				}
			}
		}
	}

	e.mu.RLock()
	if e.closed.Load() {
		e.mu.RUnlock()
		return nil, ErrClosed
	}

	snap := e.current.Load()
	ids := make([]model.ID, n)

	// Batch allocate LSNs and IDs (single atomic op each instead of N)
	lsnStart := e.lsn.Add(uint64(n)) - uint64(n) + 1   // Reserve n LSNs, get first one
	idStart := e.nextID.Add(uint64(n)) - uint64(n) + 1 // Reserve n IDs, get first one

	// Shard-aware batching:
	// 1. Group items by shard (id % 16) - single pass O(n)
	// 2. Process each shard's batch in parallel (16 goroutines max)
	// 3. Each shard processes its items sequentially (no lock contention within shard)
	//
	// This is optimal because:
	// - Minimizes goroutine overhead (16 goroutines vs N)
	// - Maximizes parallelism across shards
	// - No lock contention within shard (sequential within each shard goroutine)

	const shardCount = 16
	type shardItem struct {
		idx int // Original index in vectors slice
		id  model.ID
		lsn uint64
	}
	shardBatches := make([][]shardItem, shardCount)

	// Pre-allocate shard batches (assume uniform distribution)
	avgPerShard := (n + shardCount - 1) / shardCount
	for i := 0; i < shardCount; i++ {
		shardBatches[i] = make([]shardItem, 0, avgPerShard)
	}

	// Group items by shard (single pass)
	for i := 0; i < n; i++ {
		id := model.ID(idStart + uint64(i))
		ids[i] = id
		shardIdx := id & (shardCount - 1)
		shardBatches[shardIdx] = append(shardBatches[shardIdx], shardItem{
			idx: i,
			id:  id,
			lsn: lsnStart + uint64(i),
		})
	}

	// Process all shards in parallel
	var wg sync.WaitGroup
	var errMu sync.Mutex
	var firstErr error

	for shardIdx := 0; shardIdx < shardCount; shardIdx++ {
		batch := shardBatches[shardIdx]
		if len(batch) == 0 {
			continue
		}

		wg.Add(1)
		go func(batch []shardItem) {
			defer wg.Done()

			// Check for early exit
			errMu.Lock()
			if firstErr != nil {
				errMu.Unlock()
				return
			}
			errMu.Unlock()

			// Process this shard's items sequentially (no lock contention)
			for _, item := range batch {
				var md metadata.Document
				if mds != nil {
					md = mds[item.idx]
				}

				var payload []byte
				if payloads != nil {
					payload = payloads[item.idx]
				}

				// Insert into Active MemTable using DEFERRED mode (no graph)
				rowID, err := snap.active.InsertDeferred(item.id, vectors[item.idx], md, payload)
				if err != nil {
					errMu.Lock()
					if firstErr == nil {
						firstErr = err
					}
					errMu.Unlock()
					return
				}

				// Update PK Index (concurrent-safe)
				oldLoc, exists := e.pkIndex.Upsert(item.id, model.Location{
					SegmentID: snap.active.ID(),
					RowID:     rowID,
				}, item.lsn)

				if exists {
					e.tombstonesMu.RLock()
					if vt, ok := e.tombstones[oldLoc.SegmentID]; ok {
						vt.MarkDeleted(uint32(oldLoc.RowID), item.lsn)
					}
					e.tombstonesMu.RUnlock()
				}

				// Update Lexical Index (skip if not configured - common for bulk load)
				if e.lexicalIndex != nil && e.lexicalField != "" && md != nil {
					if val, ok := md[e.lexicalField]; ok {
						if str := val.StringValue(); str != "" {
							if err := e.lexicalIndex.Add(item.id, str); err != nil {
								if e.logger != nil {
									e.logger.Error("failed to update lexical index", "id", item.id, "error", err)
								}
							}
						}
					}
				}
			}
		}(batch)
	}

	wg.Wait()

	if firstErr != nil {
		e.mu.RUnlock()
		return nil, firstErr
	}

	// Check triggers
	memSize := snap.active.Size()
	shouldFlush := memSize > e.flushConfig.MaxMemTableSize

	e.mu.RUnlock()

	if shouldFlush {
		select {
		case e.flushCh <- struct{}{}:
		default:
		}
	}

	return ids, nil
}

// Delete removes a vector from the engine.
func (e *Engine) Delete(id model.ID) (err error) {
	start := time.Now()
	defer func() {
		e.metrics.OnDelete(time.Since(start), err)
	}()

	// Read-only mode check
	if e.readOnly {
		return ErrReadOnly
	}

	e.mu.Lock()
	if e.closed.Load() {
		e.mu.Unlock()
		return ErrClosed
	}

	// 1. Check PK Index at current LSN
	// We use the current LSN to check visibility before we create a new deletion LSN.
	currentLSN := e.lsn.Load()
	oldLoc, exists := e.pkIndex.Get(id, currentLSN)
	if !exists {
		e.mu.Unlock()
		return nil // Idempotent
	}

	// Check if already deleted
	// This optimization avoids unnecessary WAL writes
	e.tombstonesMu.RLock()
	if vt, ok := e.tombstones[oldLoc.SegmentID]; ok {
		if vt.IsDeleted(uint32(oldLoc.RowID), currentLSN) {
			e.tombstonesMu.RUnlock()
			e.mu.Unlock()
			return nil
		}
	}
	e.tombstonesMu.RUnlock()

	// Get LSN for ordering
	lsn := e.lsn.Add(1)

	// Mark Deleted in Tombstones
	e.tombstonesMu.RLock()
	if vt, ok := e.tombstones[oldLoc.SegmentID]; ok {
		vt.MarkDeleted(uint32(oldLoc.RowID), lsn)
	}
	e.tombstonesMu.RUnlock()

	// Update PK Index
	e.pkIndex.Delete(id, lsn)

	// Update Lexical Index
	if e.lexicalIndex != nil {
		if err := e.lexicalIndex.Delete(id); err != nil {
			if e.logger != nil {
				e.logger.Error("failed to delete from lexical index", "id", id, "error", err)
			}
		}
	}

	e.mu.Unlock()

	return nil
}

// BatchDelete removes multiple vectors from the engine in a single operation.
// It is atomic and more efficient than calling Delete in a loop.
func (e *Engine) BatchDelete(ids []model.ID) error {
	e.mu.RLock()
	if e.closed.Load() {
		e.mu.RUnlock()
		return ErrClosed
	}

	for _, id := range ids {
		lsn := e.lsn.Add(1)

		// Check Global PK Index (Optimization)
		if _, exists := e.pkIndex.Get(id, lsn); !exists {
			continue // Already deleted or doesn't exist
		}

		// Update Global PK Index & Tombstones
		oldLoc, existed := e.pkIndex.Delete(id, lsn)
		if existed {
			if vt, ok := e.tombstones[oldLoc.SegmentID]; ok {
				vt.MarkDeleted(uint32(oldLoc.RowID), lsn)
			}
		}

		// Update Lexical Index
		if e.lexicalIndex != nil {
			if err := e.lexicalIndex.Delete(id); err != nil {
				if e.logger != nil {
					e.logger.Error("failed to delete from lexical index", "id", id, "error", err)
				}
			}
		}
	}

	e.mu.RUnlock()

	return nil
}

// BatchSearch performs multiple k-NN searches in parallel.
func (e *Engine) BatchSearch(ctx context.Context, queries [][]float32, k int, opts ...func(*model.SearchOptions)) ([][]model.Candidate, error) {
	if k <= 0 {
		return nil, fmt.Errorf("%w: k must be > 0", ErrInvalidArgument)
	}
	for i, q := range queries {
		if len(q) != e.dim {
			return nil, fmt.Errorf("query %d: %w: expected %d, got %d", i, ErrInvalidArgument, e.dim, len(q))
		}
	}

	results := make([][]model.Candidate, len(queries))
	var wg sync.WaitGroup
	var errOnce sync.Once
	var firstErr error

	// Limit concurrency to avoid exploding goroutines
	sem := make(chan struct{}, 100) // Max 100 concurrent searches

	for i, q := range queries {
		wg.Add(1)
		go func(i int, q []float32) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			next := e.SearchIter(ctx, q, k, opts...)
			var batch []model.Candidate
			for c, err := range next {
				if err != nil {
					errOnce.Do(func() {
						firstErr = err
					})
					return
				}
				batch = append(batch, c)
			}
			results[i] = batch
		}(i, q)
	}

	wg.Wait()
	if firstErr != nil {
		return nil, firstErr
	}
	return results, nil
}

// ScanOption configures a Scan operation.
type ScanOption func(*ScanConfig)

// ScanConfig holds configuration for Scan.
type ScanConfig struct {
	BatchSize int
	Filter    *metadata.Filter
}

// WithScanBatchSize sets the batch size for prefetching (hint).
func WithScanBatchSize(n int) ScanOption {
	return func(c *ScanConfig) {
		c.BatchSize = n
	}
}

// WithScanFilter sets a metadata filter for the scan.
func WithScanFilter(f *metadata.Filter) ScanOption {
	return func(c *ScanConfig) {
		c.Filter = f
	}
}

// Scan returns an iterator over all records matching the filter.
// Uses Go 1.23+ iter.Seq2 for best-in-class ergonomics and performance.
func (e *Engine) Scan(ctx context.Context, opts ...ScanOption) iter.Seq2[*model.Record, error] {
	cfg := ScanConfig{
		BatchSize: 100,
	}
	for _, opt := range opts {
		opt(&cfg)
	}

	return func(yield func(*model.Record, error) bool) {
		snap, err := e.loadSnapshot()
		if err != nil {
			yield(nil, err)
			return
		}
		defer snap.DecRef()

		for id, loc := range e.pkIndex.Scan(e.lsn.Load()) {
			if ctx.Err() != nil {
				yield(nil, ctx.Err())
				return
			}

			// Skip empty slots in the persistent index
			if loc.SegmentID == 0 {
				continue
			}

			// Check tombstones (Redundant if PK Index is consistent, but safe)
			if vt, ok := e.tombstones[loc.SegmentID]; ok {
				if vt.IsDeleted(uint32(loc.RowID), e.lsn.Load()) {
					continue
				}
			}

			var rec *model.Record

			if loc.SegmentID == snap.active.ID() {
				batch, err := snap.active.Fetch(context.Background(), []uint32{uint32(loc.RowID)}, []string{"vector", "metadata", "payload"})
				if err != nil {
					if !yield(nil, err) {
						return
					}
					return
				}
				rec = &model.Record{
					ID:       id,
					Vector:   batch.Vector(0),
					Metadata: batch.Metadata(0),
					Payload:  batch.Payload(0),
				}
			} else {
				seg, ok := snap.segments[loc.SegmentID]
				if !ok {
					yield(nil, fmt.Errorf("segment %d not found in snapshot", loc.SegmentID))
					return
				}
				batch, err := seg.Fetch(context.Background(), []uint32{uint32(loc.RowID)}, []string{"vector", "metadata", "payload"})
				if err != nil {
					if !yield(nil, err) {
						return
					}
					return
				}
				rec = &model.Record{
					ID:       id,
					Vector:   batch.Vector(0),
					Metadata: batch.Metadata(0),
					Payload:  batch.Payload(0),
				}
			}

			// Ensure ID is set
			rec.ID = id

			if cfg.Filter != nil {
				if !cfg.Filter.Matches(rec.Metadata) {
					continue
				}
			}

			if !yield(rec, nil) {
				return
			}
		}
	}
}

// SearchThreshold returns all vectors within the given distance threshold.
// It uses maxResults to bound the search.
func (e *Engine) SearchThreshold(ctx context.Context, q []float32, threshold float32, maxResults int, opts ...func(*model.SearchOptions)) ([]model.Candidate, error) {
	// We use Search with k=maxResults to find candidates, then filter by threshold.
	// Note: This assumes that the nearest neighbors are the ones within the threshold.
	// For L2, smaller score is better. For Cosine/Dot, larger score is better (usually).
	// Vecgo uses:
	// L2: score = distance (smaller is better)
	// Cosine: score = 1 - cosine_similarity (smaller is better)
	// Dot: score = -dot_product (smaller is better)
	// Wait, let's check distance metric implementation.

	// If metric is L2, we want score <= threshold.
	// If metric is Cosine, we want score <= threshold (if score is distance).
	// If metric is Dot, we want score <= threshold (if score is negative dot).

	// Let's check distance package or assume standard behavior where "Score" is "Distance".
	// In engine.go Search:
	// descending: e.metric != distance.MetricL2
	// If L2, descending=false (Ascending). Smaller is better.
	// If Dot/Cosine, descending=true. Larger is better?
	// Wait, if descending=true, then heap pops the SMALLEST value (min-heap of scores).
	// If we want top-k largest scores, we use a min-heap of size k. The top is the smallest of the top-k.
	// If we find a new candidate with score > top, we replace top.
	// So for Dot/Cosine, larger score is better.

	// So:
	// L2: Score is distance. We want Score <= threshold.
	// Dot/Cosine: Score is similarity. We want Score >= threshold.

	cands, err := e.Search(ctx, q, maxResults, opts...)
	if err != nil {
		return nil, err
	}

	var filtered []model.Candidate
	for _, c := range cands {
		keep := false
		if e.metric == distance.MetricL2 {
			keep = c.Score <= threshold
		} else {
			keep = c.Score >= threshold
		}

		if keep {
			filtered = append(filtered, c)
		}
	}
	return filtered, nil
}

// HybridSearch performs a combination of vector search and lexical search.
// It uses Reciprocal Rank Fusion (RRF) to combine the results.
// rrfK is the constant k in the RRF formula: score = 1 / (k + rank).
// Typically rrfK is 60.
func (e *Engine) HybridSearch(ctx context.Context, q []float32, textQuery string, k int, rrfK int, opts ...func(*model.SearchOptions)) ([]model.Candidate, error) {
	if e.lexicalIndex == nil {
		return nil, fmt.Errorf("lexical index not configured")
	}

	// 1. Vector Search
	// We fetch more candidates to increase overlap
	vectorK := k * 2
	if vectorK < 50 {
		vectorK = 50
	}
	vecResults, err := e.Search(ctx, q, vectorK, opts...)
	if err != nil {
		return nil, fmt.Errorf("vector search failed: %w", err)
	}

	// 2. Lexical Search
	lexResults, err := e.lexicalIndex.Search(textQuery, vectorK)
	if err != nil {
		return nil, fmt.Errorf("lexical search failed: %w", err)
	}

	// 3. RRF Fusion
	s := searcher.Get()
	defer searcher.Put(s)

	finalScores := s.ScratchMap
	// clear(finalScores) // Reset() calls clear()

	// Process Vector Results
	for rank, c := range vecResults {
		score := 1.0 / float32(rrfK+rank+1)
		finalScores[c.ID] = score
	}

	// Process Lexical Results
	for rank, c := range lexResults {
		score := 1.0 / float32(rrfK+rank+1)
		finalScores[c.ID] += score
	}

	// 4. Convert to Candidates and Sort
	candidates := s.Results[:0]
	for id, score := range finalScores {
		candidates = append(candidates, model.Candidate{
			ID:    id,
			Score: score,
		})
	}

	// Sort by RRF score descending
	slices.SortFunc(candidates, func(a, b model.Candidate) int {
		if a.Score > b.Score {
			return -1
		}
		if a.Score < b.Score {
			return 1
		}
		return 0
	})

	// 5. Top K
	if len(candidates) > k {
		candidates = candidates[:k]
	}

	// Copy to safe buffer to avoid race conditions when searcher is reused
	safeCandidates := make([]model.Candidate, len(candidates))
	copy(safeCandidates, candidates)
	candidates = safeCandidates

	// Note: The returned candidates have RRF scores, not distance/similarity.
	// We might want to fetch vectors if needed, but Search returns candidates with PKs.
	// If the user needs vectors, they can call Get(pk).
	// However, Search usually returns Location.
	// We need to fill Location for these candidates if possible.
	// But RRF candidates might come from Lexical only, so we don't know Location easily without lookup.
	// We can do a lookup in PK Index.

	// Acquire Snapshot for PK lookup
	snap, err := e.loadSnapshot()
	if err != nil {
		return nil, err
	}
	defer snap.DecRef()

	validCount := 0
	for _, cand := range candidates {
		if loc, ok := e.pkIndex.Get(cand.ID, e.lsn.Load()); ok {
			cand.Loc = loc
			candidates[validCount] = cand
			validCount++
		}
	}
	candidates = candidates[:validCount]

	return candidates, nil
}

// Get returns the full record (vector, metadata, payload) for the given primary key.
func (e *Engine) Get(id model.ID) (rec *model.Record, err error) {
	start := time.Now()
	defer func() {
		e.metrics.OnGet(time.Since(start), err)
	}()
	select {
	case <-e.closeCh:
		return nil, ErrClosed
	default:
	}

	snap, err := e.loadSnapshot()
	if err != nil {
		return nil, err
	}
	defer snap.DecRef()

	// Lookup ID
	loc, ok := e.pkIndex.Get(id, e.lsn.Load())
	if !ok {
		return nil, fmt.Errorf("%w: id %d", ErrNotFound, id)
	}

	// Check if in active memtable
	if loc.SegmentID == snap.active.ID() {
		// MemTable fetch
		batch, err := snap.active.Fetch(context.Background(), []uint32{uint32(loc.RowID)}, []string{"vector", "metadata", "payload"})
		if err != nil {
			return nil, err
		}
		return &model.Record{
			ID:       id,
			Vector:   batch.Vector(0),
			Metadata: batch.Metadata(0),
			Payload:  batch.Payload(0),
		}, nil
	}

	// Check segments
	seg, ok := snap.segments[loc.SegmentID]
	if !ok {
		return nil, fmt.Errorf("segment %d not found for id %d", loc.SegmentID, id)
	}

	batch, err := seg.Fetch(context.Background(), []uint32{uint32(loc.RowID)}, []string{"vector", "metadata", "payload"})
	if err != nil {
		return nil, err
	}
	return &model.Record{
		ID:       id,
		Vector:   batch.Vector(0),
		Metadata: batch.Metadata(0),
		Payload:  batch.Payload(0),
	}, nil
}

// Commit flushes the in-memory buffer to a durable immutable segment.
//
// This is the durability boundary for NoWAL mode (the default).
// After Commit() returns successfully, all previously inserted/deleted
// data is guaranteed to survive crashes.
//
// In WAL mode, Commit() is equivalent to Flush().
//
// Usage pattern:
//
//	db.Insert(vec1, meta1, payload1)
//	db.Insert(vec2, meta2, payload2)  // Buffered, not yet durable
//	db.Commit()                        // Now durable
//
// Commit is safe to call concurrently. It blocks until the flush completes.
// If the buffer is empty, Commit returns immediately with no error.
func (e *Engine) Commit() error {
	return e.Flush()
}

// Flush flushes the active MemTable to disk.
func (e *Engine) Flush() (err error) {
	start := time.Now()
	var itemsFlushed int
	var bytesFlushed uint64

	defer func() {
		e.metrics.OnFlush(time.Since(start), itemsFlushed, bytesFlushed, err)
	}()

	// --- Phase 1: Rotate (Holding Lock) ---
	e.mu.Lock()

	currentLSN := e.lsn.Load()
	snap := e.current.Load()
	if snap.active.RowCount() == 0 {
		e.mu.Unlock()
		return nil
	}

	active := snap.active
	rowCount := int(active.RowCount())
	itemsFlushed = rowCount
	activeID := active.ID()

	if e.logger != nil {
		e.logger.Info("Flush started", "segmentID", activeID, "rowCount", rowCount)
	}

	// Create New Snapshot
	newSnap := snap.Clone()

	// Move active to "frozen" (segments map)
	// Wrap active in RefCountedSegment
	frozenSeg := NewRefCountedSegment(active)
	// Ensure we handle concurrent access correctly:
	// The frozen MemTable is now effectively read-only for Inserts (since active is new),
	// but it is valid for Search/Fetch via the segments map.
	newSnap.segments[activeID] = frozenSeg

	// Update sorted segments
	newSnap.sortedSegments = append(newSnap.sortedSegments, frozenSeg)
	slices.SortFunc(newSnap.sortedSegments, func(a, b *RefCountedSegment) int {
		return cmp.Compare(a.ID(), b.ID())
	})

	// Create New Active MemTable
	newActiveID := e.manifest.NextSegmentID
	e.manifest.NextSegmentID++
	newSnap.active, err = memtable.New(newActiveID, e.dim, e.metric, e.resourceController)
	if err != nil {
		e.mu.Unlock()
		// Try to restore old WAL? Very bad state.
		return fmt.Errorf("failed to create new memtable: %w", err)
	}
	e.tombstonesMu.Lock()
	e.tombstones[newActiveID] = NewVersionedTombstones(1024)
	e.tombstonesMu.Unlock()
	newSnap.activeWatermark = 0

	// Persist tombstones to ensure durability of deletes covered by the old WAL.
	if err := e.persistTombstones(); err != nil {
		e.mu.Unlock()
		return err
	}

	// Publish Rotation
	newSnap.RebuildSorted() // Ensure consistent state
	e.current.Store(newSnap)
	snap.DecRef()

	e.mu.Unlock()
	// --- End of Phase 1 (Inserts can proceed) ---

	// --- Phase 2: Write (No Lock) ---
	// Data Paths
	filename := fmt.Sprintf("segment_%d.bin", activeID)
	path := filepath.Join(e.dir, filename)
	tmpPath := path + ".tmp"

	payloadFilename := fmt.Sprintf("segment_%d.payload", activeID)
	payloadPath := filepath.Join(e.dir, payloadFilename)
	payloadTmpPath := payloadPath + ".tmp"

	f, err := e.fs.OpenFile(tmpPath, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0644)
	if err != nil {
		return err
	}
	defer func() {
		if f != nil {
			_ = f.Close() // Intentionally ignore: cleanup path
			_ = e.fs.Remove(tmpPath)
		}
	}()

	payloadF, err := e.fs.OpenFile(payloadTmpPath, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0644)
	if err != nil {
		return err
	}
	defer func() {
		if payloadF != nil {
			_ = payloadF.Close() // Intentionally ignore: cleanup path
			_ = e.fs.Remove(payloadTmpPath)
		}
	}()

	w := flat.NewWriter(f, payloadF, activeID, e.dim, e.metric, 0, flat.QuantizationNone)

	// Collect moves for PK update
	type flushMove struct {
		id     model.ID
		oldRow uint32
		newRow uint32
	}
	moves := make([]flushMove, 0, 1024)
	var count uint32

	// Write data
	err = active.Iterate(func(rowID uint32, id model.ID, vec []float32, md metadata.Document, payload []byte) error {
		if err := w.Add(id, vec, md, payload); err != nil {
			return err
		}

		moves = append(moves, flushMove{id: id, oldRow: rowID, newRow: count})
		count++
		return nil
	})
	if err != nil {
		return err
	}

	if err := w.Flush(); err != nil {
		return err
	}
	// Ensure durability
	if err := f.Sync(); err != nil {
		return err
	}
	if err := payloadF.Sync(); err != nil {
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	f = nil
	if err := payloadF.Close(); err != nil {
		return err
	}
	payloadF = nil

	// Publish atomically
	if err := e.fs.Rename(tmpPath, path); err != nil {
		return err
	}
	if info, _ := e.fs.Stat(path); info != nil {
		bytesFlushed += uint64(info.Size())
	}

	if err := e.fs.Rename(payloadTmpPath, payloadPath); err != nil {
		_ = e.fs.Remove(path)
		return err
	}
	if info, _ := e.fs.Stat(payloadPath); info != nil {
		bytesFlushed += uint64(info.Size())
	}

	if err := syncDir(e.fs, e.dir); err != nil {
		return err
	}

	// --- Phase 3: Commit (Holding Lock) ---
	e.mu.Lock()
	defer e.mu.Unlock()

	// Open new segment for reading
	blob, err := e.store.Open(context.Background(), filename)
	if err != nil {
		return err
	}

	opts := []flat.Option{flat.WithBlockCache(e.blockCache)}
	if payloadBlob, err := e.store.Open(context.Background(), payloadFilename); err == nil {
		opts = append(opts, flat.WithPayloadBlob(payloadBlob))
	}

	newSeg, err := flat.Open(blob, opts...)
	if err != nil {
		_ = e.fs.Remove(path)
		_ = e.fs.Remove(payloadPath)
		return err
	}

	if stat, err := e.fs.Stat(path); err == nil {
		e.metrics.OnThroughput("flush_write", stat.Size())
	}

	// Update Snapshot
	snap = e.current.Load()
	newSnap = snap.Clone()

	// Update Snapshot: Replace MemTable segment with Disk segment
	// ID stays the same (activeID), but underlying implementation changes.
	newSnap.segments[activeID] = NewRefCountedSegment(newSeg)

	// Update PK Index to point to new RowIDs
	flushLSN := e.lsn.Add(1)

	// Create mapping for tombstone migration
	oldToNew := make(map[uint32]uint32, len(moves))

	for _, m := range moves {
		oldToNew[m.oldRow] = m.newRow

		currentLoc, exists := e.pkIndex.Get(m.id, flushLSN)
		// Only update if PK still points to the segment we flushed (CAS-like check)
		if exists && currentLoc.SegmentID == activeID && currentLoc.RowID == model.RowID(m.oldRow) {
			e.pkIndex.Upsert(m.id, model.Location{
				SegmentID: activeID,
				RowID:     model.RowID(m.newRow),
			}, flushLSN)
		}
	}

	// Migrate Tombstones
	e.tombstonesMu.Lock()
	if ts, ok := e.tombstones[activeID]; ok {
		// Existing tombstone structure for activeID is effectively replaced/migrated
		// Since activeID is now a Flat segment (immutable rowIDs), we need a new Tombstone tracker?
		// No, `tombstones[activeID]` tracks deletes for `segment activeID`.
		// But the underlying segment CHANGED from MemTable (sharded) to Disk (linear).
		// RowIDs CHANGED.
		// So we must migrate the deletion bitmap.
		// We can reuse the `activeID` key, but we need a fresh Tombstone object derived from old.

		newTombstones := NewVersionedTombstones(1024)

		// Get all deleted rows
		snapBM := ts.ToBitmap(e.lsn.Load())
		for oldRowID := range snapBM.Iterator() {
			oldRow := uint32(oldRowID)
			if newRow, exists := oldToNew[oldRow]; exists {
				newTombstones.MarkDeleted(newRow, e.lsn.Load())
			}
		}
		// Replace
		e.tombstones[activeID] = newTombstones
	}
	e.tombstonesMu.Unlock()

	newSnap.RebuildSorted()
	newSnap.lsn = flushLSN

	e.current.Store(newSnap)
	snap.DecRef()

	// Update Manifest
	e.manifest.Segments = append(e.manifest.Segments, manifest.SegmentInfo{
		ID:       activeID,
		Level:    0, // L0
		RowCount: uint32(count),
		Path:     filename,
	})
	if e.logger != nil {
		e.logger.Info("Manifest updated", "total_segments", len(e.manifest.Segments))
	}

	e.manifest.MaxLSN = currentLSN

	mStore := manifest.NewStore(e.manifestStore)
	if err := mStore.Save(e.manifest); err != nil {
		return err
	}

	// Signal compaction
	select {
	case e.compactionCh <- struct{}{}:
	default:
	}

	if e.logger != nil {
		e.logger.Info("Flush completed", "duration", time.Since(start), "rowCount", rowCount)
	}

	return nil
}

// Vacuum cleans up old versions based on the retention policy.
// It deletes old manifest files and any segments that are no longer referenced by
// kept manifest versions.
func (e *Engine) Vacuum(ctx context.Context) error {
	// 1. Check if retention policy is enabled
	if e.retentionPolicy.KeepVersions == 0 && e.retentionPolicy.KeepDuration == 0 {
		return nil // No policy configured
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	start := time.Now()
	// Vacuum can be called manually even if no automatic policy.
	// Logging
	if e.logger != nil {
		e.logger.Info("Starting vacuum", "policy", e.retentionPolicy)
	}

	// 2. Load all available manifest versions
	mStore := manifest.NewStore(e.manifestStore)
	allVersions, err := mStore.ListVersions(ctx)
	if err != nil {
		return fmt.Errorf("failed to list versions: %w", err)
	}

	// Sort versions descending (newest first)
	slices.SortFunc(allVersions, func(a, b *manifest.Manifest) int {
		return b.CreatedAt.Compare(a.CreatedAt)
	})

	if len(allVersions) == 0 {
		return nil
	}
	latestVersion := allVersions[0]

	// 3. Determine which versions to keep
	type keepReason string
	const (
		ReasonLatest   keepReason = "latest"
		ReasonCount    keepReason = "count"
		ReasonDuration keepReason = "duration"
	)

	toKeep := make(map[uint64]keepReason)
	toKeep[latestVersion.ID] = ReasonLatest // Always keep current/latest

	// Apply KeepVersions (count based)
	if e.retentionPolicy.KeepVersions > 0 {
		keepCount := 0
		for _, m := range allVersions {
			if _, exists := toKeep[m.ID]; !exists {
				if keepCount < e.retentionPolicy.KeepVersions {
					toKeep[m.ID] = ReasonCount
					keepCount++
				}
			}
		}
	}

	// Apply KeepDuration (time based)
	if e.retentionPolicy.KeepDuration > 0 {
		cutoff := time.Now().Add(-e.retentionPolicy.KeepDuration)
		for _, m := range allVersions {
			if _, exists := toKeep[m.ID]; !exists {
				if m.CreatedAt.After(cutoff) {
					toKeep[m.ID] = ReasonDuration
				}
			}
		}
	}

	// 4. Identify Referenced Segments in KEPT versions
	referencedSegments := make(map[model.SegmentID]struct{})
	for _, m := range allVersions {
		if _, keeping := toKeep[m.ID]; keeping {
			for _, seg := range m.Segments {
				referencedSegments[seg.ID] = struct{}{}
			}
		}
	}

	// 5. Delete versions NOT in toKeep
	deletedVersions := 0
	deletedSegments := 0
	var reclaimedBytes int64

	for _, m := range allVersions {
		if _, keeping := toKeep[m.ID]; !keeping {
			// Delete Manifest File
			if err := mStore.DeleteVersion(ctx, m.ID); err != nil {
				e.logger.Warn("Failed to delete manifest version", "id", m.ID, "error", err)
			} else {
				deletedVersions++
			}
		}
	}

	// 6. Identify Orphaned Segments
	// Only delete segments that are NOT referenced by ANY kept version.
	// We need to list all segments on disk roughly?
	// Or we can infer candidates from the set of ALL versions we just loaded.
	// If a segment was present in a DELETED version and is NOT in `referencedSegments`, it is an orphan.

	// Collect all known segments from all loaded versions
	candidateSegments := make(map[model.SegmentID]struct{})
	for _, m := range allVersions {
		for _, seg := range m.Segments {
			candidateSegments[seg.ID] = struct{}{}
		}
	}

	for segID := range candidateSegments {
		if _, needed := referencedSegments[segID]; !needed {
			// Safe to delete
			// Delete segment data file
			dataFilename := fmt.Sprintf("segment_%d.bin", segID)
			dataPath := filepath.Join(e.dir, dataFilename)
			if info, err := e.fs.Stat(dataPath); err == nil {
				reclaimedBytes += info.Size()
				if err := e.fs.Remove(dataPath); err != nil {
					e.logger.Warn("Failed to delete segment file", "file", dataPath, "error", err)
				}
			}

			// Delete payload file
			payloadFilename := fmt.Sprintf("segment_%d.payload", segID)
			payloadPath := filepath.Join(e.dir, payloadFilename)
			if info, err := e.fs.Stat(payloadPath); err == nil {
				reclaimedBytes += info.Size()
				if err := e.fs.Remove(payloadPath); err != nil {
					e.logger.Warn("Failed to delete payload file", "file", payloadPath, "error", err)
				}
			}

			// Delete tombstone file?
			// Tombstones are managed separately in memory/checkpoint,
			// but if they are persisted as part of segment logic (e.g. bloom filters sidecar), handling needed.
			// Current arch: Tombstones are in memory map `e.tombstones` and flushed to... where?
			// `persistTombstones` writes to WAL or separate file?
			// The flush logic uses `NewVersionedTombstones(1024)`.
			// If persistent tombstones exist as files (unlikely in current code, they seem WAL-based or memory-only),
			// we skip.
			// Re-reading flush: `e.persistTombstones()` is called. Checking that.

			deletedSegments++
		}
	}

	if e.logger != nil {
		e.logger.Info("Vacuum completed",
			"duration", time.Since(start),
			"deletedVersions", deletedVersions,
			"deletedSegments", deletedSegments,
			"reclaimedBytes", reclaimedBytes,
		)
	}

	return nil
}

// EngineStats contains runtime statistics for the engine.
type EngineStats struct {
	SegmentCount     int
	RowCount         int
	TombstoneCount   int
	DiskUsageBytes   int64
	MemoryUsageBytes int64 // L0 + overhead
}

// Stats returns the current engine statistics.
func (e *Engine) Stats() EngineStats {
	// No lock needed:
	// - e.current is atomic
	// - segment/memtable methods are thread-safe

	snap, err := e.loadSnapshot()
	if err != nil {
		return EngineStats{}
	}
	defer snap.DecRef()

	stats := EngineStats{
		SegmentCount:     len(snap.segments),
		RowCount:         int(snap.active.RowCount()),
		MemoryUsageBytes: snap.active.Size(),
	}

	for _, seg := range snap.segments {
		stats.RowCount += int(seg.RowCount())
		stats.DiskUsageBytes += seg.Size()
	}

	e.mu.RLock()
	for _, vt := range e.tombstones {
		count := vt.Count(snap.lsn)
		stats.TombstoneCount += count
		stats.RowCount -= count // Subtract tombstones from row count
	}
	e.mu.RUnlock()

	return stats
}

// Close closes the engine.
func (e *Engine) Close() error {
	if !e.closed.CompareAndSwap(false, true) {
		return ErrClosed
	}

	e.cancel() // Cancel background contexts
	close(e.closeCh)

	// Wait for background tasks
	e.wg.Wait()

	snap := e.current.Load()
	if snap != nil {
		snap.DecRef() // Release Engine's reference
		if snap.active != nil {
			_ = snap.active.Close()
		}
	}

	// Save Tombstones
	if err := e.persistTombstones(); err != nil {
		return err
	}

	// Save PK Index (Checkpoint)
	if err := e.persistPKIndex(); err != nil {
		if e.logger != nil {
			e.logger.Error("failed to persist pk index", "error", err)
		}
	}

	if e.diskCache != nil {
		_ = e.diskCache.Close()
	}

	return nil
}

// CacheStats returns the combined statistics of the block caches.
func (e *Engine) CacheStats() (hits, misses int64) {
	if e.blockCache != nil {
		h, m := e.blockCache.Stats()
		hits += h
		misses += m
	}
	if e.diskCache != nil {
		h, m := e.diskCache.Stats()
		hits += h
		misses += m
	}
	return
}

// persistPKIndex saves the current PK index to disk and updates the manifest.
func (e *Engine) persistPKIndex() error {
	// Skip for read-only engines (time-travel, etc.)
	if e.readOnly {
		return nil
	}

	pkIndexFilename := fmt.Sprintf("pkindex_%d.bin", e.lsn.Load())
	pkIndexPath := filepath.Join(e.dir, pkIndexFilename)
	pkIndexTmpPath := pkIndexPath + ".tmp"

	// Save to temp file
	f, err := e.fs.OpenFile(pkIndexTmpPath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0644)
	if err != nil {
		return err
	}
	defer func() {
		_ = f.Close()                   // Intentionally ignore: cleanup path
		_ = e.fs.Remove(pkIndexTmpPath) // Intentionally ignore: best-effort cleanup
	}()

	if err := e.pkIndex.Save(f); err != nil {
		return err
	}

	if err := f.Sync(); err != nil {
		return err
	}
	_ = f.Close() // Close explicit for Windows/FS safety (ignore error, Sync was called)

	// Rename to final
	if err := e.fs.Rename(pkIndexTmpPath, pkIndexPath); err != nil {
		return err
	}

	// Update Manifest
	// Need to be careful not to introduce race if called outside Close (e.g. background checkpoint).
	// Here we are in Close, holding mu, so it's safe.

	// Copy manifest to avoid mutating the one in use? We are closing, so it's fine.
	// But let's follow the pattern of creating a new manifest object for save.

	newManifest := *e.manifest
	newManifest.PKIndex.Path = pkIndexFilename
	// Do NOT update MaxLSN here. MaxLSN tracks flushed segments.
	// If we update MaxLSN, recovery will skip WAL replay for the active MemTable, losing data.

	store := manifest.NewStore(e.manifestStore)
	if err := store.Save(&newManifest); err != nil {
		return err
	}

	// Update in-memory manifest to reflect new state
	e.manifest.PKIndex.Path = pkIndexFilename
	// e.manifest.MaxLSN remains as is

	return nil
}

// SegmentInfo returns information about all segments in the engine.
func (e *Engine) SegmentInfo() []manifest.SegmentInfo {
	e.mu.Lock()
	defer e.mu.Unlock()
	// Return a copy to avoid races if manifest changes
	infos := make([]manifest.SegmentInfo, len(e.manifest.Segments))
	copy(infos, e.manifest.Segments)
	return infos
}

// DebugInfo returns a detailed string representation of the engine state.
func (e *Engine) DebugInfo() string {
	stats := e.Stats()
	return fmt.Sprintf(
		"Engine State:\n"+
			"  Segments:       %d\n"+
			"  Rows:           %d\n"+
			"  Tombstones:     %d\n"+
			"  MemTable Usage: %d bytes\n"+
			"  Disk Usage:     %d bytes\n",
		stats.SegmentCount,
		stats.RowCount,
		stats.TombstoneCount,
		stats.MemoryUsageBytes,
		stats.DiskUsageBytes,
	)
}

func (e *Engine) runFlushLoop() {
	defer e.wg.Done()
	for {
		select {
		case <-e.closeCh:
			return
		case <-e.flushCh:
			if err := e.Flush(); err != nil {
				if e.logger != nil {
					e.logger.Error("Background flush failed", "error", err)
				}
			}
		}
	}
}

func (e *Engine) runCompactionLoop() {
	defer e.wg.Done()
	for {
		select {
		case <-e.closeCh:
			return
		case <-e.compactionCh:
			e.metrics.OnQueueDepth("compaction_queue", len(e.compactionCh))
			e.checkCompaction(e.ctx)
		}
	}
}

func (e *Engine) checkCompaction(ctx context.Context) {
	snap, err := e.loadSnapshot()
	if err != nil {
		return
	}
	defer snap.DecRef()

	// Get valid IDs from manifest to avoid compacting segments that are being flushed (frozen memtables)
	// Snapshot manifest to get candidates
	e.mu.RLock()
	// Capture copy of manifest segments for policy decision
	manifestSegments := make([]manifest.SegmentInfo, len(e.manifest.Segments))
	copy(manifestSegments, e.manifest.Segments)
	e.mu.RUnlock()

	// Gather candidate segments
	candidates := make([]SegmentStats, len(manifestSegments))
	for i, seg := range manifestSegments {
		candidates[i] = SegmentStats{
			ID:    seg.ID,
			Size:  seg.Size,
			Level: seg.Level,
			MinID: seg.MinID,
			MaxID: seg.MaxID,
		}
	}

	task := e.policy.Pick(candidates)
	if task != nil && len(task.Segments) > 0 {
		if e.logger != nil {
			e.logger.Info("Compaction started", "segments", len(task.Segments), "target_level", task.TargetLevel)
		}
		if err := e.CompactWithContext(ctx, task.Segments, task.TargetLevel); err != nil {
			e.metrics.OnCompaction(0, len(task.Segments), 0, err)
			if e.logger != nil {
				e.logger.Error("Compaction failed", "error", err)
			}
		} else {
			if e.logger != nil {
				e.logger.Info("Compaction completed", "segments", len(task.Segments))
			}
		}
	}
}

func (e *Engine) persistTombstones() error {
	for id, vt := range e.tombstones {
		bm := vt.ToBitmap(math.MaxUint64)
		if bm.IsEmpty() {
			continue // Don't write empty
		}
		path := filepath.Join(e.dir, fmt.Sprintf("segment_%d.tomb", id))
		tmpPath := path + ".tmp"

		f, err := e.fs.OpenFile(tmpPath, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0644)
		if err != nil {
			return err
		}
		if _, err := bm.WriteTo(f); err != nil {
			_ = f.Close()            // Intentionally ignore: cleanup path
			_ = e.fs.Remove(tmpPath) // Intentionally ignore: best-effort cleanup
			return err
		}
		if err := f.Close(); err != nil {
			_ = e.fs.Remove(tmpPath) // Intentionally ignore: best-effort cleanup
			return err
		}
		if err := e.fs.Rename(tmpPath, path); err != nil {
			_ = e.fs.Remove(tmpPath) // Intentionally ignore: best-effort cleanup
			return err
		}
	}
	return nil
}
