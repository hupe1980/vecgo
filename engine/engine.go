package engine

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"log/slog"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/hupe1980/vecgo/blobstore"
	"github.com/hupe1980/vecgo/cache"
	"github.com/hupe1980/vecgo/distance"
	"github.com/hupe1980/vecgo/internal/fs"
	"github.com/hupe1980/vecgo/internal/manifest"
	imetadata "github.com/hupe1980/vecgo/internal/metadata"
	"github.com/hupe1980/vecgo/internal/searcher"
	"github.com/hupe1980/vecgo/internal/segment"
	"github.com/hupe1980/vecgo/internal/segment/diskann"
	"github.com/hupe1980/vecgo/internal/segment/flat"
	"github.com/hupe1980/vecgo/internal/segment/memtable"
	"github.com/hupe1980/vecgo/internal/wal"
	"github.com/hupe1980/vecgo/lexical"
	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/model"
	"github.com/hupe1980/vecgo/pk"
	"github.com/hupe1980/vecgo/resource"
)

// Engine is the main entry point for the vector database.
type Engine struct {
	mu     sync.RWMutex
	dir    string
	store  blobstore.BlobStore
	dim    int
	metric distance.Metric

	manifest *manifest.Manifest
	wal      *wal.WAL
	// pkIndex moved to Snapshot
	lsn atomic.Uint64

	current atomic.Pointer[Snapshot]

	policy       CompactionPolicy
	compactionCh chan struct{}
	closeCh      chan struct{}
	wg           sync.WaitGroup

	metrics MetricsObserver

	walOptions WALOptions

	resourceController *resource.Controller
	blockCache         cache.BlockCache

	compactionConfig CompactionConfig
	flushConfig      FlushConfig
	flushCh          chan struct{}
	closed           bool

	lexicalIndex lexical.Index
	lexicalField string

	schema metadata.Schema

	fs     fs.FileSystem
	logger *slog.Logger

	watermarkFilterPool sync.Pool
	tombstoneFilterPool sync.Pool
}

// FlushConfig holds configuration for automatic flushing.
type FlushConfig struct {
	// MaxMemTableSize is the maximum size of the MemTable in bytes before a flush is triggered.
	// If 0, defaults to 64MB.
	MaxMemTableSize int64

	// MaxWALSize is the maximum size of the WAL in bytes before a flush is triggered.
	// If 0, defaults to 1GB.
	MaxWALSize int64
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

// WithWALOptions sets the WAL options.
func WithWALOptions(opts WALOptions) Option {
	return func(e *Engine) {
		e.walOptions = opts
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

// WithBlobStore overrides the BlobStore used for reading immutable blobs (segments/payloads).
//
// Note: The Engine still uses the local filesystem rooted at dir for manifest/WAL/PK index.
// BlobStore is intended to abstract segment and payload reads (e.g. object store / remote disk).
func WithBlobStore(st blobstore.BlobStore) Option {
	return func(e *Engine) {
		if st != nil {
			e.store = st
		}
	}
}

// Open opens or creates a new Engine.
func Open(dir string, dim int, metric distance.Metric, opts ...Option) (*Engine, error) {
	e := &Engine{
		dir:          dir,
		store:        blobstore.NewLocalStore(dir),
		dim:          dim,
		metric:       metric,
		policy:       &BoundedSizeTieredPolicy{Threshold: 4},
		compactionCh: make(chan struct{}, 1),
		flushCh:      make(chan struct{}, 1),
		closeCh:      make(chan struct{}),
		metrics:      &NoopMetricsObserver{},
		walOptions:   DefaultWALOptions(),
	}

	for _, opt := range opts {
		opt(e)
	}

	if e.resourceController == nil {
		e.resourceController = resource.NewController(resource.Config{
			MemoryLimitBytes: 1 << 30, // 1GB default
		})
	}

	if e.flushConfig.MaxMemTableSize == 0 {
		e.flushConfig.MaxMemTableSize = 64 * 1024 * 1024 // 64MB
	}
	if e.flushConfig.MaxWALSize == 0 {
		e.flushConfig.MaxWALSize = 1024 * 1024 * 1024 // 1GB
	}

	e.blockCache = cache.NewLRUBlockCache(256<<20, e.resourceController) // 256MB default

	if e.fs == nil {
		e.fs = fs.LocalFS{}
	}

	// Ensure dir exists
	if err := e.fs.MkdirAll(dir, 0755); err != nil {
		return nil, err
	}

	// 1. Load Manifest
	mStore := manifest.NewStore(e.fs, dir)
	m, err := mStore.Load()
	if err != nil {
		return nil, fmt.Errorf("failed to load manifest: %w", err)
	}

	// Clean up orphan segments (crash recovery)
	validSegIDs := make(map[uint64]struct{})
	for _, segMeta := range m.Segments {
		validSegIDs[uint64(segMeta.ID)] = struct{}{}
	}
	if entries, err := e.fs.ReadDir(dir); err == nil {
		for _, entry := range entries {
			name := entry.Name()
			if strings.HasPrefix(name, "segment_") {
				parts := strings.Split(name, ".")
				if len(parts) >= 2 {
					var id uint64
					if n, _ := fmt.Sscanf(parts[0], "segment_%d", &id); n == 1 {
						if _, ok := validSegIDs[id]; !ok {
							_ = e.fs.Remove(filepath.Join(dir, name))
						}
					}
				}
			}
		}
	}

	// 2. Open Segments and Tombstones (and Rebuild PK Index)
	segments := make(map[model.SegmentID]*RefCountedSegment)
	tombstones := make(map[model.SegmentID]*imetadata.LocalBitmap)
	pkIdx := pk.NewPersistentIndex()

	for _, segMeta := range m.Segments {
		// Optional payload file.
		payloadPath := fmt.Sprintf("segment_%d.payload", segMeta.ID)
		var payloadBlob blobstore.Blob
		if b, err := e.store.Open(payloadPath); err == nil {
			payloadBlob = b
		} else if !errors.Is(err, blobstore.ErrNotFound) {
			return nil, fmt.Errorf("failed to open payload blob for segment %d: %w", segMeta.ID, err)
		}

		seg, err := openSegment(e.store, segMeta.Path, e.blockCache, payloadBlob)
		if err != nil {
			return nil, fmt.Errorf("failed to open segment %d: %w", segMeta.ID, err)
		}
		segments[segMeta.ID] = NewRefCountedSegment(seg)

		// Populate PK Index
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

			batch, err := seg.Fetch(context.Background(), rows, []string{})
			if err != nil {
				return nil, fmt.Errorf("failed to fetch pks from segment %d: %w", segMeta.ID, err)
			}

			for j := 0; j < batch.RowCount(); j++ {
				pkIdx = pkIdx.Insert(batch.PK(j), model.Location{
					SegmentID: segMeta.ID,
					RowID:     model.RowID(rows[j]),
				})
			}
		}

		// Load Tombstone
		ts := imetadata.NewLocalBitmap()
		tombPath := filepath.Join(dir, fmt.Sprintf("segment_%d.tomb", segMeta.ID))
		if _, err := e.fs.Stat(tombPath); err == nil {
			f, err := e.fs.OpenFile(tombPath, os.O_RDONLY, 0)
			if err != nil {
				return nil, fmt.Errorf("failed to open tombstone file for segment %d: %w", segMeta.ID, err)
			}
			if _, err := ts.ReadFrom(f); err != nil {
				f.Close()
				return nil, fmt.Errorf("failed to read tombstone file for segment %d: %w", segMeta.ID, err)
			}
			f.Close()
		}
		tombstones[segMeta.ID] = ts
	}

	// 3. Open WAL
	walPath := e.getWALPath(m.WALID)
	w, err := wal.Open(e.fs, walPath, toInternalWALOptions(e.walOptions))
	if err != nil {
		return nil, fmt.Errorf("failed to open wal: %w", err)
	}

	// 4. Create Active MemTable
	activeID := m.NextSegmentID
	m.NextSegmentID++ // Reserve ID for active MemTable
	active, err := memtable.New(activeID, dim, metric, e.resourceController)
	if err != nil {
		w.Close()
		return nil, fmt.Errorf("failed to create memtable: %w", err)
	}

	// Initial Snapshot
	snap := &Snapshot{
		refs:           1,
		segments:       segments,
		sortedSegments: make([]*RefCountedSegment, 0),
		tombstones:     tombstones,
		active:         active,
		pkIndex:        pkIdx,
	}
	snap.RebuildSorted()

	// Replay WAL
	reader, err := w.Reader()
	if err != nil {
		return nil, fmt.Errorf("failed to open wal reader: %w", err)
	}
	defer reader.Close()

	maxReplayedLSN := m.MaxLSN

	for {
		rec, err := reader.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			if errors.Is(err, wal.ErrInvalidCRC) || errors.Is(err, wal.ErrShortRead) || errors.Is(err, wal.ErrRecordTooLarge) || errors.Is(err, io.ErrUnexpectedEOF) {
				if e.metrics != nil {
					// We can't easily log via metrics, but we can log if logger exists
				}
				if e.logger != nil {
					e.logger.Warn("Truncating corrupted WAL", "error", err, "path", walPath)
				}
				// Corruption detected (tail garbage). Truncate and ignore.
				validOffset := reader.Offset()
				// Determine truncation behavior:
				// We must close the reader and the file before truncating (on Windows at least, safer generally).
				// reader.Close uses a separate file handle. w uses a separate file handle.
				_ = reader.Close()
				_ = w.Close()

				if tErr := e.fs.Truncate(walPath, validOffset); tErr != nil {
					return nil, fmt.Errorf("failed to truncate corrupt wal: %w (original error: %v)", tErr, err)
				}

				// Reopen WAL in append mode
				w, err = wal.Open(e.fs, walPath, toInternalWALOptions(e.walOptions))
				if err != nil {
					return nil, fmt.Errorf("failed to reopen wal after truncation: %w", err)
				}

				// Stop replaying
				// Since we truncated the rest, there are no more records.
				// However, 'reader' is invalid now. We break.
				// We need to ensure 'defer reader.Close()' doesn't crash or double close (which is fine).
				break
			}
			return nil, fmt.Errorf("wal replay failed: %w", err)
		}

		if rec.LSN <= m.MaxLSN {
			continue // Already flushed
		}
		if rec.LSN > maxReplayedLSN {
			maxReplayedLSN = rec.LSN
		}

		if rec.Type == wal.RecordTypeUpsert {
			// Deserialize metadata
			var md metadata.Document
			if len(rec.Metadata) > 0 {
				if err := json.Unmarshal(rec.Metadata, &md); err != nil {
					return nil, fmt.Errorf("failed to unmarshal metadata during replay: %w", err)
				}
			}

			// Check if PK exists and update tombstones (handle updates)
			if oldLoc, exists := snap.pkIndex.Lookup(rec.PK); exists {
				ts, ok := tombstones[oldLoc.SegmentID]
				if !ok {
					ts = imetadata.NewLocalBitmap()
				}
				ts.Add(uint32(oldLoc.RowID))
				tombstones[oldLoc.SegmentID] = ts
			}

			// Insert into active MemTable
			rowID, err := active.InsertWithPayload(rec.PK, rec.Vector, md, rec.Payload)
			if err != nil {
				return nil, fmt.Errorf("failed to replay insert: %w", err)
			}

			// Update PK Index
			snap.pkIndex = snap.pkIndex.Insert(rec.PK, model.Location{
				SegmentID: activeID,
				RowID:     rowID,
			})
		} else if rec.Type == wal.RecordTypeDelete {
			// Handle delete
			if oldLoc, exists := snap.pkIndex.Lookup(rec.PK); exists {
				ts, ok := tombstones[oldLoc.SegmentID]
				if !ok {
					ts = imetadata.NewLocalBitmap()
				}
				ts.Add(uint32(oldLoc.RowID))
				tombstones[oldLoc.SegmentID] = ts

				snap.pkIndex = snap.pkIndex.Delete(rec.PK)
			}
		}
	}

	e.manifest = m
	e.wal = w
	e.lsn.Store(maxReplayedLSN)

	// Ensure active watermark covers replayed data
	snap.activeWatermark = active.RowCount()

	e.current.Store(snap)

	e.wg.Add(2)
	GoSafe(e.runCompactionLoop)
	GoSafe(e.runFlushLoop)

	return e, nil
}

// Insert adds a vector to the engine.
func (e *Engine) Insert(pk model.PK, vec []float32, md map[string]interface{}, payload []byte) (err error) {
	start := time.Now()
	defer func() {
		e.metrics.OnInsert(time.Since(start), err)
	}()

	if len(vec) != e.dim {
		return fmt.Errorf("%w: expected %d, got %d", ErrInvalidArgument, e.dim, len(vec))
	}

	if e.schema != nil {
		if err := e.schema.Validate(md); err != nil {
			return fmt.Errorf("%w: %v", ErrInvalidArgument, err)
		}
	}

	e.mu.Lock()
	if e.closed {
		e.mu.Unlock()
		return ErrClosed
	}

	snap := e.current.Load()

	// Always clone snapshot because we update pkIndex
	newSnap := snap.Clone()
	newSnap.RebuildSorted()

	// 1. Check PK Index
	if oldLoc, exists := newSnap.pkIndex.Lookup(pk); exists {
		// Mark old location as deleted
		ts, ok := newSnap.tombstones[oldLoc.SegmentID]
		if !ok {
			ts = imetadata.NewLocalBitmap()
		} else {
			ts = ts.Clone()
		}
		ts.Add(uint32(oldLoc.RowID))
		newSnap.tombstones[oldLoc.SegmentID] = ts
	}

	// Serialize metadata
	var mdBytes []byte
	if md != nil {
		var err error
		mdBytes, err = json.Marshal(md)
		if err != nil {
			e.mu.Unlock()
			return fmt.Errorf("failed to marshal metadata: %w", err)
		}
	}

	// 2. Write to WAL
	lsn := e.lsn.Add(1)
	rec := &wal.Record{
		LSN:      lsn,
		Type:     wal.RecordTypeUpsert,
		PK:       pk,
		Vector:   vec,
		Metadata: mdBytes,
		Payload:  payload,
	}
	walStart := time.Now()
	offset, err := e.wal.AppendAsync(rec)
	e.metrics.OnWALWrite(time.Since(walStart), rec.Size())
	if err != nil {
		e.mu.Unlock()
		return err
	}

	// 3. Insert into Active MemTable
	mdDoc, err := metadata.FromMap(md)
	if err != nil {
		e.mu.Unlock()
		return fmt.Errorf("failed to convert metadata: %w", err)
	}
	rowID, err := newSnap.active.InsertWithPayload(pk, vec, mdDoc, payload)
	if err != nil {
		e.mu.Unlock()
		return err
	}
	newSnap.activeWatermark = uint32(rowID) + 1

	// 4. Update PK Index
	newSnap.pkIndex = newSnap.pkIndex.Insert(pk, model.Location{
		SegmentID: newSnap.active.ID(),
		RowID:     rowID,
	})

	// 5. Update Lexical Index
	if e.lexicalIndex != nil && e.lexicalField != "" && md != nil {
		if val, ok := md[e.lexicalField]; ok {
			if str, ok := val.(string); ok {
				if err := e.lexicalIndex.Add(pk, str); err != nil {
					if e.logger != nil {
						e.logger.Error("failed to update lexical index", "pk", pk, "error", err)
					}
				}
			}
		}
	}

	// Publish new snapshot
	e.current.Store(newSnap)
	snap.DecRef()

	// Check triggers
	memSize := newSnap.active.Size()
	walSize := e.wal.Size()
	shouldFlush := memSize > e.flushConfig.MaxMemTableSize || walSize > e.flushConfig.MaxWALSize

	if e.flushConfig.MaxMemTableSize > 0 {
		e.metrics.OnMemTableStatus(memSize, float64(memSize)/float64(e.flushConfig.MaxMemTableSize))
	}

	e.mu.Unlock()

	if shouldFlush {
		select {
		case e.flushCh <- struct{}{}:
		default:
		}
	}

	if e.walOptions.Durability == DurabilitySync {
		return e.wal.WaitFor(offset)
	}

	return nil
}

// BatchInsert adds multiple vectors to the engine.
// It amortizes WAL writes and locking overhead.
func (e *Engine) BatchInsert(records []model.Record) error {
	if len(records) == 0 {
		return nil
	}
	// Validate all first
	for i, r := range records {
		if len(r.Vector) != e.dim {
			return fmt.Errorf("record %d: %w: expected %d, got %d", i, ErrInvalidArgument, e.dim, len(r.Vector))
		}
		if e.schema != nil {
			if err := e.schema.Validate(r.Metadata); err != nil {
				return fmt.Errorf("record %d: %w: %v", i, ErrInvalidArgument, err)
			}
		}
	}

	e.mu.Lock()
	if e.closed {
		e.mu.Unlock()
		return ErrClosed
	}

	snap := e.current.Load()

	// Always clone snapshot
	newSnap := snap.Clone()
	newSnap.RebuildSorted()

	var maxOffset int64

	for _, r := range records {
		// 1. Check PK Index
		if oldLoc, exists := newSnap.pkIndex.Lookup(r.PK); exists {
			// Mark old location as deleted
			ts, ok := newSnap.tombstones[oldLoc.SegmentID]
			if !ok {
				ts = imetadata.NewLocalBitmap()
			} else {
				// Check if we need to clone (COW)
				oldTs := snap.tombstones[oldLoc.SegmentID]
				if ts == oldTs {
					ts = ts.Clone()
					newSnap.tombstones[oldLoc.SegmentID] = ts
				}
			}
			ts.Add(uint32(oldLoc.RowID))
		}

		// Serialize metadata
		var mdBytes []byte
		if r.Metadata != nil {
			var err error
			mdBytes, err = json.Marshal(r.Metadata)
			if err != nil {
				e.mu.Unlock()
				return fmt.Errorf("failed to marshal metadata for pk %v: %w", r.PK, err)
			}
		}

		// 2. Write to WAL
		lsn := e.lsn.Add(1)
		rec := &wal.Record{
			LSN:      lsn,
			Type:     wal.RecordTypeUpsert,
			PK:       r.PK,
			Vector:   r.Vector,
			Metadata: mdBytes,
			Payload:  r.Payload,
		}
		offset, err := e.wal.AppendAsync(rec)
		if err != nil {
			e.mu.Unlock()
			return err
		}
		if offset > maxOffset {
			maxOffset = offset
		}

		// 3. Insert into Active MemTable
		mdDoc, err := metadata.FromMap(r.Metadata)
		if err != nil {
			e.mu.Unlock()
			return fmt.Errorf("failed to convert metadata for pk %v: %w", r.PK, err)
		}
		rowID, err := newSnap.active.InsertWithPayload(r.PK, r.Vector, mdDoc, r.Payload)
		if err != nil {
			e.mu.Unlock()
			return err
		}

		// 4. Update PK Index
		newSnap.pkIndex = newSnap.pkIndex.Insert(r.PK, model.Location{
			SegmentID: newSnap.active.ID(),
			RowID:     rowID,
		})

		// 5. Update Lexical Index
		if e.lexicalIndex != nil && e.lexicalField != "" && r.Metadata != nil {
			if val, ok := r.Metadata[e.lexicalField]; ok {
				if str, ok := val.(string); ok {
					if err := e.lexicalIndex.Add(r.PK, str); err != nil {
						if e.logger != nil {
							e.logger.Error("failed to update lexical index", "pk", r.PK, "error", err)
						}
					}
				}
			}
		}
	}

	// Ensure active watermark covers all new records
	newSnap.activeWatermark = newSnap.active.RowCount()

	e.current.Store(newSnap)
	snap.DecRef()

	// Check triggers
	memSize := newSnap.active.Size()
	walSize := e.wal.Size()
	shouldFlush := memSize > e.flushConfig.MaxMemTableSize || walSize > e.flushConfig.MaxWALSize

	e.mu.Unlock()

	if shouldFlush {
		select {
		case e.flushCh <- struct{}{}:
		default:
		}
	}

	if e.walOptions.Durability == DurabilitySync {
		return e.wal.WaitFor(maxOffset)
	}

	return nil
}

// Delete removes a vector from the engine.
func (e *Engine) Delete(pk model.PK) (err error) {
	start := time.Now()
	defer func() {
		e.metrics.OnDelete(time.Since(start), err)
	}()

	e.mu.Lock()
	if e.closed {
		e.mu.Unlock()
		return ErrClosed
	}

	snap := e.current.Load()

	// 1. Check PK Index
	oldLoc, exists := snap.pkIndex.Lookup(pk)
	if !exists {
		e.mu.Unlock()
		return nil // Idempotent
	}

	// 2. Write to WAL
	lsn := e.lsn.Add(1)
	rec := &wal.Record{
		LSN:  lsn,
		Type: wal.RecordTypeDelete,
		PK:   pk,
	}
	offset, err := e.wal.AppendAsync(rec)
	if err != nil {
		e.mu.Unlock()
		return err
	}

	// 3. Apply Delete
	// Always clone snapshot
	newSnap := snap.Clone()
	newSnap.RebuildSorted()

	ts, ok := newSnap.tombstones[oldLoc.SegmentID]
	if !ok {
		ts = imetadata.NewLocalBitmap()
	} else {
		ts = ts.Clone()
	}
	ts.Add(uint32(oldLoc.RowID))
	newSnap.tombstones[oldLoc.SegmentID] = ts

	// 4. Update PK Index
	newSnap.pkIndex = newSnap.pkIndex.Delete(pk)

	// 5. Update Lexical Index
	if e.lexicalIndex != nil {
		if err := e.lexicalIndex.Delete(pk); err != nil {
			if e.logger != nil {
				e.logger.Error("failed to delete from lexical index", "pk", pk, "error", err)
			}
		}
	}

	e.current.Store(newSnap)
	snap.DecRef()

	e.mu.Unlock()

	if e.walOptions.Durability == DurabilitySync {
		return e.wal.WaitFor(offset)
	}

	return nil
}

// BatchDelete removes multiple vectors from the engine in a single operation.
// It is atomic and more efficient than calling Delete in a loop.
func (e *Engine) BatchDelete(pks []model.PK) error {
	e.mu.Lock()
	if e.closed {
		e.mu.Unlock()
		return ErrClosed
	}

	snap := e.current.Load()
	newSnap := snap.Clone()
	newSnap.RebuildSorted()

	var maxOffset int64
	var anyDeleted bool
	clonedTombstones := make(map[model.SegmentID]bool)

	for _, pk := range pks {
		// 1. Check PK Index
		oldLoc, exists := newSnap.pkIndex.Lookup(pk)
		if !exists {
			continue
		}
		anyDeleted = true

		// 2. Write to WAL
		lsn := e.lsn.Add(1)
		rec := &wal.Record{
			LSN:  lsn,
			Type: wal.RecordTypeDelete,
			PK:   pk,
		}
		offset, err := e.wal.AppendAsync(rec)
		if err != nil {
			e.mu.Unlock()
			// Note: We might have written some WAL records but not updated memory.
			// This is safe because replay will just delete them again (idempotent).
			// But we fail the batch.
			return err
		}
		if offset > maxOffset {
			maxOffset = offset
		}

		// 3. Apply Delete
		ts, ok := newSnap.tombstones[oldLoc.SegmentID]
		if !ok {
			ts = imetadata.NewLocalBitmap()
			newSnap.tombstones[oldLoc.SegmentID] = ts
			clonedTombstones[oldLoc.SegmentID] = true
		} else if !clonedTombstones[oldLoc.SegmentID] {
			ts = ts.Clone()
			newSnap.tombstones[oldLoc.SegmentID] = ts
			clonedTombstones[oldLoc.SegmentID] = true
		}
		ts.Add(uint32(oldLoc.RowID))

		// 4. Update PK Index
		newSnap.pkIndex = newSnap.pkIndex.Delete(pk)

		// 5. Update Lexical Index
		if e.lexicalIndex != nil {
			if err := e.lexicalIndex.Delete(pk); err != nil {
				if e.logger != nil {
					e.logger.Error("failed to delete from lexical index", "pk", pk, "error", err)
				}
			}
		}
	}

	if anyDeleted {
		e.current.Store(newSnap)
		snap.DecRef()
	} else {
		newSnap.DecRef()
	}

	e.mu.Unlock()

	if anyDeleted && e.walOptions.Durability == DurabilitySync {
		return e.wal.WaitFor(maxOffset)
	}

	return nil
}

// Search performs a k-NN search.
func (e *Engine) Search(ctx context.Context, q []float32, k int, opts ...func(*model.SearchOptions)) (res []model.Candidate, err error) {
	start := time.Now()
	defer func() {
		// Heuristic: determine segment type.
		// For now we just say "mixed" or "l0" depending on snapshot.
		// Detailed breakdown per segment is hard here.
		e.metrics.OnSearch(time.Since(start), "mixed", k, len(res), err)
	}()

	select {
	case <-e.closeCh:
		return nil, ErrClosed
	default:
	}

	if len(q) != e.dim {
		return nil, fmt.Errorf("%w: expected %d, got %d", ErrInvalidArgument, e.dim, len(q))
	}
	if k <= 0 {
		return nil, fmt.Errorf("%w: k must be > 0", ErrInvalidArgument)
	}

	// Parse options
	options := model.SearchOptions{
		K:            k,
		RefineFactor: 1.0,
	}
	for _, opt := range opts {
		opt(&options)
	}

	// Acquire Snapshot
	snap := e.current.Load()
	snap.IncRef()
	defer snap.DecRef()

	// Acquire Searcher
	s := searcher.Get()
	defer searcher.Put(s)

	// Normalize cosine queries once at the engine boundary.
	// Internal HNSW also normalizes, but flat/disk segments rely on the engine.
	qExec := q
	if e.metric == distance.MetricCosine {
		if cap(s.ScratchVec) < e.dim {
			s.ScratchVec = make([]float32, e.dim)
		}
		s.ScratchVec = s.ScratchVec[:e.dim]
		copy(s.ScratchVec, q)
		if !distance.NormalizeL2InPlace(s.ScratchVec) {
			return nil, fmt.Errorf("%w: cannot normalize zero query", ErrInvalidArgument)
		}
		qExec = s.ScratchVec
	}

	// Initialize Global Heap
	descending := e.metric != distance.MetricL2
	s.Heap.Reset(descending)

	// 1. Gather candidates from all segments
	// We use a larger K for the initial gathering to allow for reranking refinement.
	searchK := int(float32(k) * options.RefineFactor)
	if searchK < k {
		searchK = k
	}

	// Search Active MemTable
	var activeFilter segment.Filter
	if ts, ok := snap.tombstones[snap.active.ID()]; ok && !ts.IsEmpty() {
		activeFilter = &tombstoneFilter{ts: ts}
	}

	// Apply Snapshot Isolation to Active Segment
	// Pooling watermarkFilter to avoid allocation
	wfObj := e.watermarkFilterPool.Get()
	var wFilter *watermarkFilter
	if wfObj == nil {
		wFilter = &watermarkFilter{}
	} else {
		wFilter = wfObj.(*watermarkFilter)
	}
	wFilter.limit = snap.activeWatermark
	wFilter.next = activeFilter
	activeFilter = wFilter

	err = snap.active.Search(ctx, qExec, searchK, activeFilter, options, s)

	// Release filter back to pool
	wFilter.next = nil
	e.watermarkFilterPool.Put(wFilter)

	if err != nil {
		return nil, err
	}

	// Search Immutable Segments
	for _, seg := range snap.sortedSegments {
		id := seg.ID()
		var filter segment.Filter

		// Combine user filter and tombstone filter
		ts, hasTombstones := snap.tombstones[id]
		if hasTombstones && ts.IsEmpty() {
			hasTombstones = false
		}

		var tsFilter *tombstoneFilter
		if hasTombstones {
			tfObj := e.tombstoneFilterPool.Get()
			if tfObj == nil {
				tsFilter = &tombstoneFilter{ts: ts}
			} else {
				tsFilter = tfObj.(*tombstoneFilter)
				tsFilter.ts = ts
			}
			filter = tsFilter
		}

		err := seg.Search(ctx, qExec, searchK, filter, options, s)

		if tsFilter != nil {
			tsFilter.ts = nil
			e.tombstoneFilterPool.Put(tsFilter)
		}

		if err != nil {
			return nil, err
		}
	}

	// 2. Extract candidates for reranking
	s.CandidateBuffer = s.CandidateBuffer[:0]
	for s.Heap.Len() > 0 {
		s.CandidateBuffer = append(s.CandidateBuffer, s.Heap.Pop())
	}

	// 3. Rerank
	// Sort by SegmentID for batching
	slices.SortFunc(s.CandidateBuffer, func(a, b model.Candidate) int {
		if a.Loc.SegmentID < b.Loc.SegmentID {
			return -1
		}
		if a.Loc.SegmentID > b.Loc.SegmentID {
			return 1
		}
		return 0
	})

	s.Results = s.Results[:0]

	for i := 0; i < len(s.CandidateBuffer); {
		segID := s.CandidateBuffer[i].Loc.SegmentID
		j := i + 1
		for j < len(s.CandidateBuffer) && s.CandidateBuffer[j].Loc.SegmentID == segID {
			j++
		}

		// Batch for segID is [i, j)
		var seg segment.Segment
		if segID == snap.active.ID() {
			seg = snap.active
		} else {
			if s, ok := snap.segments[segID]; ok {
				seg = s
			}
		}

		if seg != nil {
			// Rerank appends to s.Results
			var err error
			s.Results, err = seg.Rerank(ctx, qExec, s.CandidateBuffer[i:j], s.Results)
			if err != nil {
				return nil, err
			}
		}
		i = j
	}

	// 4. Final Top-K Selection
	s.Heap.Reset(descending)
	for _, c := range s.Results {
		if s.Heap.Len() < k {
			s.Heap.Push(c)
		} else {
			top := s.Heap.Candidates[0]
			if searcher.CandidateBetter(c, top, descending) {
				s.Heap.ReplaceTop(c)
			}
		}
	}

	// 5. Extract Final Results
	s.Results = s.Results[:0]
	for s.Heap.Len() > 0 {
		s.Results = append(s.Results, s.Heap.Pop())
	}
	slices.Reverse(s.Results)

	// 6. Fetch PKs and Materialize Data
	// Determine columns to fetch
	var cols []string
	if options.IncludeVector {
		cols = append(cols, "vector")
	}
	if options.IncludeMetadata {
		cols = append(cols, "metadata")
	}
	if options.IncludePayload {
		cols = append(cols, "payload")
	}

	// Sort by SegmentID to optimize IO
	slices.SortFunc(s.Results, func(a, b model.Candidate) int {
		if a.Loc.SegmentID < b.Loc.SegmentID {
			return -1
		}
		if a.Loc.SegmentID > b.Loc.SegmentID {
			return 1
		}
		return 0
	})

	for i := 0; i < len(s.Results); {
		segID := s.Results[i].Loc.SegmentID
		j := i + 1
		for j < len(s.Results) && s.Results[j].Loc.SegmentID == segID {
			j++
		}

		var seg segment.Segment
		if segID == snap.active.ID() {
			seg = snap.active
		} else {
			if s, ok := snap.segments[segID]; ok {
				seg = s
			}
		}

		if seg != nil {
			count := j - i
			if cap(s.ScratchIDs) < count {
				s.ScratchIDs = make([]uint32, count)
			}
			s.ScratchIDs = s.ScratchIDs[:count]

			for k := 0; k < count; k++ {
				s.ScratchIDs[k] = uint32(s.Results[i+k].Loc.RowID)
			}

			if len(cols) == 0 {
				// Fast path: Fetch only PKs
				if cap(s.ScratchPKs) < count {
					s.ScratchPKs = make([]model.PK, count)
				}
				s.ScratchPKs = s.ScratchPKs[:count]

				if err := seg.FetchPKs(ctx, s.ScratchIDs, s.ScratchPKs); err != nil {
					return nil, err
				}
				for k := 0; k < count; k++ {
					s.Results[i+k].PK = s.ScratchPKs[k]
				}
			} else {
				batch, err := seg.Fetch(ctx, s.ScratchIDs, cols)
				if err != nil {
					return nil, err
				}

				for k := 0; k < count; k++ {
					s.Results[i+k].PK = batch.PK(k)
					if options.IncludeVector {
						s.Results[i+k].Vector = batch.Vector(k)
					}
					if options.IncludeMetadata {
						s.Results[i+k].Metadata = batch.Metadata(k).ToMap()
					}
					if options.IncludePayload {
						s.Results[i+k].Payload = batch.Payload(k)
					}
				}
			}
		}
		i = j
	}

	// 7. Restore Order (Best Score First)
	slices.SortFunc(s.Results, func(a, b model.Candidate) int {
		if descending {
			// Larger is better
			if a.Score > b.Score {
				return -1
			}
			if a.Score < b.Score {
				return 1
			}
		} else {
			// Smaller is better
			if a.Score < b.Score {
				return -1
			}
			if a.Score > b.Score {
				return 1
			}
		}
		// Tie-breaker: SegmentID, RowID
		if a.Loc.SegmentID < b.Loc.SegmentID {
			return -1
		}
		if a.Loc.SegmentID > b.Loc.SegmentID {
			return 1
		}
		if a.Loc.RowID < b.Loc.RowID {
			return -1
		}
		if a.Loc.RowID > b.Loc.RowID {
			return 1
		}
		return 0
	})

	// 8. Visibility Check (Belt-and-Suspenders)
	// Ensure that the returned candidates are consistent with the snapshot's PK index.
	// This protects against stale versions resurfacing due to compaction bugs or tombstone issues.
	validCount := 0
	for _, cand := range s.Results {
		loc, ok := snap.pkIndex.Lookup(cand.PK)
		if !ok {
			// PK not found in index -> deleted
			continue
		}
		if loc != cand.Loc {
			// Location mismatch -> stale version
			continue
		}
		s.Results[validCount] = cand
		validCount++
	}
	s.Results = s.Results[:validCount]

	// Return copy
	ret := make([]model.Candidate, len(s.Results))
	copy(ret, s.Results)
	return ret, nil
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

			res, err := e.Search(ctx, q, k, opts...)
			if err != nil {
				errOnce.Do(func() {
					firstErr = err
				})
				return
			}
			results[i] = res
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
		snap := e.current.Load()
		if snap != nil {
			snap.IncRef()
		}
		if snap == nil {
			yield(nil, errors.New("engine closed or not initialized"))
			return
		}
		defer snap.DecRef()

		for pk, loc := range snap.pkIndex.Scan() {
			if ctx.Err() != nil {
				yield(nil, ctx.Err())
				return
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
					PK:       pk,
					Vector:   batch.Vector(0),
					Metadata: batch.Metadata(0).ToMap(),
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
					PK:       pk,
					Vector:   batch.Vector(0),
					Metadata: batch.Metadata(0).ToMap(),
					Payload:  batch.Payload(0),
				}
			}

			// Ensure PK is set
			rec.PK = pk

			if cfg.Filter != nil {
				doc, err := metadata.FromMap(rec.Metadata)
				if err != nil {
					if !yield(nil, err) {
						return
					}
					return
				}
				if !cfg.Filter.Matches(doc) {
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
		finalScores[c.PK] = score
	}

	// Process Lexical Results
	for rank, c := range lexResults {
		score := 1.0 / float32(rrfK+rank+1)
		finalScores[c.PK] += score
	}

	// 4. Convert to Candidates and Sort
	candidates := s.Results[:0]
	for pk, score := range finalScores {
		candidates = append(candidates, model.Candidate{
			PK:    pk,
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
	snap := e.current.Load()
	snap.IncRef()
	defer snap.DecRef()

	validCount := 0
	for _, cand := range candidates {
		if loc, ok := snap.pkIndex.Lookup(cand.PK); ok {
			cand.Loc = loc
			candidates[validCount] = cand
			validCount++
		}
	}
	candidates = candidates[:validCount]

	return candidates, nil
}

// Get returns the full record (vector, metadata, payload) for the given primary key.
func (e *Engine) Get(pk model.PK) (rec *model.Record, err error) {
	start := time.Now()
	defer func() {
		e.metrics.OnGet(time.Since(start), err)
	}()
	select {
	case <-e.closeCh:
		return nil, ErrClosed
	default:
	}

	// Acquire Snapshot
	snap := e.current.Load()
	snap.IncRef()
	defer snap.DecRef()

	// Lookup PK
	loc, ok := snap.pkIndex.Lookup(pk)
	if !ok {
		return nil, fmt.Errorf("%w: pk %v", ErrInvalidArgument, pk)
	}

	// Check if in active memtable
	if loc.SegmentID == snap.active.ID() {
		// MemTable fetch
		batch, err := snap.active.Fetch(context.Background(), []uint32{uint32(loc.RowID)}, []string{"vector", "metadata", "payload"})
		if err != nil {
			return nil, err
		}
		return &model.Record{
			PK:       pk,
			Vector:   batch.Vector(0),
			Metadata: batch.Metadata(0).ToMap(),
			Payload:  batch.Payload(0),
		}, nil
	}

	// Check segments
	seg, ok := snap.segments[loc.SegmentID]
	if !ok {
		return nil, fmt.Errorf("segment %d not found for pk %v", loc.SegmentID, pk)
	}

	batch, err := seg.Fetch(context.Background(), []uint32{uint32(loc.RowID)}, []string{"vector", "metadata", "payload"})
	if err != nil {
		return nil, err
	}
	return &model.Record{
		PK:       pk,
		Vector:   batch.Vector(0),
		Metadata: batch.Metadata(0).ToMap(),
		Payload:  batch.Payload(0),
	}, nil
}

// Flush flushes the active MemTable to disk.
func (e *Engine) Flush() (err error) {
	start := time.Now()
	var itemsFlushed int
	var bytesFlushed uint64

	defer func() {
		e.metrics.OnFlush(time.Since(start), itemsFlushed, bytesFlushed, err)
	}()

	e.mu.Lock()
	defer e.mu.Unlock()

	currentLSN := e.lsn.Load()
	snap := e.current.Load()
	rowCount := int(snap.active.RowCount())
	itemsFlushed = rowCount

	if rowCount == 0 {
		return nil
	}

	if e.logger != nil {
		e.logger.Info("Flush started", "segmentID", snap.active.ID(), "rowCount", rowCount)
	}

	// 1. Create new segment file (atomic publication)
	segID := snap.active.ID()
	filename := fmt.Sprintf("segment_%d.bin", segID)
	path := filepath.Join(e.dir, filename)
	tmpPath := path + ".tmp"

	payloadFilename := fmt.Sprintf("segment_%d.payload", segID)
	payloadPath := filepath.Join(e.dir, payloadFilename)
	payloadTmpPath := payloadPath + ".tmp"

	f, err := e.fs.OpenFile(tmpPath, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0644)
	if err != nil {
		return err
	}
	defer func() {
		if f != nil {
			f.Close()
			_ = e.fs.Remove(tmpPath)
		}
	}()

	payloadF, err := e.fs.OpenFile(payloadTmpPath, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0644)
	if err != nil {
		return err
	}
	defer func() {
		if payloadF != nil {
			payloadF.Close()
			_ = e.fs.Remove(payloadTmpPath)
		}
	}()

	// For Flush (L0 -> L1), we disable partitioning (k=0) to keep it fast.
	// Partitioning happens during Compaction (L1 -> L2).
	// We also disable quantization for L1 (keep it exact/fast).
	w := flat.NewWriter(f, payloadF, segID, e.dim, e.metric, 0, flat.QuantizationNone)

	// 2. Write data
	var count uint32
	var updates []struct {
		pk  model.PK
		loc model.Location
	}

	err = snap.active.Iterate(func(rowID uint32, pk model.PK, vec []float32, md metadata.Document, payload []byte) error {
		// Skip deleted items
		if ts, ok := snap.tombstones[segID]; ok && ts.Contains(rowID) {
			return nil
		}

		if err := w.Add(pk, vec, md, payload); err != nil {
			return err
		}
		updates = append(updates, struct {
			pk  model.PK
			loc model.Location
		}{pk, model.Location{SegmentID: segID, RowID: model.RowID(count)}})
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
	// Explicit close to ensure flush to disk
	if err := f.Close(); err != nil {
		return err
	}
	f = nil
	if err := payloadF.Close(); err != nil {
		return err
	}
	payloadF = nil

	// Publish atomically (rename + dir fsync)
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

	// 3. Open new segment for reading
	blob, err := e.store.Open(filename)
	if err != nil {
		return err
	}

	opts := []flat.Option{flat.WithBlockCache(e.blockCache)}
	if payloadBlob, err := e.store.Open(payloadFilename); err == nil {
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

	// 4. Update Snapshot
	newSnap := snap.Clone()
	newSnap.segments[segID] = NewRefCountedSegment(newSeg)
	newSnap.tombstones[segID] = imetadata.NewLocalBitmap()

	// Persist tombstones before rotating WAL
	if err := e.persistTombstones(newSnap); err != nil {
		return err
	}

	// 5. Rotate WAL
	if err := e.wal.Close(); err != nil {
		return err
	}

	// Prepare new WAL
	oldWALID := e.manifest.WALID
	newWALID := oldWALID + 1
	if newWALID == 0 {
		newWALID = 1
	}

	newWALPath := e.getWALPath(newWALID)
	newWal, err := wal.Open(e.fs, newWALPath, toInternalWALOptions(e.walOptions))
	if err != nil {
		return err
	}
	e.wal = newWal

	// 6. Create New Snapshot
	// newSnap is already created in step 4

	// Release the ref to the old active memtable that Clone() added,
	// because we are about to replace it with a new one.
	newSnap.active.DecRef()

	newActiveID := e.manifest.NextSegmentID
	e.manifest.NextSegmentID++
	newSnap.active, err = memtable.New(newActiveID, e.dim, e.metric, e.resourceController)
	if err != nil {
		newWal.Close()
		return fmt.Errorf("failed to create new memtable: %w", err)
	}

	// Update PK Index
	for _, u := range updates {
		newSnap.pkIndex = newSnap.pkIndex.Insert(u.pk, u.loc)
	}

	// Update Engine State
	newSnap.RebuildSorted()
	e.current.Store(newSnap)
	snap.DecRef()

	// 7. Update Manifest
	e.manifest.Segments = append(e.manifest.Segments, manifest.SegmentInfo{
		ID:       segID,
		Level:    0, // L0
		RowCount: count,
		Path:     filename,
	})
	e.manifest.MaxLSN = currentLSN
	e.manifest.WALID = newWALID
	// NextSegmentID already incremented for new active table

	mStore := manifest.NewStore(e.fs, e.dir)
	if err := mStore.Save(e.manifest); err != nil {
		return err
	}

	// 8. Cleanup Old WAL
	oldWALPath := e.getWALPath(oldWALID)
	if err := e.fs.Remove(oldWALPath); err != nil {
		if e.logger != nil {
			e.logger.Warn("Failed to remove old WAL", "path", oldWALPath, "error", err)
		}
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

// EngineStats contains runtime statistics for the engine.
type EngineStats struct {
	SegmentCount     int
	RowCount         int
	TombstoneCount   int
	DiskUsageBytes   int64
	WALSizeBytes     int64
	MemoryUsageBytes int64 // L0 + overhead
}

// Stats returns the current engine statistics.
func (e *Engine) Stats() EngineStats {
	// No lock needed:
	// - e.current is atomic
	// - e.wal.Size() is thread-safe
	// - segment/memtable methods are thread-safe

	snap := e.current.Load()
	stats := EngineStats{
		SegmentCount:     len(snap.segments),
		RowCount:         int(snap.active.RowCount()),
		MemoryUsageBytes: snap.active.Size(),
	}

	for _, seg := range snap.segments {
		stats.RowCount += int(seg.RowCount())
		stats.DiskUsageBytes += seg.Size()
	}

	for _, ts := range snap.tombstones {
		stats.TombstoneCount += int(ts.Cardinality())
	}

	stats.WALSizeBytes = e.wal.Size()

	return stats
}

// Close closes the engine.
func (e *Engine) Close() error {
	e.mu.Lock()
	if e.closed {
		e.mu.Unlock()
		return ErrClosed
	}
	e.closed = true
	e.mu.Unlock()

	select {
	case <-e.closeCh:
		return ErrClosed // Already closed (should be caught by check above)
	default:
		close(e.closeCh)
	}
	e.wg.Wait()

	e.mu.Lock()
	defer e.mu.Unlock()

	snap := e.current.Load()
	snap.DecRef() // Release Engine's reference

	// Note: snap.active.Close() is called when snap.DecRef() reaches zero?
	// No, Snapshot.DecRef() calls segments.DecRef(), but active is a *MemTable.
	// MemTable doesn't have ref counting in the same way as segments in this implementation.
	// But Snapshot holds a pointer to it.
	// The Engine owns the active memtable via the current snapshot.
	// When we close the engine, we should close the active memtable.
	// However, if other snapshots are still alive (readers), they might reference it?
	// In this simple implementation, we just close it.
	if err := snap.active.Close(); err != nil {
		// Log error?
	}

	// Save Tombstones
	if err := e.persistTombstones(snap); err != nil {
		return err
	}

	// Save PK Index
	// TODO: Implement persistence for PersistentIndex
	/*
		pkPath := filepath.Join(e.dir, "pk_index.bin")
		f, err := os.Create(pkPath)
		if err != nil {
			return err
		}
		defer f.Close()
		if err := e.pkIndex.Save(f); err != nil {
			return err
		}
	*/

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
			"  WAL Size:       %d bytes\n"+
			"  MemTable Usage: %d bytes\n"+
			"  Disk Usage:     %d bytes\n",
		stats.SegmentCount,
		stats.RowCount,
		stats.TombstoneCount,
		stats.WALSizeBytes,
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
			e.checkCompaction()
		}
	}
}

func (e *Engine) checkCompaction() {
	snap := e.current.Load()
	snap.IncRef()
	defer snap.DecRef()

	// Gather candidate segments
	candidates := make([]SegmentStats, 0, len(snap.segments))
	for id, seg := range snap.segments {
		candidates = append(candidates, SegmentStats{
			ID:   id,
			Size: seg.Size(),
		})
	}

	toCompact := e.policy.Pick(candidates)
	if len(toCompact) > 0 {
		if e.logger != nil {
			e.logger.Info("Compaction started", "segments", len(toCompact))
		}
		if err := e.Compact(toCompact); err != nil {
			e.metrics.OnCompaction(0, len(toCompact), 0, err)
			if e.logger != nil {
				e.logger.Error("Compaction failed", "error", err)
			}
		} else {
			if e.logger != nil {
				e.logger.Info("Compaction completed", "segments", len(toCompact))
			}
		}
	}
}

func (e *Engine) persistTombstones(snap *Snapshot) error {
	for id, ts := range snap.tombstones {
		if ts.IsEmpty() {
			// Ensure we write empty file if it exists?
			// Or just delete it?
			// If we delete it, Open assumes empty.
			// But if we overwrite an existing non-empty file with empty, we must ensure it's empty.
			// Let's write empty.
		}
		path := filepath.Join(e.dir, fmt.Sprintf("segment_%d.tomb", id))
		tmpPath := path + ".tmp"

		f, err := e.fs.OpenFile(tmpPath, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0644)
		if err != nil {
			return err
		}
		if _, err := ts.WriteTo(f); err != nil {
			f.Close()
			e.fs.Remove(tmpPath)
			return err
		}
		if err := f.Close(); err != nil {
			e.fs.Remove(tmpPath)
			return err
		}
		if err := e.fs.Rename(tmpPath, path); err != nil {
			e.fs.Remove(tmpPath)
			return err
		}
	}
	return nil
}

func (e *Engine) getWALPath(id uint64) string {
	if id == 0 {
		return filepath.Join(e.dir, "wal.log")
	}
	return filepath.Join(e.dir, fmt.Sprintf("wal_%d.log", id))
}
