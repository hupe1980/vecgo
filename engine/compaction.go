package engine

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/hupe1980/vecgo/blobstore"
	"github.com/hupe1980/vecgo/cache"
	"github.com/hupe1980/vecgo/internal/manifest"
	imetadata "github.com/hupe1980/vecgo/internal/metadata"
	"github.com/hupe1980/vecgo/internal/segment"
	"github.com/hupe1980/vecgo/internal/segment/diskann"
	"github.com/hupe1980/vecgo/internal/segment/flat"
	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/model"
	"github.com/hupe1980/vecgo/resource"
)

// Compact merges the specified segments into a new segment.
// It is designed to be concurrency-friendly:
// 1. Holds Lock to snapshot state (segments + tombstones).
// 2. Releases Lock to perform heavy I/O (merge).
// 3. Re-acquires Lock to commit changes (CAS on PK Index).
func (e *Engine) Compact(segmentIDs []model.SegmentID) (err error) {
	start := time.Now()
	var dropped, created int
	defer func() {
		e.metrics.OnCompaction(time.Since(start), dropped, created, err)
	}()

	// --- Phase 1: Snapshot State ---
	// We need a snapshot to read from.
	snap := e.current.Load()
	snap.IncRef()
	defer snap.DecRef()

	// Validate segments and snapshot tombstones
	segments := make([]segment.Segment, 0, len(segmentIDs))
	tombstones := make(map[model.SegmentID]*imetadata.LocalBitmap)

	for _, id := range segmentIDs {
		seg, ok := snap.segments[id]
		if !ok {
			return fmt.Errorf("segment %d not found in current snapshot", id)
		}
		segments = append(segments, seg)

		if ts, ok := snap.tombstones[id]; ok {
			tombstones[id] = ts.Clone()
		}
	}

	// Reserve new SegmentID
	e.mu.Lock()
	newSegID := e.manifest.NextSegmentID
	e.manifest.NextSegmentID++
	e.mu.Unlock()

	// --- Phase 2: Merge (No Lock) ---

	filename := fmt.Sprintf("segment_%d.bin", newSegID)
	path := filepath.Join(e.dir, filename)
	tmpPath := path + ".tmp"

	payloadFilename := fmt.Sprintf("segment_%d.payload", newSegID)
	payloadPath := filepath.Join(e.dir, payloadFilename)
	payloadTmpPath := payloadPath + ".tmp"

	f, err := e.fs.OpenFile(tmpPath, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0644)
	if err != nil {
		return err
	}
	// Defer close in case of error, but we will close explicitly on success
	defer func() {
		if f != nil {
			f.Close()
			e.fs.Remove(tmpPath) // Clean up temp file on error
		}
	}()

	payloadF, err := e.fs.OpenFile(payloadTmpPath, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0644)
	if err != nil {
		return err
	}
	defer func() {
		if payloadF != nil {
			payloadF.Close()
			e.fs.Remove(payloadTmpPath) // Clean up temp file on error
		}
	}()

	// Calculate total rows to estimate k (partitions) or choose segment type
	var totalRows uint32
	for _, seg := range segments {
		totalRows += seg.RowCount()
	}

	var writer io.Writer = f
	if e.resourceController != nil {
		writer = resource.NewRateLimitedWriter(f, e.resourceController, context.Background())
	}

	var (
		addFunc   func(pk model.PrimaryKey, vec []float32, md metadata.Document, payload []byte) error
		flushFunc func() error
	)

	// Heuristic: Use DiskANN for larger segments (> 10k vectors), Flat for smaller.
	// This ensures "best" performance: Flat is faster for small, DiskANN scales for large.
	diskAnnThreshold := uint32(10000)
	if e.compactionConfig.DiskANNThreshold > 0 {
		diskAnnThreshold = uint32(e.compactionConfig.DiskANNThreshold)
	}

	if totalRows >= diskAnnThreshold {
		opts := e.compactionConfig.DiskANNOptions
		if opts.R == 0 {
			opts = diskann.DefaultOptions()
		}
		opts.ResourceController = e.resourceController

		w := diskann.NewWriter(writer, payloadF, uint64(newSegID), e.dim, e.metric, opts)

		addFunc = func(pk model.PrimaryKey, vec []float32, md metadata.Document, payload []byte) error {
			return w.Add(pk, vec, md, payload)
		}
		flushFunc = func() error {
			if err := w.Write(context.Background()); err != nil {
				return err
			}
			return nil
		}
	} else {
		// Heuristic: rows / 8192
		k := int(totalRows / 8192)
		if k < 1 {
			k = 1
		}

		quantType := e.compactionConfig.FlatQuantizationType
		w := flat.NewWriter(writer, payloadF, newSegID, e.dim, e.metric, k, quantType)

		addFunc = w.Add
		flushFunc = func() error {
			if err := w.Flush(); err != nil {
				return err
			}
			return nil
		}
	}

	// Map to track new locations for PK index update
	// PK -> (OldRowID, NewRowID)
	// We need OldRowID to verify validity during commit.
	type move struct {
		OldSegID model.SegmentID
		OldRowID uint32
		NewRowID uint32
	}
	moves := make(map[model.PrimaryKey]move)

	var count uint32
	for i, seg := range segments {
		segID := segmentIDs[i]
		ts := tombstones[segID]

		err := seg.Iterate(func(rowID uint32, pk model.PrimaryKey, vec []float32, md metadata.Document, payload []byte) error {
			// Check tombstone (snapshot)
			if ts != nil && ts.Contains(uint32(rowID)) {
				return nil // Skip deleted
			}

			// Write to new segment
			if err := addFunc(pk, vec, md, payload); err != nil {
				return err
			}

			moves[pk] = move{
				OldSegID: segID,
				OldRowID: rowID,
				NewRowID: count,
			}
			count++
			return nil
		})
		if err != nil {
			return err
		}
	}

	if err := flushFunc(); err != nil {
		return err
	}
	if err := payloadF.Sync(); err != nil {
		return err
	}
	if err := f.Sync(); err != nil {
		return err
	}
	if err := payloadF.Close(); err != nil {
		return err
	}
	payloadF = nil // Prevent deferred close/remove
	if err := f.Close(); err != nil {
		return err
	}
	f = nil // Prevent defer close/remove

	// Atomic Rename (publish) + dir fsync
	if err := e.fs.Rename(tmpPath, path); err != nil {
		return err
	}
	if err := e.fs.Rename(payloadTmpPath, payloadPath); err != nil {
		_ = e.fs.Remove(path)
		return err
	}
	if err := syncDir(e.fs, e.dir); err != nil {
		return err
	}

	// Open the new segment to verify and have it ready
	var payloadBlob blobstore.Blob
	if b, err := e.store.Open(context.Background(), payloadFilename); err == nil {
		payloadBlob = b
	} else if !errors.Is(err, blobstore.ErrNotFound) {
		fmt.Printf("Compaction failed: open payload blob: %v\n", err)
		e.fs.Remove(path)
		e.fs.Remove(payloadPath)
		return err
	}

	newSeg, err := openSegment(context.Background(), e.store, filename, e.blockCache, payloadBlob)
	if err != nil {
		fmt.Printf("Compaction failed: openSegment: %v\n", err)
		e.fs.Remove(path)
		e.fs.Remove(payloadPath)
		return err
	}

	if stat, err := e.fs.Stat(path); err == nil {
		e.metrics.OnThroughput("compaction_write", stat.Size())
	}

	// --- Phase 3: Commit (Lock) ---
	e.mu.Lock()
	defer e.mu.Unlock()

	// Reload current snapshot to check validity
	currentSnap := e.current.Load()

	// 1. Verify inputs are still valid (not already compacted/deleted)
	for _, id := range segmentIDs {
		if _, ok := currentSnap.segments[id]; !ok {
			// Abort! Segments disappeared (maybe another compaction?)
			newSeg.Close()
			e.fs.Remove(path)
			return fmt.Errorf("compaction aborted: segment %d missing", id)
		}
	}

	// 2. Prepare Manifest Update
	newSegInfo := manifest.SegmentInfo{
		ID:       newSegID,
		Level:    1, // L1 (Compacted)
		RowCount: count,
		Path:     filename,
	}

	newSegments := make([]manifest.SegmentInfo, 0, len(e.manifest.Segments)-len(segmentIDs)+1)
	compactedSet := make(map[model.SegmentID]bool)
	for _, id := range segmentIDs {
		compactedSet[id] = true
	}

	for _, sm := range e.manifest.Segments {
		if !compactedSet[sm.ID] {
			newSegments = append(newSegments, sm)
		}
	}
	newSegments = append(newSegments, newSegInfo)

	tempManifest := *e.manifest
	tempManifest.Segments = newSegments

	// Save Manifest FIRST
	if err := manifest.NewStore(e.fs, e.dir).Save(&tempManifest); err != nil {
		newSeg.Close()
		e.fs.Remove(path)
		e.fs.Remove(payloadPath)
		return fmt.Errorf("failed to save manifest: %w", err)
	}

	e.manifest.Segments = newSegments

	// 3. Update Engine State (Create New Snapshot)
	newSnap := currentSnap.Clone()

	// Update PK Index (CAS)
	for pk, m := range moves {
		currentLoc, exists := newSnap.pkIndex.Lookup(pk)
		if exists && currentLoc.SegmentID == m.OldSegID && currentLoc.RowID == model.RowID(m.OldRowID) {
			newSnap.pkIndex = newSnap.pkIndex.Insert(pk, model.Location{
				SegmentID: newSegID,
				RowID:     model.RowID(m.NewRowID),
			})
		}
	}

	// Add new segment
	newSnap.segments[newSegID] = NewRefCountedSegment(newSeg)
	newSnap.tombstones[newSegID] = imetadata.NewLocalBitmap()

	// Remove old segments
	for _, id := range segmentIDs {
		if seg, ok := newSnap.segments[id]; ok {
			seg.DecRef()
		}
		delete(newSnap.segments, id)
		delete(newSnap.tombstones, id)

		oldPath := filepath.Join(e.dir, fmt.Sprintf("segment_%d.bin", id))
		oldPayloadPath := filepath.Join(e.dir, fmt.Sprintf("segment_%d.payload", id))
		if seg, ok := currentSnap.segments[id]; ok {
			seg.SetOnClose(func() {
				e.fs.Remove(oldPath)
				e.fs.Remove(oldPayloadPath)
			})
		}
	}

	newSnap.RebuildSorted()
	e.current.Store(newSnap)
	currentSnap.DecRef() // Release Engine's reference to old snapshot

	dropped = int(totalRows) - int(count)
	created = 1
	return nil
}

// openSegment opens a segment file, detecting its type (Flat or DiskANN) via magic number.
func openSegment(ctx context.Context, st blobstore.BlobStore, name string, c cache.BlockCache, payloadBlob blobstore.Blob) (segment.Segment, error) {
	blob, err := st.Open(ctx, name)
	if err != nil {
		return nil, err
	}

	// Read magic
	b := make([]byte, 4)
	if _, err := blob.ReadAt(b, 0); err != nil {
		blob.Close()
		return nil, err
	}
	magic := binary.LittleEndian.Uint32(b)

	switch magic {
	case flat.MagicNumber:
		opts := []flat.Option{flat.WithBlockCache(c)}
		if payloadBlob != nil {
			opts = append(opts, flat.WithPayloadBlob(payloadBlob))
		}
		return flat.Open(blob, opts...)
	case diskann.MagicNumber:
		opts := []diskann.Option{diskann.WithBlockCache(c)}
		if payloadBlob != nil {
			opts = append(opts, diskann.WithPayloadBlob(payloadBlob))
		}
		return diskann.Open(blob, opts...)
	default:
		blob.Close()
		return nil, fmt.Errorf("unknown segment magic: 0x%x", magic)
	}
}
