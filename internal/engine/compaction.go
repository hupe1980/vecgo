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
	"github.com/hupe1980/vecgo/internal/cache"
	"github.com/hupe1980/vecgo/internal/manifest"
	imetadata "github.com/hupe1980/vecgo/internal/metadata"
	"github.com/hupe1980/vecgo/internal/quantization"
	"github.com/hupe1980/vecgo/internal/resource"
	"github.com/hupe1980/vecgo/internal/segment"
	"github.com/hupe1980/vecgo/internal/segment/diskann"
	"github.com/hupe1980/vecgo/internal/segment/flat"
	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/model"
)

// Compact merges the specified segments into a new segment.
// It is designed to be concurrency-friendly:
// 1. Holds Lock to snapshot state (segments + tombstones).
// 2. Releases Lock to perform heavy I/O (merge).
// 3. Re-acquires Lock to commit changes (CAS on PK Index).
func (e *Engine) Compact(segmentIDs []model.SegmentID, targetLevel int) error {
	return e.CompactWithContext(context.Background(), segmentIDs, targetLevel)
}

// CompactWithContext performs compaction with a context for cancellation.
func (e *Engine) CompactWithContext(ctx context.Context, segmentIDs []model.SegmentID, targetLevel int) (err error) {
	start := time.Now()
	var dropped, created int
	defer func() {
		e.metrics.OnCompaction(time.Since(start), dropped, created, err)
	}()

	// --- Phase 1: Snapshot State ---
	// We need a snapshot to read from.
	snap, err := e.loadSnapshot()
	if err != nil {
		return err
	}
	defer snap.DecRef()

	// Validate segments and snapshot tombstones
	segments := make([]segment.Segment, 0, len(segmentIDs))
	tombstones := make(map[model.SegmentID]*imetadata.LocalBitmap)

	e.mu.RLock()
	e.tombstonesMu.RLock()
	for _, id := range segmentIDs {
		seg, ok := snap.segments[id]
		if !ok {
			e.tombstonesMu.RUnlock()
			e.mu.RUnlock()
			return fmt.Errorf("segment %d not found in current snapshot", id)
		}
		segments = append(segments, seg)

		if vt, ok := e.tombstones[id]; ok {
			tombstones[id] = vt.ToBitmap(e.lsn.Load())
		}
	}
	e.tombstonesMu.RUnlock()
	e.mu.RUnlock()

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
			_ = f.Close()            // Intentionally ignore: cleanup path
			_ = e.fs.Remove(tmpPath) // Intentionally ignore: best-effort cleanup
		}
	}()

	payloadF, err := e.fs.OpenFile(payloadTmpPath, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0644)
	if err != nil {
		return err
	}
	defer func() {
		if payloadF != nil {
			_ = payloadF.Close()            // Intentionally ignore: cleanup path
			_ = e.fs.Remove(payloadTmpPath) // Intentionally ignore: best-effort cleanup
		}
	}()

	// Calculate total rows to estimate k (partitions) or choose segment type
	var totalRows uint32
	for _, seg := range segments {
		totalRows += seg.RowCount()
	}

	var writer io.Writer = f
	if e.resourceController != nil {
		writer = resource.NewRateLimitedWriter(f, e.resourceController, ctx)
	}

	var (
		addFunc   func(id model.ID, vec []float32, md metadata.Document, payload []byte) error
		flushFunc func() error
	)

	// Heuristic: Use DiskANN for larger segments (> 10k vectors), Flat for smaller.
	// This ensures "best" performance: Flat is faster for small, DiskANN scales for large.
	diskAnnThreshold := uint32(10000)
	if e.compactionConfig.DiskANNThreshold > 0 {
		diskAnnThreshold = uint32(e.compactionConfig.DiskANNThreshold)
	}

	// Track DiskANN writer to get ID→RowID mapping after Write()
	// (DiskANN reorders vectors during BFS optimization)
	var diskannWriter *diskann.Writer

	if totalRows >= diskAnnThreshold {
		opts := e.compactionConfig.DiskANNOptions
		if opts.R == 0 {
			opts = diskann.DefaultOptions()
		}
		// Apply engine-level quantization preference if set
		if e.quantizationType != quantization.TypeNone {
			opts.QuantizationType = e.quantizationType
		}
		opts.ResourceController = e.resourceController

		diskannWriter = diskann.NewWriter(writer, payloadF, uint64(newSegID), e.dim, e.metric, opts)

		addFunc = func(id model.ID, vec []float32, md metadata.Document, payload []byte) error {
			return diskannWriter.Add(id, vec, md, payload)
		}
		flushFunc = func() error {
			if err := diskannWriter.Write(ctx); err != nil {
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

	// Map to track new locations for ID index update
	// ID -> (OldRowID, NewRowID)
	// We need OldRowID to verify validity during commit.
	// Note: For DiskANN, NewRowID is initially 0 and updated after Write()
	// because BFS reordering changes final positions.
	type move struct {
		OldSegID model.SegmentID
		OldRowID uint32
		NewRowID uint32
	}
	moves := make(map[model.ID]move)

	var count uint32
	var minID, maxID model.ID
	first := true

	for i, seg := range segments {
		segID := segmentIDs[i]
		ts := tombstones[segID]

		err := seg.Iterate(func(rowID uint32, id model.ID, vec []float32, md metadata.Document, payload []byte) error {
			// Check tombstone (snapshot)
			if ts != nil && ts.Contains(uint32(rowID)) {
				return nil // Skip deleted
			}

			// Update Range
			if first {
				minID = id
				maxID = id
				first = false
			} else {
				if id < minID {
					minID = id
				}
				if id > maxID {
					maxID = id
				}
			}

			// Write to new segment
			if err := addFunc(id, vec, md, payload); err != nil {
				return err
			}

			// For DiskANN, NewRowID will be updated after Write() via GetIDMapping()
			// For flat, count is the correct final position (when k=1, no reordering)
			moves[id] = move{
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

	// For DiskANN, update NewRowID from actual ID→RowID mapping after BFS reordering
	if diskannWriter != nil {
		idMapping := diskannWriter.GetIDMapping()
		for id, m := range moves {
			if finalRow, ok := idMapping[id]; ok {
				m.NewRowID = finalRow
				moves[id] = m
			}
		}
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
	if b, err := e.store.Open(ctx, payloadFilename); err == nil {
		payloadBlob = b
	} else if !errors.Is(err, blobstore.ErrNotFound) {
		fmt.Printf("Compaction failed: open payload blob: %v\n", err)
		_ = e.fs.Remove(path)        // Intentionally ignore: best-effort cleanup
		_ = e.fs.Remove(payloadPath) // Intentionally ignore: best-effort cleanup
		return err
	}

	newSeg, err := openSegment(ctx, e.store, filename, e.blockCache, payloadBlob)
	if err != nil {
		_ = e.fs.Remove(path)        // Intentionally ignore: best-effort cleanup
		_ = e.fs.Remove(payloadPath) // Intentionally ignore: best-effort cleanup
		return err
	}

	var newSegSize int64
	if stat, err := e.fs.Stat(path); err == nil {
		newSegSize = stat.Size()
		e.metrics.OnThroughput("compaction_write", newSegSize)
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
			_ = newSeg.Close()    // Intentionally ignore: cleanup path
			_ = e.fs.Remove(path) // Intentionally ignore: best-effort cleanup
			return fmt.Errorf("compaction aborted: segment %d missing", id)
		}
	}

	// 2. Prepare Manifest Update
	newSegInfo := manifest.SegmentInfo{
		ID:       newSegID,
		Level:    targetLevel,
		RowCount: count,
		Path:     filename,
		Size:     newSegSize,
		MinID:    minID,
		MaxID:    maxID,
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
	// Critical H14 Fix: Clear PKIndex path in manifest.
	// The existing PK index checkpoint (if any) is now invalid because row IDs have changed.
	// We must ensure recovery rebuilds the index from segments rather than loading a stale checkpoint.
	tempManifest.PKIndex = manifest.PKIndexInfo{}

	// Save Manifest FIRST
	if err := manifest.NewStore(e.manifestStore).Save(&tempManifest); err != nil {
		_ = newSeg.Close()           // Intentionally ignore: cleanup path
		_ = e.fs.Remove(path)        // Intentionally ignore: best-effort cleanup
		_ = e.fs.Remove(payloadPath) // Intentionally ignore: best-effort cleanup
		return fmt.Errorf("failed to save manifest: %w", err)
	}

	e.manifest.Segments = newSegments
	e.manifest.PKIndex = manifest.PKIndexInfo{} // Update memory state to match disk

	// 3. Update Engine State (Create New Snapshot)
	newSnap := currentSnap.Clone()

	// Update ID Index (Global MVCC)
	// We use the current LSN for these updates?
	// Compaction doesn't advance LSN usually, but MVCC needs an LSN.
	// We should probably bump LSN for structural changes or use the current LSN + epsilon?
	// Or maybe Engine.lsn tracks WAL usage strictly?
	// If we use current LSN, concurrent readers at that LSN might see moved rows.
	// Let's use e.lsn.Load(). Since we hold the lock, no new writes are happening with *this* or *newer* LSNs that conflict?
	// Wait, Insert grabs LSN atomic.Add.
	// If we use e.lsn.Load(), it's <= next write LSN.
	// Let's increment LSN for the compaction event to ensure ordering.
	compactionLSN := e.lsn.Add(1)

	var movedCount, skippedCount int
	for id, m := range moves {
		currentLoc, exists := e.pkIndex.Get(id, compactionLSN)
		// NOTE: We check if the KEY is pointing to the OLD location.
		// If concurrent Insert updated it to MemTable, currentLoc would be in Active segment.
		// So we only update if it still points to the segment we compacted.

		isMoved := exists && currentLoc.SegmentID == m.OldSegID && currentLoc.RowID == model.RowID(m.OldRowID)
		if isMoved {
			e.pkIndex.Upsert(id, model.Location{
				SegmentID: newSegID,
				RowID:     model.RowID(m.NewRowID),
			}, compactionLSN)
			movedCount++
		} else {
			skippedCount++
			if e.logger != nil && skippedCount <= 5 {
				e.logger.Warn("Compaction pkIndex skip",
					"id", id,
					"exists", exists,
					"currentSegID", currentLoc.SegmentID,
					"expectedSegID", m.OldSegID,
					"currentRowID", currentLoc.RowID,
					"expectedRowID", m.OldRowID,
				)
			}
		}
	}
	if e.logger != nil {
		e.logger.Info("Compaction pkIndex update", "moved", movedCount, "skipped", skippedCount)
	}

	newSnap.lsn = compactionLSN // Update snapshot LSN?

	// Add new segment
	newSnap.segments[newSegID] = NewRefCountedSegment(newSeg)

	// Remove old segments
	for _, id := range segmentIDs {
		if seg, ok := newSnap.segments[id]; ok {
			seg.DecRef()
		}
		delete(newSnap.segments, id)

		oldPath := filepath.Join(e.dir, fmt.Sprintf("segment_%d.bin", id))
		oldPayloadPath := filepath.Join(e.dir, fmt.Sprintf("segment_%d.payload", id))
		if seg, ok := currentSnap.segments[id]; ok {
			seg.SetOnClose(func() {
				_ = e.fs.Remove(oldPath)        // Intentionally ignore: best-effort cleanup
				_ = e.fs.Remove(oldPayloadPath) // Intentionally ignore: best-effort cleanup
			})
		}
	}

	newSnap.RebuildSorted()

	// Critical Section: Swap Snapshot and update Tombstones
	e.tombstonesMu.Lock()
	e.tombstones[newSegID] = NewVersionedTombstones(int(newSeg.RowCount()))
	for _, id := range segmentIDs {
		delete(e.tombstones, id)
	}
	e.tombstonesMu.Unlock()
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
		_ = blob.Close() // Intentionally ignore: cleanup path
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
		_ = blob.Close() // Intentionally ignore: cleanup path
		return nil, fmt.Errorf("unknown segment magic: 0x%x", magic)
	}
}
