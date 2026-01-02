// Package wal provides Write-Ahead Logging for durability and crash recovery.
//
// The WAL ensures that all insert, update, and delete operations are persisted to disk
// before being acknowledged. This provides crash recovery and durability guarantees.
//
// Features:
//   - Individual operation logging (LogInsert, LogUpdate, LogDelete)
//   - Batch operation logging (LogBatchInsert) for efficient bulk writes
//   - Configurable fsync behavior for performance vs durability tradeoff
//   - Checkpoint support for log truncation after snapshots
//   - Sequential ordering via sequence numbers
package wal

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/hupe1980/vecgo/core"
	"github.com/hupe1980/vecgo/metadata"
	"github.com/klauspost/compress/zstd"
)

// WAL provides write-ahead logging for durability.
type WAL struct {
	mu               sync.Mutex
	file             *os.File
	writer           io.Writer     // May be compressed or direct
	bufWriter        *bufio.Writer // Buffered writer for performance
	compressor       *zstd.Encoder
	decompressor     *zstd.Decoder
	seqNum           uint64
	filePath         string
	compressed       bool
	compressionLevel int
	dataOffset       int64 // start of entry stream (after optional header)

	// Auto-checkpoint tracking
	autoCheckpointOps int          // Threshold for operations
	autoCheckpointMB  int          // Threshold for size in MB
	committedOps      int          // Counter for committed operations
	checkpointFunc    func() error // Callback to trigger checkpoint

	// Group commit support (background goroutine lifecycle)
	durabilityMode      DurabilityMode
	groupCommitInterval time.Duration
	groupCommitMaxOps   int
	groupCommitTicker   *time.Ticker
	groupCommitStopCh   chan struct{}  // Shutdown signal for worker goroutine
	groupCommitPending  int            // Operations since last fsync
	groupCommitWg       sync.WaitGroup // Tracks worker goroutine lifecycle

	// Blocking Group Commit
	syncCond        *sync.Cond // Condition variable for blocking group commit
	persistedSeqNum uint64     // Highest sequence number persisted to disk
}

// FilePath returns the path to the WAL file.
func (w *WAL) FilePath() string {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.filePath
}

// New creates a new WAL instance.
func New(optFns ...func(o *Options)) (*WAL, error) {
	opts := DefaultOptions

	for _, fn := range optFns {
		fn(&opts)
	}

	// Ensure directory exists
	if err := os.MkdirAll(opts.Path, 0750); err != nil {
		return nil, fmt.Errorf("failed to create WAL directory: %w", err)
	}

	filePath := filepath.Join(opts.Path, "vecgo.wal")

	// Open or create WAL file (we manage seek explicitly)
	file, err := os.OpenFile(filePath, os.O_CREATE|os.O_RDWR, 0600) //nolint:gosec // G304: Path is configurable
	if err != nil {
		return nil, fmt.Errorf("failed to open WAL file: %w", err)
	}
	st, err := file.Stat()
	if err != nil {
		_ = file.Close()
		return nil, fmt.Errorf("failed to stat WAL file: %w", err)
	}

	w := &WAL{
		file:                file,
		filePath:            filePath,
		compressionLevel:    opts.CompressionLevel,
		autoCheckpointOps:   opts.AutoCheckpointOps,
		autoCheckpointMB:    opts.AutoCheckpointMB,
		committedOps:        0,
		durabilityMode:      opts.DurabilityMode,
		groupCommitInterval: opts.GroupCommitInterval,
		groupCommitMaxOps:   opts.GroupCommitMaxOps,
		groupCommitPending:  0,
	}
	w.syncCond = sync.NewCond(&w.mu)

	if err := w.initializeFile(st, opts); err != nil {
		_ = file.Close()
		return nil, err
	}

	// Position at the start of the entry stream before initializing codecs.
	if _, err := w.file.Seek(w.dataOffset, 0); err != nil {
		_ = w.file.Close()
		return nil, fmt.Errorf("failed to seek WAL data offset: %w", err)
	}

	// Set up compression if enabled
	if w.compressed {
		// Create zstd encoder with specified compression level
		level := zstd.EncoderLevelFromZstd(w.compressionLevel)
		compressor, err := zstd.NewWriter(w.file, zstd.WithEncoderLevel(level))
		if err != nil {
			_ = file.Close()
			return nil, fmt.Errorf("failed to create compressor: %w", err)
		}
		w.compressor = compressor
		w.bufWriter = bufio.NewWriter(compressor)
		w.writer = w.bufWriter

		// Create decompressor for replay
		decompressor, err := zstd.NewReader(nil)
		if err != nil {
			_ = compressor.Close()
			_ = file.Close()
			return nil, fmt.Errorf("failed to create decompressor: %w", err)
		}
		w.decompressor = decompressor
	} else {
		// No compression - use buffered writer directly
		w.bufWriter = bufio.NewWriter(w.file)
		w.writer = w.bufWriter
	}

	// Read existing entries to determine next sequence number
	if err := w.scanForSeqNum(); err != nil {
		_ = file.Close()
		return nil, fmt.Errorf("failed to scan WAL: %w", err)
	}

	// Start group commit goroutine if in GroupCommit mode
	if w.durabilityMode == DurabilityGroupCommit && w.groupCommitInterval > 0 {
		w.groupCommitStopCh = make(chan struct{})
		w.groupCommitTicker = time.NewTicker(w.groupCommitInterval)
		w.groupCommitWg.Add(1)
		go w.groupCommitWorker()
	}

	return w, nil
}

// initializeFile handles the file opening and initialization logic for the WAL.
func (w *WAL) initializeFile(info os.FileInfo, opts Options) error {
	if info.Size() == 0 {
		return w.writeNewHeader(opts)
	}
	return w.readExistingHeader()
}

func (w *WAL) writeNewHeader(opts Options) error {
	hdrLen, err := writeWALHeader(w.file, walHeaderInfo{
		Compressed:       opts.Compress,
		CompressionLevel: opts.CompressionLevel,
	})
	if err != nil {
		return fmt.Errorf("failed to write WAL header: %w", err)
	}
	w.dataOffset = hdrLen
	w.compressed = opts.Compress
	return nil
}

func (w *WAL) readExistingHeader() error {
	hdrInfo, valid, err := readWALHeader(w.file)
	if err != nil {
		return fmt.Errorf("failed to read WAL header: %w", err)
	}
	if !valid {
		return fmt.Errorf("invalid WAL header")
	}
	w.dataOffset = hdrInfo.HeaderLen
	w.compressed = hdrInfo.Compressed
	w.compressionLevel = hdrInfo.CompressionLevel
	return nil
}

// syncIfNeeded performs fsync based on the configured durability mode.
func (w *WAL) syncIfNeeded() error {
	switch w.durabilityMode {
	case DurabilityAsync:
		// No fsync - fastest but least durable
		return nil

	case DurabilitySync:
		// Immediate fsync - slowest but most durable
		return w.file.Sync()

	case DurabilityGroupCommit:
		// Increment pending operations counter
		w.groupCommitPending++
		targetSeq := w.seqNum

		// Trigger immediate fsync if batch size threshold reached
		if w.groupCommitPending >= w.groupCommitMaxOps {
			if err := w.doGroupCommit(); err != nil {
				return err
			}
		} else {
			// Wait for background sync
			// Note: syncCond.Wait() releases w.mu, allowing the background worker
			// (or other writers) to acquire it and perform the sync.
			for w.persistedSeqNum < targetSeq {
				w.syncCond.Wait()
			}
		}
		return nil

	default:
		return nil
	}
}

// doGroupCommit performs the actual fsync and resets the pending counter.
// Caller must hold w.mu.
func (w *WAL) doGroupCommit() error {
	if w.groupCommitPending == 0 {
		return nil
	}

	if err := w.file.Sync(); err != nil {
		return err
	}

	w.groupCommitPending = 0
	w.persistedSeqNum = w.seqNum
	w.syncCond.Broadcast()
	return nil
}

// groupCommitWorker runs in a background goroutine and performs periodic fsync.
func (w *WAL) groupCommitWorker() {
	defer w.groupCommitWg.Done()

	// Safety check: ticker must exist
	if w.groupCommitTicker == nil {
		return
	}

	for {
		select {
		case <-w.groupCommitStopCh:
			// Final fsync before shutdown
			w.mu.Lock()
			_ = w.doGroupCommit()
			w.mu.Unlock()
			return

		case <-w.groupCommitTicker.C:
			w.mu.Lock()
			_ = w.doGroupCommit()
			w.mu.Unlock()
		}
	}
}

// scanForSeqNum scans the WAL to find the highest sequence number.
func (w *WAL) scanForSeqNum() error {
	// Seek to the start of the entry stream for reading
	if _, err := w.file.Seek(w.dataOffset, 0); err != nil {
		return err
	}

	var reader io.Reader
	if w.compressed {
		// Reset decompressor for the file
		if err := w.decompressor.Reset(w.file); err != nil {
			return fmt.Errorf("failed to reset decompressor: %w", err)
		}
		reader = w.decompressor
	} else {
		reader = w.file
	}

	var maxSeqNum uint64

	for {
		var entry Entry
		if err := w.decodeEntry(reader, &entry); err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			// Corrupted entry - stop here
			break
		}
		if entry.SeqNum > maxSeqNum {
			maxSeqNum = entry.SeqNum
		}
	}

	w.seqNum = maxSeqNum

	// Seek back to end for appending
	if _, err := w.file.Seek(0, 2); err != nil {
		return err
	}

	return nil
}

// LogInsert logs an insert operation.
//
// This uses the prepare/commit protocol (two entries) so recovery is atomic.
func (w *WAL) LogInsert(id core.LocalID, vector []float32, data []byte, meta metadata.Metadata) error {
	return w.logOperation(OpPrepareInsert, OpCommitInsert, id, vector, data, meta)
}

// LogUpdate logs an update operation.
//
// This uses the prepare/commit protocol (two entries) so recovery is atomic.
func (w *WAL) LogUpdate(id core.LocalID, vector []float32, data []byte, meta metadata.Metadata) error {
	return w.logOperation(OpPrepareUpdate, OpCommitUpdate, id, vector, data, meta)
}

// LogDelete logs a delete operation.
//
// This uses the prepare/commit protocol (two entries) so recovery is atomic.
func (w *WAL) LogDelete(id core.LocalID) error {
	return w.logOperation(OpPrepareDelete, OpCommitDelete, id, nil, nil, nil)
}

func (w *WAL) logOperation(prepareType, commitType OperationType, id core.LocalID, vector []float32, data []byte, meta metadata.Metadata) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	w.seqNum++
	prepare := Entry{Type: prepareType, ID: id, Vector: vector, Data: data, Metadata: meta, SeqNum: w.seqNum}
	if err := w.encodeEntry(&prepare); err != nil {
		return fmt.Errorf("failed to encode WAL prepare entry: %w", err)
	}

	w.seqNum++
	commit := Entry{Type: commitType, ID: id, SeqNum: w.seqNum}
	if err := w.encodeEntry(&commit); err != nil {
		return fmt.Errorf("failed to encode WAL commit entry: %w", err)
	}
	if err := w.flushLocked(); err != nil {
		return err
	}
	return w.syncCommitLocked()
}

// LogPrepareInsert writes a prepare entry for an insert.
// Prepare entries are NOT durability boundaries; Commit entries are.
func (w *WAL) LogPrepareInsert(id core.LocalID, vector []float32, data []byte, meta metadata.Metadata) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	w.seqNum++
	entry := Entry{Type: OpPrepareInsert, ID: id, Vector: vector, Data: data, Metadata: meta, SeqNum: w.seqNum}
	if err := w.encodeEntry(&entry); err != nil {
		return fmt.Errorf("failed to encode WAL entry: %w", err)
	}
	return nil
}

// LogCommitInsert writes a commit entry for an insert and fsyncs the WAL.
func (w *WAL) LogCommitInsert(id core.LocalID) error {
	return w.logCommit(OpCommitInsert, id)
}

// LogPrepareUpdate writes a prepare entry for an update.
func (w *WAL) LogPrepareUpdate(id core.LocalID, vector []float32, data []byte, meta metadata.Metadata) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	w.seqNum++
	entry := Entry{Type: OpPrepareUpdate, ID: id, Vector: vector, Data: data, Metadata: meta, SeqNum: w.seqNum}
	if err := w.encodeEntry(&entry); err != nil {
		return fmt.Errorf("failed to encode WAL entry: %w", err)
	}
	return nil
}

// LogCommitUpdate writes a commit entry for an update and fsyncs the WAL.
func (w *WAL) LogCommitUpdate(id core.LocalID) error {
	return w.logCommit(OpCommitUpdate, id)
}

// LogPrepareDelete writes a prepare entry for a delete.
func (w *WAL) LogPrepareDelete(id core.LocalID) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	w.seqNum++
	entry := Entry{Type: OpPrepareDelete, ID: id, SeqNum: w.seqNum}
	if err := w.encodeEntry(&entry); err != nil {
		return fmt.Errorf("failed to encode WAL entry: %w", err)
	}
	return nil
}

// LogCommitDelete writes a commit entry for a delete and fsyncs the WAL.
func (w *WAL) LogCommitDelete(id core.LocalID) error {
	return w.logCommit(OpCommitDelete, id)
}

func (w *WAL) logCommit(commitType OperationType, id core.LocalID) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	w.seqNum++
	entry := Entry{Type: commitType, ID: id, SeqNum: w.seqNum}
	if err := w.encodeEntry(&entry); err != nil {
		return fmt.Errorf("failed to encode WAL entry: %w", err)
	}
	if err := w.flushLocked(); err != nil {
		return err
	}
	w.committedOps++
	if err := w.syncCommitLocked(); err != nil {
		return err
	}
	return w.maybeCheckpointLocked()
}

// LogPrepareBatchInsert writes prepare entries for a batch insert.
func (w *WAL) LogPrepareBatchInsert(ids []core.LocalID, vectors [][]float32, dataSlice [][]byte, metadataSlice []metadata.Metadata) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	for i := range ids {
		w.seqNum++
		entry := Entry{Type: OpPrepareInsert, ID: ids[i], Vector: vectors[i], Data: dataSlice[i], Metadata: metadataSlice[i], SeqNum: w.seqNum}
		if err := w.encodeEntry(&entry); err != nil {
			return fmt.Errorf("failed to encode WAL entry %d: %w", i, err)
		}
	}
	return nil
}

// LogCommitBatchInsert writes commit entries for a batch insert and fsyncs once.
func (w *WAL) LogCommitBatchInsert(ids []core.LocalID) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	for i := range ids {
		w.seqNum++
		entry := Entry{Type: OpCommitInsert, ID: ids[i], SeqNum: w.seqNum}
		if err := w.encodeEntry(&entry); err != nil {
			return fmt.Errorf("failed to encode WAL entry %d: %w", i, err)
		}
	}
	if err := w.flushLocked(); err != nil {
		return err
	}
	w.committedOps += len(ids)
	if err := w.syncCommitLocked(); err != nil {
		return err
	}
	return w.maybeCheckpointLocked()
}

// LogBatchInsert logs multiple insert operations efficiently.
//
// This uses the prepare/commit protocol and fsyncs once at the end (depending on Options.Sync).
func (w *WAL) LogBatchInsert(ids []core.LocalID, vectors [][]float32, dataSlice [][]byte, metadataSlice []metadata.Metadata) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	// Prepare all
	for i := range ids {
		w.seqNum++
		entry := Entry{Type: OpPrepareInsert, ID: ids[i], Vector: vectors[i], Data: dataSlice[i], Metadata: metadataSlice[i], SeqNum: w.seqNum}
		if err := w.encodeEntry(&entry); err != nil {
			return fmt.Errorf("failed to encode WAL prepare entry %d: %w", i, err)
		}
	}

	// Commit all
	for i := range ids {
		w.seqNum++
		entry := Entry{Type: OpCommitInsert, ID: ids[i], SeqNum: w.seqNum}
		if err := w.encodeEntry(&entry); err != nil {
			return fmt.Errorf("failed to encode WAL commit entry %d: %w", i, err)
		}
	}
	if err := w.flushLocked(); err != nil {
		return err
	}
	w.committedOps += len(ids)
	return w.syncCommitLocked()
}

// Checkpoint writes a checkpoint marker and truncates the WAL.
// This should be called after a successful snapshot/save.
func (w *WAL) Checkpoint() error {
	w.mu.Lock()
	defer w.mu.Unlock()

	w.seqNum++
	entry := Entry{
		Type:   OpCheckpoint,
		SeqNum: w.seqNum,
	}

	if err := w.encodeEntry(&entry); err != nil {
		return fmt.Errorf("failed to encode checkpoint: %w", err)
	}

	if err := w.flushLocked(); err != nil {
		return err
	}

	// Checkpoint is an explicit durability boundary.
	if err := w.file.Sync(); err != nil {
		return err
	}

	// Truncate the file after checkpoint
	return w.truncate()
}

// truncate truncates the WAL file (called after checkpoint).
func (w *WAL) truncate() error {
	// Flush bufWriter before closing
	if w.bufWriter != nil {
		if err := w.bufWriter.Flush(); err != nil {
			return fmt.Errorf("failed to flush buffer: %w", err)
		}
	}

	// Close compressor if using compression
	if w.compressed && w.compressor != nil {
		if err := w.compressor.Close(); err != nil {
			return fmt.Errorf("failed to close compressor: %w", err)
		}
	}

	// Close current file
	if err := w.file.Close(); err != nil {
		return err
	}

	// Create new empty file
	file, err := os.OpenFile(w.filePath, os.O_CREATE|os.O_RDWR|os.O_TRUNC, 0600)
	if err != nil {
		return fmt.Errorf("failed to truncate WAL file: %w", err)
	}

	w.file = file

	// Always write a self-describing header after truncation.
	hdrLen, err := writeWALHeader(w.file, walHeaderInfo{
		Compressed:       w.compressed,
		CompressionLevel: w.compressionLevel,
	})
	if err != nil {
		_ = w.file.Close()
		return err
	}
	w.dataOffset = hdrLen
	if _, err := w.file.Seek(w.dataOffset, 0); err != nil {
		_ = w.file.Close()
		return fmt.Errorf("failed to seek WAL data offset: %w", err)
	}

	// Recreate compressor and bufWriter if using compression
	if w.compressed {
		level := zstd.EncoderLevelFromZstd(w.compressionLevel)
		compressor, err := zstd.NewWriter(file, zstd.WithEncoderLevel(level))
		if err != nil {
			_ = file.Close()
			return fmt.Errorf("failed to recreate compressor: %w", err)
		}
		w.compressor = compressor
		w.bufWriter = bufio.NewWriter(compressor)
		w.writer = w.bufWriter
	} else {
		w.bufWriter = bufio.NewWriter(file)
		w.writer = w.bufWriter
	}

	w.seqNum = 0

	return nil
}

// Close closes the WAL file gracefully.
//
// This method:
// 1. Signals the group commit worker to stop (if running)
// 2. Waits for the worker to finish (ensuring clean shutdown)
// 3. Performs final fsync to flush any pending entries
// 4. Flushes and closes the file
//
// After Close() returns, the WAL is no longer usable.
func (w *WAL) Close() error {
	w.mu.Lock()
	defer w.mu.Unlock()

	// Check if already closed (idempotency)
	if w.file == nil {
		return nil
	}

	// Stop group commit worker if running (only once)
	if w.groupCommitTicker != nil {
		// Signal worker to stop first
		close(w.groupCommitStopCh)
		w.mu.Unlock()
		w.groupCommitWg.Wait() // Wait for worker to finish (ensures no goroutine leak)
		w.mu.Lock()
		// Now safe to stop and nil the ticker
		w.groupCommitTicker.Stop()
		w.groupCommitTicker = nil
	}

	// Flush bufWriter before closing
	if w.bufWriter != nil {
		if err := w.bufWriter.Flush(); err != nil {
			return fmt.Errorf("failed to flush buffer: %w", err)
		}
	}

	// Close compressor if using compression
	if w.compressed && w.compressor != nil {
		if err := w.compressor.Close(); err != nil {
			return fmt.Errorf("failed to close compressor: %w", err)
		}
	}

	// Close decompressor if it exists
	if w.decompressor != nil {
		w.decompressor.Close()
	}

	err := w.file.Close()
	w.file = nil // Mark as closed
	return err
}

// Len returns the number of entries in the WAL (approximate, for testing).
func (w *WAL) Len() (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()

	// Save current position
	currentPos, err := w.file.Seek(0, 1)
	if err != nil {
		return 0, err
	}

	// Seek to the start of the entry stream
	if _, err := w.file.Seek(w.dataOffset, 0); err != nil {
		return 0, err
	}

	var reader io.Reader
	if w.compressed {
		if err := w.decompressor.Reset(w.file); err != nil {
			return 0, fmt.Errorf("failed to reset decompressor: %w", err)
		}
		reader = w.decompressor
	} else {
		reader = bufio.NewReader(w.file)
	}

	count := 0

	for {
		var entry Entry
		if err := w.decodeEntry(reader, &entry); err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			break
		}
		count++
	}

	// Restore position
	if _, err := w.file.Seek(currentPos, 0); err != nil {
		return count, err
	}

	return count, nil
}

// SetCheckpointCallback sets the function to call when auto-checkpoint is triggered.
// The callback is typically vecgo's SaveToFile method.
func (w *WAL) SetCheckpointCallback(fn func() error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.checkpointFunc = fn
}

// maybeCheckpointLocked checks if auto-checkpoint thresholds are exceeded and triggers checkpoint.
// Must be called with w.mu held.
func (w *WAL) maybeCheckpointLocked() error {
	// Check operation count threshold
	if w.autoCheckpointOps > 0 && w.committedOps >= w.autoCheckpointOps {
		return w.triggerAutoCheckpointLocked()
	}

	// Check file size threshold
	if w.autoCheckpointMB > 0 {
		stat, err := w.file.Stat()
		if err == nil {
			sizeMB := stat.Size() / (1024 * 1024)
			if sizeMB >= int64(w.autoCheckpointMB) {
				return w.triggerAutoCheckpointLocked()
			}
		}
	}

	return nil
}

// triggerAutoCheckpointLocked executes the checkpoint callback.
// Must be called with w.mu held.
func (w *WAL) triggerAutoCheckpointLocked() error {
	if w.checkpointFunc == nil {
		// No checkpoint callback set; skip auto-checkpoint
		return nil
	}

	// Reset operation counter
	w.committedOps = 0

	// Release lock before calling checkpoint (callback may acquire locks)
	w.mu.Unlock()
	err := w.checkpointFunc()
	w.mu.Lock()

	return err
}
