// Package persistence provides unified persistence management for vecgo.
//
// The Manager coordinates snapshots, WAL, and recovery operations in a single
// abstraction, eliminating code duplication and ensuring consistent atomicity.
package persistence

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"

	"github.com/hupe1980/vecgo/codec"
	"github.com/hupe1980/vecgo/wal"
)

var (
	// ErrManagerClosed is returned when operations are attempted on a closed manager.
	ErrManagerClosed = errors.New("persistence manager is closed")

	// ErrNoWAL is returned when WAL operations are attempted without WAL configured.
	ErrNoWAL = errors.New("WAL not configured")

	// ErrNoSnapshotPath is returned when snapshot operations require a path but none is set.
	ErrNoSnapshotPath = errors.New("snapshot path not configured")
)

// Snapshotable represents a component that can be saved to a snapshot.
type Snapshotable interface {
	// Save writes the component state to w.
	// The context allows cancellation of long-running snapshot operations.
	Save(ctx context.Context, w io.Writer) error
}

// SnapshotLoader can load state from a snapshot file.
type SnapshotLoader interface {
	// LoadSnapshot loads state from the given file path.
	// The context allows cancellation of long-running load operations.
	LoadSnapshot(ctx context.Context, path string) error
}

// WALReplayer can replay WAL entries to restore state.
type WALReplayer interface {
	// ReplayEntry applies a single WAL entry to restore state.
	// The context allows cancellation during long replay sequences.
	ReplayEntry(ctx context.Context, entry wal.Entry) error
}

// ManagerOptions configures the persistence manager.
type ManagerOptions struct {
	// SnapshotPath is the path for snapshot files (optional).
	SnapshotPath string

	// WALPath is the path for WAL files (optional, enables WAL if set).
	WALPath string

	// WALOptions are additional options for WAL configuration.
	WALOptions []func(*wal.Options)

	// Codec is used for serializing snapshot data.
	Codec codec.Codec

	// AutoCheckpoint enables automatic WAL checkpointing after snapshots.
	AutoCheckpoint bool
}

// Manager coordinates all persistence operations (snapshots, WAL, recovery).
//
// It provides a unified interface for:
//   - Atomic snapshot creation with WAL coordination
//   - Unified recovery (snapshot + WAL replay)
//   - WAL append operations with durability guarantees
//
// The Manager is thread-safe and can be used concurrently.
type Manager struct {
	snapshotPath   string
	walPath        string
	wal            *wal.WAL
	codec          codec.Codec
	autoCheckpoint bool

	// Lifecycle
	mu     sync.RWMutex
	closed bool
}

// NewManager creates a new persistence manager with the given options.
//
// If WALPath is set, a new WAL is created. If SnapshotPath is set, snapshots
// will be written to that path.
func NewManager(opts ManagerOptions) (*Manager, error) {
	pm := &Manager{
		snapshotPath:   opts.SnapshotPath,
		walPath:        opts.WALPath,
		codec:          opts.Codec,
		autoCheckpoint: opts.AutoCheckpoint,
	}

	// Set default codec if not provided
	if pm.codec == nil {
		pm.codec = codec.Default
	}

	// Initialize WAL if path provided
	if opts.WALPath != "" {
		// Prepare WAL options
		walOptFns := append([]func(*wal.Options){
			func(o *wal.Options) {
				o.Path = opts.WALPath
			},
		}, opts.WALOptions...)

		w, err := wal.New(walOptFns...)
		if err != nil {
			return nil, fmt.Errorf("persistence: failed to create WAL: %w", err)
		}
		pm.wal = w
	}

	return pm, nil
}

// NewManagerWithWAL creates a manager using an existing WAL instance.
// This is useful when the WAL is created elsewhere (e.g., in vecgo.go).
func NewManagerWithWAL(snapshotPath string, w *wal.WAL, c codec.Codec) *Manager {
	if c == nil {
		c = codec.Default
	}
	return &Manager{
		snapshotPath:   snapshotPath,
		wal:            w,
		codec:          c,
		autoCheckpoint: true,
	}
}

// WAL returns the underlying WAL instance, or nil if WAL is not configured.
func (pm *Manager) WAL() *wal.WAL {
	pm.mu.RLock()
	defer pm.mu.RUnlock()
	return pm.wal
}

// SnapshotPath returns the configured snapshot path.
func (pm *Manager) SnapshotPath() string {
	pm.mu.RLock()
	defer pm.mu.RUnlock()
	return pm.snapshotPath
}

// SetSnapshotPath updates the snapshot path.
func (pm *Manager) SetSnapshotPath(path string) {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	pm.snapshotPath = path
}

// Codec returns the configured codec.
func (pm *Manager) Codec() codec.Codec {
	pm.mu.RLock()
	defer pm.mu.RUnlock()
	return pm.codec
}

// Snapshot saves state atomically and optionally checkpoints the WAL.
//
// The snapshot is written to a temporary file first, then atomically renamed
// to the final path. If WAL is enabled and autoCheckpoint is true, the WAL
// is checkpointed after a successful snapshot.
//
// The context allows cancellation of the snapshot operation.
func (pm *Manager) Snapshot(ctx context.Context, writeFunc func(ctx context.Context, w io.Writer) error) error {
	pm.mu.RLock()
	if pm.closed {
		pm.mu.RUnlock()
		return ErrManagerClosed
	}
	snapshotPath := pm.snapshotPath
	w := pm.wal
	autoCheckpoint := pm.autoCheckpoint
	pm.mu.RUnlock()

	if snapshotPath == "" {
		return ErrNoSnapshotPath
	}

	// Check context before starting
	if err := ctx.Err(); err != nil {
		return err
	}

	// Save snapshot atomically
	if err := SaveToFile(snapshotPath, func(w io.Writer) error {
		return writeFunc(ctx, w)
	}); err != nil {
		return fmt.Errorf("persistence: snapshot failed: %w", err)
	}

	// Checkpoint WAL (truncate old entries) if enabled
	if w != nil && autoCheckpoint {
		if err := w.Checkpoint(); err != nil {
			return fmt.Errorf("persistence: WAL checkpoint failed: %w", err)
		}
	}

	return nil
}

// SnapshotToPath saves state to a specific path (not the default snapshotPath).
// This is useful for creating named snapshots or backups.
func (pm *Manager) SnapshotToPath(ctx context.Context, path string, writeFunc func(ctx context.Context, w io.Writer) error) error {
	pm.mu.RLock()
	if pm.closed {
		pm.mu.RUnlock()
		return ErrManagerClosed
	}
	pm.mu.RUnlock()

	// Check context before starting
	if err := ctx.Err(); err != nil {
		return err
	}

	// Save snapshot atomically
	if err := SaveToFile(path, func(w io.Writer) error {
		return writeFunc(ctx, w)
	}); err != nil {
		return fmt.Errorf("persistence: snapshot to %s failed: %w", path, err)
	}

	return nil
}

// Recover restores state from snapshot + WAL replay.
//
// Recovery order:
//  1. Load snapshot (if exists at snapshotPath)
//  2. Replay WAL entries after snapshot (if WAL is enabled)
//
// The loader/replayer interfaces allow customization of how state is restored.
// The context allows cancellation of long-running recovery operations.
func (pm *Manager) Recover(ctx context.Context, loader SnapshotLoader, replayer WALReplayer) error {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	if pm.closed {
		return ErrManagerClosed
	}

	// Check context before starting
	if err := ctx.Err(); err != nil {
		return err
	}

	// Step 1: Load snapshot (if exists)
	if pm.snapshotPath != "" {
		if _, err := os.Stat(pm.snapshotPath); err == nil {
			if err := loader.LoadSnapshot(ctx, pm.snapshotPath); err != nil {
				return fmt.Errorf("persistence: snapshot load failed: %w", err)
			}
		} else if !os.IsNotExist(err) {
			return fmt.Errorf("persistence: failed to check snapshot: %w", err)
		}
	}

	// Check context between phases
	if err := ctx.Err(); err != nil {
		return err
	}

	// Step 2: Replay WAL entries after snapshot
	if pm.wal != nil {
		if err := pm.wal.ReplayCommitted(func(entry wal.Entry) error {
			// Check context for each entry
			if err := ctx.Err(); err != nil {
				return err
			}
			return replayer.ReplayEntry(ctx, entry)
		}); err != nil {
			return fmt.Errorf("persistence: WAL replay failed: %w", err)
		}
	}

	return nil
}

// RecoverFromPath loads a snapshot from a specific path (ignoring snapshotPath).
// Does not replay WAL.
func (pm *Manager) RecoverFromPath(ctx context.Context, path string, loader SnapshotLoader) error {
	pm.mu.RLock()
	if pm.closed {
		pm.mu.RUnlock()
		return ErrManagerClosed
	}
	pm.mu.RUnlock()

	// Check context before starting
	if err := ctx.Err(); err != nil {
		return err
	}

	if _, err := os.Stat(path); err != nil {
		return fmt.Errorf("persistence: snapshot not found at %s: %w", path, err)
	}

	if err := loader.LoadSnapshot(ctx, path); err != nil {
		return fmt.Errorf("persistence: snapshot load from %s failed: %w", path, err)
	}

	return nil
}

// Checkpoint creates a WAL checkpoint (truncates committed entries).
// This should be called after saving a snapshot.
func (pm *Manager) Checkpoint() error {
	pm.mu.RLock()
	if pm.closed {
		pm.mu.RUnlock()
		return ErrManagerClosed
	}
	w := pm.wal
	pm.mu.RUnlock()

	if w == nil {
		return ErrNoWAL
	}

	return w.Checkpoint()
}

// SetCheckpointCallback sets a callback that is invoked when the WAL
// determines that a checkpoint should be performed (based on auto-checkpoint thresholds).
func (pm *Manager) SetCheckpointCallback(callback func() error) {
	pm.mu.RLock()
	w := pm.wal
	pm.mu.RUnlock()

	if w != nil {
		w.SetCheckpointCallback(callback)
	}
}

// Close shuts down the persistence manager and releases resources.
// This closes the WAL if it was created by the manager.
func (pm *Manager) Close() error {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	if pm.closed {
		return nil
	}
	pm.closed = true

	if pm.wal != nil {
		return pm.wal.Close()
	}
	return nil
}

// AtomicSaveToDir saves multiple files atomically to a directory.
// All files are written to temp files first, then renamed together.
// This ensures either all files are saved or none are.
//
// Usage:
//
//	err := pm.AtomicSaveToDir("/path/to/index", map[string]func(io.Writer) error{
//	    "graph.bin": func(w io.Writer) error { return writeGraph(w) },
//	    "meta.bin":  func(w io.Writer) error { return writeMeta(w) },
//	})
func AtomicSaveToDir(dir string, files map[string]func(io.Writer) error) error {
	// Ensure directory exists
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("persistence: failed to create directory %s: %w", dir, err)
	}

	// Track temp files for cleanup on error
	tempFiles := make([]string, 0, len(files))
	defer func() {
		// Cleanup temp files on error
		for _, tmp := range tempFiles {
			_ = os.Remove(tmp)
		}
	}()

	// Write each file to a temp file
	type fileMapping struct {
		temp   string
		target string
	}
	mappings := make([]fileMapping, 0, len(files))

	for filename, writeFunc := range files {
		target := filepath.Join(dir, filename)

		// Create temp file in same directory for atomic rename
		tmp, err := os.CreateTemp(dir, filename+".tmp-*")
		if err != nil {
			return fmt.Errorf("persistence: failed to create temp file for %s: %w", filename, err)
		}
		tempFiles = append(tempFiles, tmp.Name())

		// Write content
		if err := writeFunc(tmp); err != nil {
			_ = tmp.Close()
			return fmt.Errorf("persistence: failed to write %s: %w", filename, err)
		}

		// Sync and close
		if err := tmp.Sync(); err != nil {
			_ = tmp.Close()
			return fmt.Errorf("persistence: failed to sync %s: %w", filename, err)
		}
		if err := tmp.Close(); err != nil {
			return fmt.Errorf("persistence: failed to close %s: %w", filename, err)
		}

		mappings = append(mappings, fileMapping{temp: tmp.Name(), target: target})
	}

	// Rename all temp files to final names (atomic on most filesystems)
	for _, m := range mappings {
		if err := os.Rename(m.temp, m.target); err != nil {
			return fmt.Errorf("persistence: failed to rename %s: %w", m.target, err)
		}
	}

	// Clear temp files list (rename succeeded)
	tempFiles = nil

	// Best-effort: fsync directory
	if d, err := os.Open(dir); err == nil {
		_ = d.Sync()
		_ = d.Close()
	}

	return nil
}
