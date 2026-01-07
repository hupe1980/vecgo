package wal

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"sync"

	"github.com/hupe1980/vecgo/internal/fs"
)

// Durability controls the durability guarantees of the WAL.
type Durability int

const (
	// DurabilityAsync relies on OS page cache. Fast but risky.
	DurabilityAsync Durability = iota
	// DurabilitySync calls fsync after every write. Slow but safe.
	DurabilitySync
)

const (
	walMagic      = "VECGOWAL" // 8 bytes
	walVersion    = 1          // 4 bytes
	walHeaderSize = 12
)

var (
	ErrIncompatibleVersion = errors.New("incompatible WAL version")
	ErrInvalidHeader       = errors.New("invalid WAL header")
)

type Options struct {
	Durability Durability
}

func DefaultOptions() Options {
	return Options{Durability: DurabilitySync}
}

// WAL manages the write-ahead log file.
type WAL struct {
	mu   sync.Mutex
	fs   fs.FileSystem
	file fs.File
	cw   *countingWriter
	path string
	opts Options

	// Group commit state
	syncedOffset int64      // Offset known to be fsync'd
	syncCond     *sync.Cond // Signals the syncer that there is data to sync
	doneCond     *sync.Cond // Signals waiters that a sync completed
	closed       bool
	lastErr      error // Terminal error encountered by background syncer
	wg           sync.WaitGroup
}

type countingWriter struct {
	w *bufio.Writer
	n int64
}

func (cw *countingWriter) Write(p []byte) (int, error) {
	n, err := cw.w.Write(p)
	cw.n += int64(n)
	return n, err
}

func (cw *countingWriter) Flush() error {
	return cw.w.Flush()
}

// Open opens or creates a WAL at the given path.
func Open(fsys fs.FileSystem, path string, opts Options) (*WAL, error) {
	if fsys == nil {
		fsys = fs.Default
	}
	f, err := fsys.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_RDWR, 0644)
	if err != nil {
		return nil, err
	}

	stat, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, err
	}
	offset := stat.Size()

	// Check/Write Header
	if offset == 0 {
		// New file
		header := make([]byte, walHeaderSize)
		copy(header[0:8], walMagic)
		binary.LittleEndian.PutUint32(header[8:12], uint32(walVersion))
		if _, err := f.Write(header); err != nil {
			f.Close()
			return nil, err
		}
		if err := f.Sync(); err != nil {
			f.Close()
			return nil, err
		}
		offset = walHeaderSize
	} else {
		// Existing file
		if offset < walHeaderSize {
			f.Close()
			return nil, fmt.Errorf("%w: file too small (%d < %d)", ErrInvalidHeader, offset, walHeaderSize)
		}
		header := make([]byte, walHeaderSize)
		if _, err := f.ReadAt(header, 0); err != nil {
			f.Close()
			return nil, err
		}
		if string(header[0:8]) != walMagic {
			f.Close()
			return nil, fmt.Errorf("%w: invalid magic %q", ErrInvalidHeader, header[0:8])
		}
		ver := binary.LittleEndian.Uint32(header[8:12])
		if ver != walVersion {
			f.Close()
			return nil, fmt.Errorf("%w: version %d (expected %d)", ErrIncompatibleVersion, ver, walVersion)
		}
	}

	cw := &countingWriter{
		w: bufio.NewWriter(f),
		n: offset,
	}

	w := &WAL{
		fs:           fsys,
		file:         f,
		cw:           cw,
		path:         path,
		opts:         opts,
		syncedOffset: offset,
	}
	w.syncCond = sync.NewCond(&w.mu)
	w.doneCond = sync.NewCond(&w.mu)

	if opts.Durability == DurabilitySync {
		w.wg.Add(1)
		go w.runSyncer()
	}

	return w, nil
}

// Size returns the current size of the WAL in bytes.
func (w *WAL) Size() int64 {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.cw.n
}

func (w *WAL) runSyncer() {
	defer w.wg.Done()
	w.mu.Lock()
	defer w.mu.Unlock()

	for {
		// Wait until there is data to sync (currOffset > syncedOffset) or we are closed
		for w.cw.n <= w.syncedOffset && !w.closed {
			w.syncCond.Wait()
		}

		// If closed and everything synced, exit
		if w.closed && w.cw.n <= w.syncedOffset {
			return
		}

		// Capture target
		target := w.cw.n

		// Unlock to sync
		w.mu.Unlock()
		err := w.file.Sync()
		w.mu.Lock()

		if err != nil {
			w.lastErr = fmt.Errorf("wal sync failed: %w", err)
			// Wake everyone up so they notice the error
			w.doneCond.Broadcast()
			return
		}

		if target > w.syncedOffset {
			w.syncedOffset = target
		}
		w.doneCond.Broadcast()
	}
}

// Append writes a record to the WAL.
// It respects the configured durability mode.
func (w *WAL) Append(rec *Record) error {
	offset, err := w.AppendAsync(rec)
	if err != nil {
		return err
	}
	if w.opts.Durability == DurabilitySync {
		return w.WaitFor(offset)
	}
	return nil
}

// AppendAsync writes a record to the WAL buffer but does not wait for sync.
// It returns the file offset of the end of the record.
func (w *WAL) AppendAsync(rec *Record) (int64, error) {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.closed {
		return 0, os.ErrClosed
	}
	if w.lastErr != nil {
		return 0, w.lastErr
	}

	if err := rec.Encode(w.cw); err != nil {
		return 0, err
	}
	if err := w.cw.Flush(); err != nil {
		return 0, err
	}

	endOffset := w.cw.n

	if w.opts.Durability == DurabilitySync {
		w.syncCond.Signal()
	}
	return endOffset, nil
}

// WaitFor waits until the WAL is synced up to the given offset.
func (w *WAL) WaitFor(offset int64) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	for w.syncedOffset < offset && !w.closed && w.lastErr == nil {
		w.doneCond.Wait()
	}
	if w.lastErr != nil {
		return w.lastErr
	}
	if w.closed && w.syncedOffset < offset {
		return os.ErrClosed
	}
	return nil
}

// Sync ensures all buffered writes are committed to stable storage.
func (w *WAL) Sync() error {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.closed {
		return os.ErrClosed
	}
	if w.lastErr != nil {
		return w.lastErr
	}

	if err := w.cw.Flush(); err != nil {
		return err
	}

	// If we are in Async mode, we still might want to Sync explicitly.
	// But runSyncer is only running if DurabilitySync.
	// If Async, we do it manually here.
	if w.opts.Durability == DurabilityAsync {
		return w.file.Sync()
	}

	// If Sync mode, we use the group commit mechanism
	target := w.cw.n
	w.syncCond.Signal()
	for w.syncedOffset < target && !w.closed && w.lastErr == nil {
		w.doneCond.Wait()
	}
	if w.lastErr != nil {
		return w.lastErr
	}
	return nil
}

// Close closes the WAL file.
func (w *WAL) Close() error {
	w.mu.Lock()

	if w.closed {
		w.mu.Unlock()
		return os.ErrClosed
	}

	// Flush buffer
	if err := w.cw.Flush(); err != nil {
		w.mu.Unlock()
		w.file.Close()
		return err
	}

	w.closed = true
	w.syncCond.Signal() // Wake up syncer to exit
	w.mu.Unlock()

	w.wg.Wait() // Wait for syncer to finish

	return w.file.Close()
}

// Reader returns a reader for replaying the WAL.
// The caller is responsible for closing the returned file.
func (w *WAL) Reader() (*Reader, error) {
	// Open a separate file handle for reading
	// Use fs abstraction if possible, but we don't store it yet. Added w.fs
	f, err := w.fs.OpenFile(w.path, os.O_RDONLY, 0)
	if err != nil {
		return nil, err
	}
	// Skip header
	if _, err := f.Seek(walHeaderSize, io.SeekStart); err != nil {
		f.Close()
		return nil, err
	}
	return &Reader{f: f, r: bufio.NewReader(f), offset: walHeaderSize}, nil
}

// Reader iterates over WAL records.
type Reader struct {
	f      fs.File
	r      *bufio.Reader
	offset int64
}

// Next reads the next record. Returns io.EOF when done.
func (r *Reader) Next() (*Record, error) {
	rec, n, err := Decode(r.r)
	if err == nil {
		r.offset += n
	}
	return rec, err
}

// Offset returns the current valid offset in the WAL.
func (r *Reader) Offset() int64 {
	return r.offset
}

// Close closes the reader.
func (r *Reader) Close() error {
	return r.f.Close()
}
