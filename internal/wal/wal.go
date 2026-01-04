package wal

import (
	"bufio"
	"os"
	"sync"
)

// Durability controls the durability guarantees of the WAL.
type Durability int

const (
	// DurabilityAsync relies on OS page cache. Fast but risky.
	DurabilityAsync Durability = iota
	// DurabilitySync calls fsync after every write. Slow but safe.
	DurabilitySync
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
	file *os.File
	cw   *countingWriter
	path string
	opts Options

	// Group commit state
	syncedOffset int64      // Offset known to be fsync'd
	syncCond     *sync.Cond // Signals the syncer that there is data to sync
	doneCond     *sync.Cond // Signals waiters that a sync completed
	closed       bool
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
func Open(path string, opts Options) (*WAL, error) {
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_RDWR, 0644)
	if err != nil {
		return nil, err
	}

	stat, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, err
	}
	offset := stat.Size()

	cw := &countingWriter{
		w: bufio.NewWriter(f),
		n: offset,
	}

	w := &WAL{
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
			// In a real system, handle error. For now, panic to avoid data corruption.
			panic("wal fsync failed: " + err.Error())
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

	for w.syncedOffset < offset && !w.closed {
		w.doneCond.Wait()
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
	for w.syncedOffset < target && !w.closed {
		w.doneCond.Wait()
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
	f, err := os.Open(w.path)
	if err != nil {
		return nil, err
	}
	return &Reader{f: f, r: bufio.NewReader(f)}, nil
}

// Reader iterates over WAL records.
type Reader struct {
	f *os.File
	r *bufio.Reader
}

// Next reads the next record. Returns io.EOF when done.
func (r *Reader) Next() (*Record, error) {
	return Decode(r.r)
}

// Close closes the reader.
func (r *Reader) Close() error {
	return r.f.Close()
}
