package mmap

import (
	"io"
	"os"
	"sync/atomic"
)

// Mapping represents a memory-mapped file.
// It owns the underlying byte slice and is responsible for unmapping it.
type Mapping struct {
	data   []byte
	size   int
	closed atomic.Bool
	// unmap is the platform-specific function to unmap the memory.
	unmap func([]byte) error
}

// Open maps the file at path into memory.
// The file is mapped as read-only.
func Open(path string) (*Mapping, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	fi, err := f.Stat()
	if err != nil {
		return nil, err
	}

	size := fi.Size()
	if size == 0 {
		return &Mapping{data: nil, size: 0}, nil
	}
	if size < 0 {
		return nil, ErrInvalidSize
	}

	// Platform-specific mapping
	data, unmapFunc, err := osMap(f, int(size))
	if err != nil {
		return nil, err
	}

	m := &Mapping{
		data:  data,
		size:  int(size),
		unmap: unmapFunc,
	}

	return m, nil
}

// Close unmaps the memory. It is idempotent.
func (m *Mapping) Close() error {
	if m.closed.Swap(true) {
		return nil // Already closed
	}
	if m.unmap != nil && m.data != nil {
		return m.unmap(m.data)
	}
	return nil
}

// Bytes returns the underlying byte slice.
// Warning: The slice is valid only until Close() is called.
// Accessing the slice after Close() results in undefined behavior (likely a crash).
func (m *Mapping) Bytes() []byte {
	if m.closed.Load() {
		return nil
	}
	return m.data
}

// Size returns the size of the mapping in bytes.
func (m *Mapping) Size() int {
	return m.size
}

// Advise provides hints to the kernel about how the memory will be accessed.
func (m *Mapping) Advise(pattern AccessPattern) error {
	if m.closed.Load() {
		return ErrClosed
	}
	if m.data == nil {
		return nil
	}
	return osAdvise(m.data, pattern)
}

// ReadAt implements io.ReaderAt.
func (m *Mapping) ReadAt(p []byte, off int64) (n int, err error) {
	if m.closed.Load() {
		return 0, ErrClosed
	}
	if off < 0 {
		return 0, ErrInvalidOffset
	}
	if off >= int64(len(m.data)) {
		return 0, io.EOF
	}
	n = copy(p, m.data[off:])
	if n < len(p) {
		return n, io.EOF
	}
	return n, nil
}
