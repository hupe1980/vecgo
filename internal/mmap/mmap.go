package mmap

import (
	"errors"
	"io"
	"os"
)

// File represents a memory-mapped file.
type File struct {
	Data []byte
	f    *os.File
}

// Close unmaps the memory and closes the underlying file.
func (m *File) Close() error {
	if m == nil {
		return nil
	}
	var err error
	if m.Data != nil {
		err = munmap(m.Data)
		m.Data = nil
	}
	if m.f != nil {
		if closeErr := m.f.Close(); closeErr != nil && err == nil {
			err = closeErr
		}
		m.f = nil
	}
	return err
}

// Open maps the file at path into memory as read-only.
func Open(path string) (*File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}

	fi, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, err
	}

	size := fi.Size()
	if size == 0 {
		return &File{Data: nil, f: f}, nil
	}

	if size < 0 {
		f.Close()
		return nil, errors.New("mmap: file size is negative")
	}

	data, err := mmap(f, int(size))
	if err != nil {
		f.Close()
		return nil, err
	}

	return &File{Data: data, f: f}, nil
}

// ReaderAt implements io.ReaderAt on a memory-mapped file.
func (m *File) ReadAt(p []byte, off int64) (n int, err error) {
	if m.Data == nil {
		return 0, io.EOF
	}
	if off < 0 || off >= int64(len(m.Data)) {
		return 0, io.EOF
	}
	n = copy(p, m.Data[off:])
	if n < len(p) {
		return n, io.EOF
	}
	return n, nil
}
