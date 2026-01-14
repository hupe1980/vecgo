package blobstore

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/hupe1980/vecgo/internal/mmap"
)

// LocalStore implements BlobStore using the local file system.
type LocalStore struct {
	root string
}

// NewLocalStore creates a new LocalStore rooted at the given directory.
func NewLocalStore(root string) *LocalStore {
	return &LocalStore{root: root}
}

// Open opens a blob for reading.
func (s *LocalStore) Open(ctx context.Context, name string) (Blob, error) {
	path := filepath.Join(s.root, name)
	// We use mmap by default for local files as it's the most efficient
	// for random access patterns in vector search.
	m, err := mmap.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, fmt.Errorf("%w: %w", ErrNotFound, err)
		}
		return nil, err
	}
	return &localBlob{m: m}, nil
}

// Put writes a blob atomically.
func (s *LocalStore) Put(ctx context.Context, name string, data []byte) error {
	w, err := s.Create(ctx, name)
	if err != nil {
		return err
	}
	if _, err := w.Write(data); err != nil {
		_ = w.Close() // Best effort cleanup
		return err
	}
	return w.Close()
}

// Create creates a new blob for writing.
// It ensures atomicity by writing to a temporary file and renaming on Close.
func (s *LocalStore) Create(ctx context.Context, name string) (WritableBlob, error) {
	path := filepath.Join(s.root, name)
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, err
	}

	// Create a temp file in the same directory to ensure atomic rename works
	tmp, err := os.CreateTemp(dir, "tmp-"+filepath.Base(name)+"-*")
	if err != nil {
		return nil, err
	}

	return &atomicFileWriter{
		f:    tmp,
		path: path,
	}, nil
}

type atomicFileWriter struct {
	f    *os.File
	path string
	done bool
}

func (w *atomicFileWriter) Write(p []byte) (int, error) {
	return w.f.Write(p)
}

func (w *atomicFileWriter) Sync() error {
	return w.f.Sync()
}

func (w *atomicFileWriter) Close() error {
	if w.done {
		return w.f.Close()
	}
	w.done = true

	// Sync to ensure data is on disk before rename
	if err := w.f.Sync(); err != nil {
		_ = w.f.Close()           // Intentionally ignore: cleanup path
		_ = os.Remove(w.f.Name()) // Intentionally ignore: best-effort cleanup
		return err
	}

	if err := w.f.Close(); err != nil {
		_ = os.Remove(w.f.Name()) // Intentionally ignore: best-effort cleanup
		return err
	}

	// Atomic rename
	return os.Rename(w.f.Name(), w.path)
}

// Delete deletes a blob.
func (s *LocalStore) Delete(ctx context.Context, name string) error {
	path := filepath.Join(s.root, name)
	return os.Remove(path)
}

// List returns all blobs matching the prefix.
func (s *LocalStore) List(ctx context.Context, prefix string) ([]string, error) {
	entries, err := os.ReadDir(s.root)
	if err != nil {
		return nil, err
	}
	var matches []string
	for _, entry := range entries {
		if !entry.IsDir() && strings.HasPrefix(entry.Name(), prefix) {
			matches = append(matches, entry.Name())
		}
	}
	return matches, nil
}

type localBlob struct {
	m *mmap.Mapping
}

func (b *localBlob) ReadAt(ctx context.Context, p []byte, off int64) (n int, err error) {
	if len(p) == 0 {
		return 0, nil
	}
	data := b.m.Bytes()
	if off < 0 || off >= int64(len(data)) {
		return 0, io.EOF
	}
	n = copy(p, data[off:])
	if n < len(p) {
		return n, io.EOF
	}
	return n, nil
}

func (b *localBlob) ReadRange(ctx context.Context, off, length int64) (io.ReadCloser, error) {
	data := b.m.Bytes()
	if off < 0 || off >= int64(cap(data)) {
		return io.NopCloser(bytes.NewReader(nil)), io.EOF
	}
	end := off + length
	if end > int64(cap(data)) {
		end = int64(cap(data))
	}
	if end < off {
		return io.NopCloser(bytes.NewReader(nil)), nil
	}

	// Create a reader from the slice
	return io.NopCloser(bytes.NewReader(data[off:end])), nil
}

func (b *localBlob) Close() error {
	return b.m.Close()
}

func (b *localBlob) Size() int64 {
	return int64(len(b.m.Bytes()))
}

func (b *localBlob) Bytes() ([]byte, error) {
	return b.m.Bytes(), nil
}
