package blobstore

import (
	"io"
	"path/filepath"

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
func (s *LocalStore) Open(name string) (Blob, error) {
	path := filepath.Join(s.root, name)
	// We use mmap by default for local files as it's the most efficient
	// for random access patterns in vector search.
	m, err := mmap.Open(path)
	if err != nil {
		return nil, err
	}
	return &localBlob{m: m}, nil
}

type localBlob struct {
	m *mmap.Mapping
}

func (b *localBlob) ReadAt(p []byte, off int64) (n int, err error) {
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

func (b *localBlob) Close() error {
	return b.m.Close()
}

func (b *localBlob) Size() int64 {
	return int64(len(b.m.Bytes()))
}

func (b *localBlob) Bytes() ([]byte, error) {
	return b.m.Bytes(), nil
}
