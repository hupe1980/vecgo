package persistence

import (
	"errors"
	"fmt"

	"github.com/hupe1980/vecgo/internal/mmap"
)

// ErrMmapUnsupported indicates that mmap isn't supported on this platform.
var ErrMmapUnsupported = errors.New("mmap unsupported")

// MappedFile represents a memory-mapped file.
//
// The returned Bytes() slice aliases the mapped file region.
// Call Close to unmap and close the underlying file.
//
// IMPORTANT: Any slices created as views into Bytes() become invalid after Close.
//
// This is intentionally small and explicit; higher-level loaders should keep the
// mapping alive for as long as they need zero-copy views.
type MappedFile struct {
	f    *mmap.File
	data []byte
}

// Bytes returns the mapped data as a byte slice.
func (m *MappedFile) Bytes() []byte {
	if m == nil {
		return nil
	}
	return m.data
}

// Close unmaps the memory and closes the underlying file.
func (m *MappedFile) Close() error {
	if m == nil {
		return nil
	}
	m.data = nil
	if m.f != nil {
		err := m.f.Close()
		m.f = nil
		return err
	}
	return nil
}

// Madvise advises the kernel about how the memory map will be used.
func (m *MappedFile) Madvise(advice int) error {
	if m == nil || m.f == nil {
		return nil
	}
	return m.f.Madvise(advice)
}

// MmapReadOnly opens path and memory-maps it as read-only.
func MmapReadOnly(path string) (*MappedFile, error) {
	f, err := mmap.Open(path)
	if err != nil {
		return nil, err
	}

	// The previous implementation returned an error for empty files.
	if len(f.Data) == 0 {
		_ = f.Close()
		return nil, fmt.Errorf("mmap: empty file")
	}

	return &MappedFile{f: f, data: f.Data}, nil
}
