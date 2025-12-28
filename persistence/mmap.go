package persistence

import (
	"errors"
	"fmt"
	"reflect"
	"unsafe"

	"golang.org/x/exp/mmap"
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
	r    *mmap.ReaderAt
	data []byte
}

func (m *MappedFile) Bytes() []byte {
	if m == nil {
		return nil
	}
	return m.data
}

func (m *MappedFile) Close() error {
	if m == nil {
		return nil
	}
	m.data = nil
	if m.r != nil {
		err := m.r.Close()
		m.r = nil
		return err
	}
	return nil
}

// MmapReadOnly opens path and memory-maps it as read-only.
func MmapReadOnly(path string) (*MappedFile, error) {
	r, err := mmap.Open(path)
	if err != nil {
		return nil, err
	}
	sz := r.Len()
	if sz <= 0 {
		_ = r.Close()
		return nil, fmt.Errorf("mmap: empty file")
	}

	data, err := readerAtBytes(r)
	if err != nil {
		_ = r.Close()
		return nil, err
	}
	if len(data) != sz {
		_ = r.Close()
		return nil, fmt.Errorf("mmap: unexpected mapping size: got %d, want %d", len(data), sz)
	}
	return &MappedFile{r: r, data: data}, nil
}

func readerAtBytes(r *mmap.ReaderAt) ([]byte, error) {
	if r == nil {
		return nil, fmt.Errorf("mmap: nil reader")
	}
	// golang.org/x/exp/mmap.ReaderAt intentionally exposes only ReaderAt APIs.
	// For true zero-copy loaders we need access to the underlying mapped []byte.
	// This uses reflection+unsafe to read the unexported `data []byte` field.
	//
	// If the upstream implementation changes, this will fail fast with a clear error.
	v := reflect.ValueOf(r)
	if v.Kind() != reflect.Pointer || v.IsNil() {
		return nil, fmt.Errorf("mmap: unexpected reader kind")
	}
	e := v.Elem()
	if e.Kind() != reflect.Struct {
		return nil, fmt.Errorf("mmap: unexpected reader layout")
	}
	f := e.FieldByName("data")
	if !f.IsValid() || f.Kind() != reflect.Slice || f.Type().Elem().Kind() != reflect.Uint8 {
		return nil, fmt.Errorf("mmap: unsupported golang.org/x/exp/mmap.ReaderAt version (missing data field)")
	}
	if !f.CanAddr() {
		return nil, fmt.Errorf("mmap: cannot address reader data")
	}
	return *(*[]byte)(unsafe.Pointer(f.UnsafeAddr())), nil
}
