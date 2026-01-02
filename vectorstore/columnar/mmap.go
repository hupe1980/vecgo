package columnar

import (
	"encoding/binary"
	"fmt"
	"io"
	"unsafe"

	"github.com/hupe1980/vecgo/core"
	"github.com/hupe1980/vecgo/internal/conv"
	"github.com/hupe1980/vecgo/internal/mmap"
)

// MmapStore is a read-only columnar store backed by memory-mapped file.
//
// It provides zero-copy access to vectors stored in columnar format,
// with no deserialization overhead. Vectors are lazily loaded from disk
// as pages are accessed.
//
// Thread safety: all read operations are safe for concurrent access.
type MmapStore struct {
	dim     uint32
	count   uint64
	live    uint64
	data    []float32 // vector data (copied from mmap)
	deleted []uint64  // deletion bitmap (copied from mmap)
	reader  *mmap.File
}

// OpenMmap opens a columnar file and returns a read-only mmap'd store.
//
// The returned io.Closer must be closed when the store is no longer needed.
// Uses golang.org/x/exp/mmap for cross-platform memory mapping.
//
// Example:
//
// store, closer, err := columnar.OpenMmap("vectors.col")
//
//	if err != nil {
//	   return err
//	}
//
// defer closer.Close()
//
// vec, ok := store.GetVector(42)
func OpenMmap(filename string) (*MmapStore, io.Closer, error) {
	reader, err := mmap.Open(filename)
	if err != nil {
		return nil, nil, fmt.Errorf("columnar: open mmap: %w", err)
	}

	store, err := parseMmap(reader)
	if err != nil {
		_ = reader.Close()
		return nil, nil, err
	}

	return store, reader, nil
}

func parseMmap(reader *mmap.File) (*MmapStore, error) {
	size := len(reader.Data)
	if size < HeaderSize {
		return nil, fmt.Errorf("columnar: file too small")
	}

	// Read header
	header := *(*FileHeader)(unsafe.Pointer(&reader.Data[0])) //nolint:gosec // unsafe is required for mmap access
	if header.Magic != FormatMagic {
		return nil, fmt.Errorf("columnar: invalid magic number")
	}
	if header.Version != FormatVersion {
		return nil, fmt.Errorf("columnar: unsupported version: %d", header.Version)
	}

	store := &MmapStore{
		dim:    header.Dimension,
		count:  header.Count,
		live:   header.LiveCount,
		reader: reader,
	}

	if err := loadVectorData(store, reader, header); err != nil {
		return nil, err
	}

	if err := loadBitmapData(store, reader, header); err != nil {
		return nil, err
	}

	return store, nil
}

func loadVectorData(store *MmapStore, reader *mmap.File, header FileHeader) error {
	if header.Count == 0 {
		return nil
	}

	offset, err := conv.Uint64ToInt(header.DataOffset)
	if err != nil {
		return err
	}
	countInt, err := conv.Uint64ToInt(header.Count)
	if err != nil {
		return err
	}
	dimInt, err := conv.Uint32ToInt(header.Dimension)
	if err != nil {
		return err
	}
	size := countInt * dimInt * 4
	if offset+size > len(reader.Data) {
		return fmt.Errorf("columnar: file too small for data")
	}

	// Ensure alignment for float32 (4 bytes)
	if offset%4 != 0 {
		// Fallback to copy if not aligned (rare)
		vecBytes := make([]byte, size)
		copy(vecBytes, reader.Data[offset:offset+size])
		store.data = make([]float32, countInt*dimInt)
		for i := range store.data {
			store.data[i] = *(*float32)(unsafe.Pointer(&vecBytes[i*4])) //nolint:gosec // unsafe is required for mmap access
		}
	} else {
		// Zero-copy access
		store.data = unsafe.Slice((*float32)(unsafe.Pointer(&reader.Data[offset])), countInt*dimInt) //nolint:gosec // unsafe is required for mmap access
	}
	return nil
}

func loadBitmapData(store *MmapStore, reader *mmap.File, header FileHeader) error {
	countInt, err := conv.Uint64ToInt(header.Count)
	if err != nil {
		return err
	}
	bitmapLen := (countInt + 63) / 64
	if bitmapLen == 0 {
		return nil
	}

	offset, err := conv.Uint64ToInt(header.BitmapOff)
	if err != nil {
		return err
	}
	size := bitmapLen * 8
	if offset+size > len(reader.Data) {
		return fmt.Errorf("columnar: file too small for bitmap")
	}

	// Ensure alignment for uint64 (8 bytes)
	if offset%8 != 0 {
		// Fallback to copy
		bitmapBytes := make([]byte, size)
		copy(bitmapBytes, reader.Data[offset:offset+size])
		store.deleted = make([]uint64, bitmapLen)
		for i := range store.deleted {
			store.deleted[i] = binary.LittleEndian.Uint64(bitmapBytes[i*8:])
		}
	} else {
		// Zero-copy access
		store.deleted = unsafe.Slice((*uint64)(unsafe.Pointer(&reader.Data[offset])), bitmapLen) //nolint:gosec // unsafe is required for mmap access
	}
	return nil
}

// Dimension returns the vector dimensionality.
func (s *MmapStore) Dimension() int {
	return int(s.dim)
}

// Count returns the total number of vectors (including deleted).
func (s *MmapStore) Count() uint64 {
	return s.count
}

// LiveCount returns the number of non-deleted vectors.
func (s *MmapStore) LiveCount() uint64 {
	return s.live
}

// GetVector returns the vector at the given ID.
// Returns nil, false if the ID is out of bounds or the vector is deleted.
// The returned slice aliases mmap'd memory; do not modify.
func (s *MmapStore) GetVector(id core.LocalID) ([]float32, bool) {
	if uint64(id) >= s.count {
		return nil, false
	}

	// Check deletion bitmap
	if s.isDeleted(id) {
		return nil, false
	}

	dim := int(s.dim)
	idInt, err := conv.Uint32ToInt(uint32(id))
	if err != nil {
		return nil, false
	}
	start := idInt * dim
	end := start + dim

	if end > len(s.data) {
		return nil, false
	}

	return s.data[start:end:end], true
}

// GetVectorUnsafe returns the vector without checking deletion status.
func (s *MmapStore) GetVectorUnsafe(id core.LocalID) ([]float32, bool) {
	if uint64(id) >= s.count {
		return nil, false
	}

	dim := int(s.dim)
	idInt, err := conv.Uint32ToInt(uint32(id))
	if err != nil {
		return nil, false
	}
	start := idInt * dim
	end := start + dim

	if end > len(s.data) {
		return nil, false
	}

	return s.data[start:end:end], true
}

// SetVector is not supported on read-only mmap stores.
func (s *MmapStore) SetVector(id core.LocalID, v []float32) error {
	return fmt.Errorf("columnar: mmap store is read-only")
}

// DeleteVector is not supported on read-only mmap stores.
func (s *MmapStore) DeleteVector(id core.LocalID) error {
	return fmt.Errorf("columnar: mmap store is read-only")
}

// IsDeleted returns true if the vector at id is deleted.
func (s *MmapStore) IsDeleted(id core.LocalID) bool {
	return s.isDeleted(id)
}

func (s *MmapStore) isDeleted(id core.LocalID) bool {
	bitmapIdx := id / 64
	bitmapIdxInt, err := conv.Uint32ToInt(uint32(bitmapIdx))
	if err != nil {
		return false
	}
	if bitmapIdxInt >= len(s.deleted) {
		return false
	}
	bitIdx := id % 64
	return s.deleted[bitmapIdxInt]&(1<<bitIdx) != 0
}

// Iterate calls fn for each live vector. Return false from fn to stop iteration.
func (s *MmapStore) Iterate(fn func(id core.LocalID, vec []float32) bool) {
	dim := int(s.dim)

	for id := uint64(0); id < s.count; id++ {
		idU32, err := conv.Uint64ToUint32(id)
		if err != nil {
			break
		}
		localID := core.LocalID(idU32)
		if s.isDeleted(localID) {
			continue
		}

		idInt, err := conv.Uint64ToInt(id)
		if err != nil {
			break
		}
		start := idInt * dim
		end := start + dim
		if end > len(s.data) {
			break
		}

		if !fn(localID, s.data[start:end:end]) {
			break
		}
	}
}

// RawData returns the underlying contiguous vector data.
// The returned slice aliases mmap'd memory; do not modify.
func (s *MmapStore) RawData() []float32 {
	return s.data
}
