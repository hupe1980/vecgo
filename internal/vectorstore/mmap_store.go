package vectorstore

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"unsafe"

	"github.com/hupe1980/vecgo/distance"
	"github.com/hupe1980/vecgo/internal/conv"
	"github.com/hupe1980/vecgo/internal/mmap"
	"github.com/hupe1980/vecgo/model"
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
	data    []float32 // vector data (view into the mmap)
	deleted []uint64  // deletion bitmap (view into the mmap)
	reader  *mmap.Mapping
}

// Close closes the memory-mapped file.
func (s *MmapStore) Close() error {
	if s.reader != nil {
		return s.reader.Close()
	}
	return nil
}

// OpenMmap opens a columnar file and returns a read-only mmap'd store.
//
// The returned io.Closer must be closed when the store is no longer needed.
//
// Example:
//
// store, closer, err := vectorstore.OpenMmap("vectors.col")
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

func parseMmap(reader *mmap.Mapping) (*MmapStore, error) {
	size := reader.Size()
	if size < HeaderSize {
		return nil, fmt.Errorf("columnar: file too small")
	}

	var header FileHeader
	if _, err := header.ReadFrom(bytes.NewReader(reader.Bytes()[:HeaderSize])); err != nil {
		return nil, err
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

func loadVectorData(store *MmapStore, reader *mmap.Mapping, header FileHeader) error {
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
	if offset+size > reader.Size() {
		return fmt.Errorf("columnar: file too small for data")
	}

	// Ensure alignment for float32 (4 bytes)
	if offset%4 != 0 {
		// Fallback to copy if not aligned (rare)
		vecBytes := make([]byte, size)
		copy(vecBytes, reader.Bytes()[offset:offset+size])
		store.data = make([]float32, countInt*dimInt)
		for i := range store.data {
			store.data[i] = *(*float32)(unsafe.Pointer(&vecBytes[i*4])) //nolint:gosec // unsafe is required for mmap access
		}
	} else {
		// Zero-copy access
		store.data = unsafe.Slice((*float32)(unsafe.Pointer(&reader.Bytes()[offset])), countInt*dimInt) //nolint:gosec // unsafe is required for mmap access
	}
	return nil
}

func loadBitmapData(store *MmapStore, reader *mmap.Mapping, header FileHeader) error {
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
	if offset+size > reader.Size() {
		return fmt.Errorf("columnar: file too small for bitmap")
	}

	// Ensure alignment for uint64 (8 bytes)
	if offset%8 != 0 {
		// Fallback to copy
		bitmapBytes := make([]byte, size)
		copy(bitmapBytes, reader.Bytes()[offset:offset+size])
		store.deleted = make([]uint64, bitmapLen)
		for i := range store.deleted {
			store.deleted[i] = binary.LittleEndian.Uint64(bitmapBytes[i*8:])
		}
	} else {
		// Zero-copy access
		store.deleted = unsafe.Slice((*uint64)(unsafe.Pointer(&reader.Bytes()[offset])), bitmapLen) //nolint:gosec // unsafe is required for mmap access
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
func (s *MmapStore) GetVector(id model.RowID) ([]float32, bool) {
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

// ComputeDistance computes the distance between the query and the vector at ID.
// This avoids allocating a slice header for the vector and allows internal optimization.
func (s *MmapStore) ComputeDistance(id model.RowID, query []float32, metric distance.Metric) (float32, bool) {
	if uint64(id) >= s.count {
		return 0, false
	}

	// Check deletion bitmap
	if s.isDeleted(id) {
		return 0, false
	}

	dim := int(s.dim)
	idInt, err := conv.Uint32ToInt(uint32(id))
	if err != nil {
		return 0, false
	}
	start := idInt * dim
	end := start + dim

	if end > len(s.data) {
		return 0, false
	}

	vec := s.data[start:end]

	switch metric {
	case distance.MetricL2:
		return distance.SquaredL2(query, vec), true
	case distance.MetricDot:
		return -distance.Dot(query, vec), true
	case distance.MetricCosine:
		return 0.5 * distance.SquaredL2(query, vec), true
	default:
		return distance.SquaredL2(query, vec), true
	}
}

// GetVectorUnsafe returns the vector without checking deletion status.
func (s *MmapStore) GetVectorUnsafe(id model.RowID) ([]float32, bool) {
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
func (s *MmapStore) SetVector(ctx context.Context, id model.RowID, v []float32) error {
	return fmt.Errorf("columnar: mmap store is read-only")
}

// DeleteVector is not supported on read-only mmap stores.
func (s *MmapStore) DeleteVector(id model.RowID) error {
	return fmt.Errorf("columnar: mmap store is read-only")
}

// IsDeleted returns true if the vector at id is deleted.
func (s *MmapStore) IsDeleted(id model.RowID) bool {
	return s.isDeleted(id)
}

func (s *MmapStore) isDeleted(id model.RowID) bool {
	idU32 := uint32(id)
	bitmapIdx := idU32 / 64
	bitmapIdxInt, err := conv.Uint32ToInt(bitmapIdx)
	if err != nil {
		return false
	}
	if bitmapIdxInt >= len(s.deleted) {
		return false
	}
	bitIdx := idU32 % 64
	return s.deleted[bitmapIdxInt]&(1<<bitIdx) != 0
}

// Iterate calls fn for each live vector. Return false from fn to stop iteration.
func (s *MmapStore) Iterate(fn func(id model.RowID, vec []float32) bool) {
	dim := int(s.dim)

	for id := uint64(0); id < s.count; id++ {
		idU32, err := conv.Uint64ToUint32(id)
		if err != nil {
			break
		}
		rowID := model.RowID(idU32)
		if s.isDeleted(rowID) {
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

		if !fn(rowID, s.data[start:end:end]) {
			break
		}
	}
}

// RawData returns the underlying contiguous vector data.
// The returned slice aliases mmap'd memory; do not modify.
func (s *MmapStore) RawData() []float32 {
	return s.data
}

// OptimizedDistanceComputer returns a function that computes distance without interface overhead.
func (s *MmapStore) OptimizedDistanceComputer(metric distance.Metric) (func(model.RowID, []float32) float32, bool) {
	dim := int(s.dim)
	data := s.data
	count := uint64(len(data) / dim)

	// fast path for L2
	if metric == distance.MetricL2 {
		return func(id model.RowID, q []float32) float32 {
			if uint64(id) >= count {
				return math.MaxFloat32
			}
			idx := int(id) * dim
			return distance.SquaredL2(data[idx:idx+dim], q)
		}, true
	}

	if metric == distance.MetricDot || metric == distance.MetricCosine {
		return func(id model.RowID, q []float32) float32 {
			if uint64(id) >= count {
				return math.MaxFloat32
			}
			idx := int(id) * dim
			return distance.Dot(data[idx:idx+dim], q)
		}, true
	}

	return nil, false
}
