package columnar

import (
	"encoding/binary"
	"hash/crc32"
	"io"
	"sync"
	"sync/atomic"
	"unsafe"

	"github.com/hupe1980/vecgo/internal/mem"
	"github.com/hupe1980/vecgo/vectorstore"
)

// Store is a high-performance columnar vector store.
//
// Vectors are stored contiguously in a single []float32 slice, providing
// excellent cache locality for sequential access and SIMD operations.
//
// The store supports:
//   - O(1) vector access by ID
//   - Append-only writes (new vectors get sequential IDs)
//   - Soft deletes with bitmap tracking
//   - Compaction to reclaim deleted space
//   - Mmap loading for zero-copy access
//
// Thread safety: concurrent reads are safe; writes require external synchronization.
type Store struct {
	dim uint32

	// Atomic for lock-free read access
	data atomic.Pointer[[]float32] // Contiguous vector storage: vectors[id] = data[id*dim : (id+1)*dim]

	mu       sync.RWMutex
	deleted  []uint64 // Bitmap: bit i = 1 means vector i is deleted
	versions []uint64 // Version numbers for each vector (optional)
	count    uint32   // Total vectors (including deleted)
	live     uint32   // Live (non-deleted) vectors

	// Mmap source (nil if in-memory)
	mmapData []byte
}

// New creates a new in-memory columnar store with the given dimension.
func New(dim int) *Store {
	if dim <= 0 {
		dim = 1
	}
	// Pre-allocate for ~1K vectors using aligned memory
	data := mem.AllocAlignedFloat32(1024 * dim)
	data = data[:0] // Reset length to 0

	s := &Store{
		dim:     uint32(dim),
		deleted: make([]uint64, 0, 16), // ~1K vectors per uint64 block
	}
	s.data.Store(&data)
	return s
}

// NewWithCapacity creates a new store with pre-allocated capacity.
func NewWithCapacity(dim, capacity int) *Store {
	if dim <= 0 {
		dim = 1
	}
	if capacity < 0 {
		capacity = 0
	}
	// Use aligned memory
	data := mem.AllocAlignedFloat32(capacity * dim)
	data = data[:0] // Reset length to 0

	bitmapCap := (capacity + 63) / 64
	s := &Store{
		dim:     uint32(dim),
		deleted: make([]uint64, 0, bitmapCap),
	}
	s.data.Store(&data)
	return s
}

// Dimension returns the vector dimensionality.
func (s *Store) Dimension() int {
	return int(s.dim)
}

// Count returns the total number of vectors (including deleted).
func (s *Store) Count() uint32 {
	s.mu.RLock()
	c := s.count
	s.mu.RUnlock()
	return c
}

// LiveCount returns the number of non-deleted vectors.
func (s *Store) LiveCount() uint32 {
	s.mu.RLock()
	l := s.live
	s.mu.RUnlock()
	return l
}

// GetVector returns the vector at the given ID.
// Returns nil, false if the ID is out of bounds or the vector is deleted.
// The returned slice may alias internal memory; do not modify.
func (s *Store) GetVector(id uint32) ([]float32, bool) {
	data := s.data.Load()
	if data == nil {
		return nil, false
	}

	dim := int(s.dim)
	start := int(id) * dim
	end := start + dim

	if end > len(*data) {
		return nil, false
	}

	// Check deletion bitmap
	s.mu.RLock()
	deleted := s.isDeletedLocked(id)
	s.mu.RUnlock()

	if deleted {
		return nil, false
	}

	return (*data)[start:end:end], true
}

// GetVectorUnsafe returns the vector without checking deletion status.
// Use with caution - mainly for internal iteration during compaction.
func (s *Store) GetVectorUnsafe(id uint32) ([]float32, bool) {
	data := s.data.Load()
	if data == nil {
		return nil, false
	}

	dim := int(s.dim)
	start := int(id) * dim
	end := start + dim

	if end > len(*data) {
		return nil, false
	}

	return (*data)[start:end:end], true
}

// SetVector sets (or replaces) the vector at the given ID.
// This is part of the vectorstore.Store interface.
func (s *Store) SetVector(id uint32, v []float32) error {
	if len(v) != int(s.dim) {
		return vectorstore.ErrWrongDimension
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	data := s.data.Load()
	dim := int(s.dim)

	// If ID is beyond current capacity, we need to extend
	requiredLen := int(id+1) * dim
	currentData := *data

	if requiredLen > len(currentData) {
		// Extend the data slice
		newCap := max(requiredLen*2, len(currentData)*2)
		newData := mem.AllocAlignedFloat32(newCap)
		newData = newData[:requiredLen]
		copy(newData, currentData)
		s.data.Store(&newData)
		currentData = newData

		// Extend deletion bitmap if needed
		requiredBitmapLen := int(id+63) / 64
		for len(s.deleted) < requiredBitmapLen+1 {
			s.deleted = append(s.deleted, 0)
		}

		// Update count if we're adding a new vector
		if id >= s.count {
			s.count = id + 1
		}
	}

	// Copy vector data
	start := int(id) * dim
	copy(currentData[start:], v)

	// Clear deletion bit if it was set
	if s.isDeletedLocked(id) {
		s.setDeletedLocked(id, false)
		s.live++
	} else if id == s.count-1 {
		// New vector
		s.live++
	}

	return nil
}

// Append adds a new vector and returns its ID.
func (s *Store) Append(v []float32) (uint32, error) {
	if len(v) != int(s.dim) {
		return 0, vectorstore.ErrWrongDimension
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	id := s.count
	dim := int(s.dim)

	data := s.data.Load()
	currentData := *data

	// Extend data slice
	requiredLen := int(id+1) * dim
	if requiredLen > cap(currentData) {
		newCap := max(requiredLen*2, len(currentData)*2)
		// Use aligned allocator
		newData := mem.AllocAlignedFloat32(newCap)
		newData = newData[:requiredLen] // Set length
		copy(newData, currentData)
		s.data.Store(&newData)
		currentData = newData
	} else {
		newData := currentData[:requiredLen]
		s.data.Store(&newData)
		currentData = newData
	}

	// Copy vector data
	start := int(id) * dim
	copy(currentData[start:], v)

	// Extend deletion bitmap if needed
	bitmapIdx := id / 64
	for uint32(len(s.deleted)) <= bitmapIdx {
		s.deleted = append(s.deleted, 0)
	}

	s.count++
	s.live++

	return id, nil
}

// DeleteVector marks a vector as deleted.
func (s *Store) DeleteVector(id uint32) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if id >= s.count {
		return ErrOutOfBounds
	}

	if !s.isDeletedLocked(id) {
		s.setDeletedLocked(id, true)
		s.live--
	}

	return nil
}

// IsDeleted returns true if the vector at id is deleted.
func (s *Store) IsDeleted(id uint32) bool {
	s.mu.RLock()
	d := s.isDeletedLocked(id)
	s.mu.RUnlock()
	return d
}

func (s *Store) isDeletedLocked(id uint32) bool {
	bitmapIdx := id / 64
	if int(bitmapIdx) >= len(s.deleted) {
		return false
	}
	bitIdx := id % 64
	return s.deleted[bitmapIdx]&(1<<bitIdx) != 0
}

func (s *Store) setDeletedLocked(id uint32, deleted bool) {
	bitmapIdx := id / 64
	for uint32(len(s.deleted)) <= bitmapIdx {
		s.deleted = append(s.deleted, 0)
	}
	bitIdx := id % 64
	if deleted {
		s.deleted[bitmapIdx] |= 1 << bitIdx
	} else {
		s.deleted[bitmapIdx] &^= 1 << bitIdx
	}
}

// Compact removes deleted vectors and defragments the store.
// Returns a map from old IDs to new IDs for live vectors.
// Deleted vectors are not included in the map.
func (s *Store) Compact() map[uint32]uint32 {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.live == s.count {
		// Nothing to compact
		return nil
	}

	dim := int(s.dim)
	oldData := *s.data.Load()
	newData := mem.AllocAlignedFloat32(int(s.live) * dim)
	newData = newData[:0]
	idMap := make(map[uint32]uint32, s.live)

	var newID uint32
	for oldID := uint32(0); oldID < s.count; oldID++ {
		if s.isDeletedLocked(oldID) {
			continue
		}

		start := int(oldID) * dim
		end := start + dim
		newData = append(newData, oldData[start:end]...)
		idMap[oldID] = newID
		newID++
	}

	s.data.Store(&newData)
	s.count = newID
	s.live = newID
	s.deleted = make([]uint64, (newID+63)/64)

	return idMap
}

// Iterate calls fn for each live vector. Return false from fn to stop iteration.
func (s *Store) Iterate(fn func(id uint32, vec []float32) bool) {
	s.mu.RLock()
	count := s.count
	s.mu.RUnlock()

	data := s.data.Load()
	if data == nil {
		return
	}

	dim := int(s.dim)

	for id := uint32(0); id < count; id++ {
		s.mu.RLock()
		deleted := s.isDeletedLocked(id)
		s.mu.RUnlock()

		if deleted {
			continue
		}

		start := int(id) * dim
		end := start + dim
		if end > len(*data) {
			break
		}

		if !fn(id, (*data)[start:end:end]) {
			break
		}
	}
}

// WriteTo writes the store to w in columnar format.
func (s *Store) WriteTo(w io.Writer) (int64, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	data := s.data.Load()
	if data == nil {
		return 0, nil
	}

	// Prepare header
	header := FileHeader{
		Magic:      FormatMagic,
		Version:    FormatVersion,
		Flags:      0,
		Dimension:  s.dim,
		Count:      uint64(s.count),
		LiveCount:  uint64(s.live),
		DataOffset: HeaderSize,
	}
	header.BitmapOff = header.DataOffset + uint64(header.VectorDataSize())

	var written int64

	// Write header
	n, err := header.WriteTo(w)
	written += n
	if err != nil {
		return written, err
	}

	// Create checksummer
	crc := crc32.NewIEEE()
	mw := io.MultiWriter(w, crc)

	// Write vector data as raw bytes
	vecData := *data
	if len(vecData) > 0 {
		byteSlice := unsafe.Slice((*byte)(unsafe.Pointer(&vecData[0])), len(vecData)*4)
		n, err := mw.Write(byteSlice)
		written += int64(n)
		if err != nil {
			return written, err
		}
	}

	// Write deletion bitmap
	for _, block := range s.deleted {
		var buf [8]byte
		binary.LittleEndian.PutUint64(buf[:], block)
		n, err := mw.Write(buf[:])
		written += int64(n)
		if err != nil {
			return written, err
		}
	}

	// Pad bitmap to expected size
	expectedBitmapBlocks := (s.count + 63) / 64
	for i := uint32(len(s.deleted)); i < expectedBitmapBlocks; i++ {
		var buf [8]byte
		n, err := mw.Write(buf[:])
		written += int64(n)
		if err != nil {
			return written, err
		}
	}

	// Write final checksum
	var checksumBuf [4]byte
	binary.LittleEndian.PutUint32(checksumBuf[:], crc.Sum32())
	n2, err := w.Write(checksumBuf[:])
	written += int64(n2)

	return written, err
}

// ReadFrom reads the store from r in columnar format.
func (s *Store) ReadFrom(r io.Reader) (int64, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	var read int64

	// Read header
	var header FileHeader
	n, err := header.ReadFrom(r)
	read += n
	if err != nil {
		return read, err
	}

	s.dim = header.Dimension
	s.count = uint32(header.Count)
	s.live = uint32(header.LiveCount)

	// Create checksummer
	crc := crc32.NewIEEE()
	tr := io.TeeReader(r, crc)

	// Read vector data
	vecSize := int(header.VectorDataSize())

	// Allocate aligned memory for vectors
	vecData := mem.AllocAlignedFloat32(vecSize / 4)

	// Create a byte slice view of the aligned memory
	vecBytes := unsafe.Slice((*byte)(unsafe.Pointer(&vecData[0])), vecSize)

	n2, err := io.ReadFull(tr, vecBytes)
	read += int64(n2)
	if err != nil {
		return read, err
	}

	s.data.Store(&vecData)

	// Read deletion bitmap
	bitmapSize := header.BitmapSize()
	numBlocks := (bitmapSize + 7) / 8
	s.deleted = make([]uint64, numBlocks)
	for i := range s.deleted {
		var buf [8]byte
		n3, err := io.ReadFull(tr, buf[:])
		read += int64(n3)
		if err != nil {
			return read, err
		}
		s.deleted[i] = binary.LittleEndian.Uint64(buf[:])
	}

	// Read and verify checksum
	var checksumBuf [4]byte
	n4, err := io.ReadFull(r, checksumBuf[:])
	read += int64(n4)
	if err != nil {
		return read, err
	}

	expectedChecksum := binary.LittleEndian.Uint32(checksumBuf[:])
	if crc.Sum32() != expectedChecksum {
		return read, ErrCorrupted
	}

	return read, nil
}

// RawData returns the underlying contiguous vector data.
// This is useful for SIMD batch operations.
// The returned slice aliases internal memory; do not modify.
func (s *Store) RawData() []float32 {
	data := s.data.Load()
	if data == nil {
		return nil
	}
	return *data
}

// Ensure Store implements vectorstore.Store interface
var _ vectorstore.Store = (*Store)(nil)
