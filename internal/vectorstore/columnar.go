package vectorstore

import (
	"context"
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"io"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/hupe1980/vecgo/distance"
	"github.com/hupe1980/vecgo/internal/arena"
	"github.com/hupe1980/vecgo/internal/conv"
	"github.com/hupe1980/vecgo/internal/mem"
	"github.com/hupe1980/vecgo/model"
)

// VectorSnapshot provides a read-only view of the vector data at a specific point in time.
// It avoids atomic loads and synchronization overhead during batched operations (like Search).
type VectorSnapshot struct {
	data []float32
	dim  int
}

// ComputeDistance computes the distance between a vector in the snapshot and a query.
// It performs no bounds checking for maximum performance (assumes valid IDs from HNSW graph).
// Use with caution.
func (vs *VectorSnapshot) ComputeDistance(id model.RowID, q []float32, metric distance.Metric) (float32, bool) {
	idx := int(id) * vs.dim
	// Manual bounds check for safety in production, can be removed with build tags if desired
	if idx < 0 || idx+vs.dim > len(vs.data) {
		return 3.402823466e+38, false // MaxFloat32
	}

	// Fast path based on metric
	switch metric {
	case distance.MetricL2:
		return distance.SquaredL2(vs.data[idx:idx+vs.dim], q), true
	case distance.MetricDot:
		return -distance.Dot(vs.data[idx:idx+vs.dim], q), true
	case distance.MetricCosine:
		return 0.5 * distance.SquaredL2(vs.data[idx:idx+vs.dim], q), true
	}

	// Fallback
	distFn, _ := distance.Provider(metric)
	return distFn(vs.data[idx:idx+vs.dim], q), true
}

// Snapshot returns a thread-safe, read-only snapshot of the current vector data.
func (s *ColumnarStore) Snapshot() VectorSnapshot {
	data := s.data.Load()
	if data == nil {
		return VectorSnapshot{dim: int(s.dim)}
	}
	return VectorSnapshot{
		data: *data,
		dim:  int(s.dim),
	}
}

// OptimizedDistanceComputer returns a function that computes distance without interface overhead.
// The returned function is safe for concurrent use.
func (s *ColumnarStore) OptimizedDistanceComputer(metric distance.Metric) (func(model.RowID, []float32) float32, bool) {
	dim := int(s.dim)

	// fast path for L2
	if metric == distance.MetricL2 {
		return func(id model.RowID, q []float32) float32 {
			dataPtr := s.data.Load()
			if dataPtr == nil {
				return 3.402823466e+38 // math.MaxFloat32
			}
			data := *dataPtr
			idx := int(id) * dim
			if idx < 0 || idx+dim > len(data) {
				return 3.402823466e+38 // math.MaxFloat32
			}
			return distance.SquaredL2(data[idx:idx+dim], q)
		}, true
	}

	// fast path for Dot/Cosine
	if metric == distance.MetricDot || metric == distance.MetricCosine {
		return func(id model.RowID, q []float32) float32 {
			dataPtr := s.data.Load()
			if dataPtr == nil {
				return 3.402823466e+38 // math.MaxFloat32
			}
			data := *dataPtr
			idx := int(id) * dim
			if idx < 0 || idx+dim > len(data) {
				return 3.402823466e+38 // math.MaxFloat32
			}
			return distance.Dot(data[idx:idx+dim], q)
		}, true
	}

	distFn, err := distance.Provider(metric)
	if err != nil {
		return nil, false
	}

	return func(id model.RowID, q []float32) float32 {
		dataPtr := s.data.Load()
		if dataPtr == nil {
			return 3.402823466e+38 // math.MaxFloat32
		}
		data := *dataPtr
		idx := int(id) * dim
		if idx < 0 || idx+dim > len(data) {
			return 3.402823466e+38 // math.MaxFloat32
		}
		return distFn(data[idx:idx+dim], q)
	}, true
}

// ColumnarStore is a high-performance columnar vector store.
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
type ColumnarStore struct {
	dim uint32

	// Atomic for lock-free read access
	data    atomic.Pointer[[]float32] // Contiguous vector storage: vectors[id] = data[id*dim : (id+1)*dim]
	deleted atomic.Pointer[[]uint64]  // Bitmap: bit i = 1 means vector i is deleted

	mu sync.RWMutex
	// deleted  []uint64 // Removed in favor of atomic.Pointer
	versions []uint64 // Version numbers for each vector (optional)
	count    uint64   // Total vectors (including deleted)
	live     uint64   // Live (non-deleted) vectors

	acquirer arena.MemoryAcquirer
}

// Close releases any resources held by the store.
func (s *ColumnarStore) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.acquirer != nil {
		data := s.data.Load()
		if data != nil {
			bytes := int64(cap(*data) * 4)
			s.acquirer.ReleaseMemory(bytes)
		}
		s.acquirer = nil
	}
	return nil
}

// New creates a new in-memory columnar store with the given dimension.
func New(dim int, acquirer arena.MemoryAcquirer) (*ColumnarStore, error) {
	if dim <= 0 {
		dim = 1
	}
	// Pre-allocate for ~1K vectors using aligned memory
	initialCap := 1024 * dim

	// Acquire memory for initial chunk
	if acquirer != nil {
		// Use a timeout to prevent deadlocks if memory is full
		ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
		defer cancel()
		if err := acquirer.AcquireMemory(ctx, int64(initialCap*4)); err != nil {
			return nil, err
		}
	}

	data := mem.AllocAlignedFloat32(initialCap)
	data = data[:0]

	dimU32, err := conv.IntToUint32(dim)
	if err != nil {
		return nil, err
	}
	s := &ColumnarStore{
		dim:      dimU32,
		versions: make([]uint64, 0, 1024),
		acquirer: acquirer,
	}
	s.data.Store(&data)

	deleted := make([]uint64, 0)
	s.deleted.Store(&deleted)

	return s, nil
}

// Dimension returns the vector dimensionality.
func (s *ColumnarStore) Dimension() int {
	return int(s.dim)
}

// Count returns the total number of vectors (including deleted).
func (s *ColumnarStore) Count() uint64 {
	s.mu.RLock()
	c := s.count
	s.mu.RUnlock()
	return c
}

// Size returns the estimated memory usage in bytes.
func (s *ColumnarStore) Size() int64 {
	data := s.data.Load()
	deleted := s.deleted.Load()

	size := int64(0)
	if data != nil {
		size += int64(cap(*data)) * 4
	}
	if deleted != nil {
		size += int64(cap(*deleted)) * 8
	}
	// Add versions size
	s.mu.RLock()
	size += int64(cap(s.versions)) * 8
	s.mu.RUnlock()

	return size
}

// LiveCount returns the number of non-deleted vectors.
func (s *ColumnarStore) LiveCount() uint64 {
	s.mu.RLock()
	l := s.live
	s.mu.RUnlock()
	return l
}

// GetVector returns the vector at the given ID.
// Returns nil, false if the ID is out of bounds or the vector is deleted.
// The returned slice may alias internal memory; do not modify.
func (s *ColumnarStore) GetVector(id model.RowID) ([]float32, bool) {
	data := s.data.Load()
	if data == nil {
		return nil, false
	}

	dim := int(s.dim)
	idInt, err := conv.Uint32ToInt(uint32(id))
	if err != nil {
		return nil, false
	}
	start := idInt * dim
	end := start + dim

	if end > len(*data) {
		return nil, false
	}

	// Check deletion bitmap
	// Lock-free check
	if s.isDeleted(id) {
		return nil, false
	}

	return (*data)[start:end:end], true
}

// ComputeDistance computes the distance between the query and the vector at ID.
// This avoids allocating a slice header for the vector and allows internal optimization.
func (s *ColumnarStore) ComputeDistance(id model.RowID, query []float32, metric distance.Metric) (float32, bool) {
	data := s.data.Load()
	if data == nil {
		return 0, false
	}

	dim := int(s.dim)
	idInt := int(id)
	start := idInt * dim
	end := start + dim

	if end > len(*data) {
		return 0, false
	}

	// Check deletion bitmap
	if s.isDeleted(id) {
		return 0, false
	}

	vec := (*data)[start:end]

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

// RawData returns the underlying contiguous float32 slice and the dimension.
// This allows for batch operations (SIMD) on the entire dataset.
// The returned slice aliases internal memory; do not modify.
func (s *ColumnarStore) RawData() ([]float32, int) {
	data := s.data.Load()
	if data == nil {
		return nil, int(s.dim)
	}
	return *data, int(s.dim)
}

func (s *ColumnarStore) isDeleted(id model.RowID) bool {
	deletedPtr := s.deleted.Load()
	if deletedPtr == nil {
		return false
	}
	idU32 := uint32(id)
	bitmapIdx := idU32 / 64
	bitmapIdxInt := int(bitmapIdx)
	if bitmapIdxInt < len(*deletedPtr) {
		bitIdx := idU32 % 64
		return atomic.LoadUint64(&(*deletedPtr)[bitmapIdxInt])&(1<<bitIdx) != 0
	}
	return false
}

// GetVectorUnsafe returns the vector without checking deletion status.
// Use with caution - mainly for internal iteration during compaction.
func (s *ColumnarStore) GetVectorUnsafe(id model.RowID) ([]float32, bool) {
	data := s.data.Load()
	if data == nil {
		return nil, false
	}

	dim := int(s.dim)
	idInt, err := conv.Uint32ToInt(uint32(id))
	if err != nil {
		return nil, false
	}
	start := idInt * dim
	end := start + dim

	if end > len(*data) {
		return nil, false
	}

	return (*data)[start:end:end], true
}

// SetVector sets (or replaces) the vector at the given ID.
// This is part of the vectorstore.VectorStore interface.
func (s *ColumnarStore) SetVector(ctx context.Context, id model.RowID, v []float32) error {
	if len(v) != int(s.dim) {
		return ErrWrongDimension
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	data := s.data.Load()
	dim := int(s.dim)

	// If ID is beyond current length, we need to extend the slice length.
	// IMPORTANT: do not publish the extended slice (via s.data.Store) until
	// after the new vector data has been fully written. Otherwise, concurrent
	// readers can observe partially-written vectors.
	idInt, err := conv.Uint32ToInt(uint32(id))
	if err != nil {
		return err
	}
	requiredLen := (idInt + 1) * dim
	currentData := *data

	// Copy-on-write for growth beyond current length.
	if requiredLen > len(currentData) {
		if err := s.growAndSet(ctx, id, requiredLen, currentData, v); err != nil {
			return err
		}
	} else {
		// In-place update of an already-published vector.
		// Writers are externally synchronized, but concurrent readers may exist.
		// This is safe for correctness if callers do not rely on atomic point updates;
		// for strict snapshot semantics, updates must use copy-on-write at a higher level.
		start := int(id) * dim
		copy(currentData[start:start+dim], v)
	}

	// Clear deletion bit if it was set
	if s.isDeletedLocked(id) {
		s.setDeletedLocked(id, false)
		s.live++
	} else if uint64(id) == s.count-1 {
		// New vector
		s.live++
	}
	return nil
}

func (s *ColumnarStore) growAndSet(ctx context.Context, id model.RowID, requiredLen int, currentData, v []float32) error {
	dim := int(s.dim)
	idInt := int(id)
	if requiredLen <= cap(currentData) {
		// Grow within capacity (same backing array). Safe as long as we only
		// write beyond the previously-published length, then publish length.
		grown := currentData[:requiredLen]
		start := idInt * dim
		copy(grown[start:start+dim], v)
		// Publish new length after data write.
		published := grown
		s.data.Store(&published)
	} else {
		// Allocate new backing array, populate fully, then publish.
		newCap := requiredLen * 2
		if len(currentData)*2 > newCap {
			newCap = len(currentData) * 2
		}

		// Check memory limits
		if s.acquirer != nil {
			delta := int64(newCap*4 - cap(currentData)*4)
			if delta > 0 {
				if err := s.acquirer.AcquireMemory(ctx, delta); err != nil {
					return err
				}
			}
		}

		newData := mem.AllocAlignedFloat32(newCap)
		newData = newData[:requiredLen]
		copy(newData, currentData)
		start := idInt * dim
		copy(newData[start:start+dim], v)
		s.data.Store(&newData)
	}

	// Extend deletion bitmap if needed
	requiredBitmapLen := (idInt + 63) / 64
	currentDeleted := *s.deleted.Load()
	if len(currentDeleted) < requiredBitmapLen+1 {
		newDeleted := make([]uint64, requiredBitmapLen+1)
		copy(newDeleted, currentDeleted)
		s.deleted.Store(&newDeleted)
	}

	// Update count if we're adding a new vector
	if uint64(id) >= s.count {
		s.count = uint64(id) + 1
	}
	return nil
}

// Append adds a new vector and returns its ID.
func (s *ColumnarStore) Append(ctx context.Context, v []float32) (model.RowID, error) {
	if len(v) != int(s.dim) {
		return 0, ErrWrongDimension
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	idU32, err := conv.Uint64ToUint32(s.count)
	if err != nil {
		return 0, err
	}
	id := model.RowID(idU32)
	dim := int(s.dim)

	data := s.data.Load()
	currentData := *data

	// Extend data slice.
	// IMPORTANT: publish updated slice only after writing the new vector.
	idInt, err := conv.Uint32ToInt(uint32(id))
	if err != nil {
		return 0, err
	}
	requiredLen := (idInt + 1) * dim
	if requiredLen > cap(currentData) {
		newCap := max(requiredLen*2, len(currentData)*2)

		// Check memory limits
		if s.acquirer != nil {
			delta := int64(newCap*4 - cap(currentData)*4)
			if delta > 0 {
				if err := s.acquirer.AcquireMemory(ctx, delta); err != nil {
					return 0, err
				}
			}
		}

		newData := mem.AllocAlignedFloat32(newCap)
		newData = newData[:requiredLen]
		copy(newData, currentData)
		start := idInt * dim
		copy(newData[start:start+dim], v)
		s.data.Store(&newData)
	} else {
		grown := currentData[:requiredLen]
		start := idInt * dim
		copy(grown[start:start+dim], v)
		published := grown
		s.data.Store(&published)
	}

	// Extend deletion bitmap if needed
	bitmapIdx := uint32(id) / 64
	currentDeleted := *s.deleted.Load()
	if int(bitmapIdx) >= len(currentDeleted) {
		newDeleted := make([]uint64, bitmapIdx+1)
		copy(newDeleted, currentDeleted)
		s.deleted.Store(&newDeleted)
	}

	s.count++
	s.live++

	return id, nil
}

// DeleteVector marks a vector as deleted.
func (s *ColumnarStore) DeleteVector(id model.RowID) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if uint64(id) >= s.count {
		return ErrOutOfBounds
	}

	if !s.isDeletedLocked(id) {
		s.setDeletedLocked(id, true)
		s.live--
	}

	return nil
}

// IsDeleted returns true if the vector at id is deleted.
func (s *ColumnarStore) IsDeleted(id model.RowID) bool {
	// Lock-free check
	deletedPtr := s.deleted.Load()
	if deletedPtr == nil {
		return false
	}
	idU32 := uint32(id)
	bitmapIdx := idU32 / 64
	if int(bitmapIdx) >= len(*deletedPtr) {
		return false
	}
	bitIdx := idU32 % 64
	return atomic.LoadUint64(&(*deletedPtr)[int(bitmapIdx)])&(1<<bitIdx) != 0
}

func (s *ColumnarStore) isDeletedLocked(id model.RowID) bool {
	return s.IsDeleted(id)
}

func (s *ColumnarStore) setDeletedLocked(id model.RowID, deleted bool) {
	idU32 := uint32(id)
	bitmapIdx := idU32 / 64
	currentDeleted := *s.deleted.Load()

	if int(bitmapIdx) >= len(currentDeleted) {
		newDeleted := make([]uint64, bitmapIdx+1)
		copy(newDeleted, currentDeleted)
		s.deleted.Store(&newDeleted)
		currentDeleted = newDeleted
	}

	bitIdx := idU32 % 64
	addr := &currentDeleted[int(bitmapIdx)]
	val := atomic.LoadUint64(addr)
	if deleted {
		val |= 1 << bitIdx
	} else {
		val &^= 1 << bitIdx
	}
	atomic.StoreUint64(addr, val)
}

// Compact removes deleted vectors and defragments the store.
// Returns a map from old IDs to new IDs for live vectors.
// Deleted vectors are not included in the map.
func (s *ColumnarStore) Compact() (map[model.RowID]model.RowID, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.live == s.count {
		// Nothing to compact
		return nil, nil
	}

	dim := int(s.dim)
	oldData := *s.data.Load()
	newData := mem.AllocAlignedFloat32(int(s.live) * dim)
	newData = newData[:0]
	idMap := make(map[model.RowID]model.RowID, s.live)

	var newID model.RowID
	for oldID := uint64(0); oldID < s.count; oldID++ {
		oldIDU32, err := conv.Uint64ToUint32(oldID)
		if err != nil {
			return nil, fmt.Errorf("columnar: compact failed: %w", err)
		}
		localOldID := model.RowID(oldIDU32)
		if s.isDeletedLocked(localOldID) {
			continue
		}

		start := int(oldID) * dim
		end := start + dim
		newData = append(newData, oldData[start:end]...)
		idMap[localOldID] = newID
		newID++
	}

	s.data.Store(&newData)
	s.count = uint64(newID)
	s.live = uint64(newID)
	newDeleted := make([]uint64, (newID+63)/64)
	s.deleted.Store(&newDeleted)

	return idMap, nil
}

// Iterate calls fn for each live vector. Return false from fn to stop iteration.
func (s *ColumnarStore) Iterate(fn func(id model.RowID, vec []float32) bool) {
	s.mu.RLock()
	count := s.count
	s.mu.RUnlock()

	data := s.data.Load()
	if data == nil {
		return
	}

	dim := int(s.dim)

	for id := uint64(0); id < count; id++ {
		idU32, err := conv.Uint64ToUint32(id)
		if err != nil {
			break
		}
		rowID := model.RowID(idU32)
		s.mu.RLock()
		deleted := s.isDeletedLocked(rowID)
		s.mu.RUnlock()

		if deleted {
			continue
		}

		start := int(id) * dim
		end := start + dim
		if end > len(*data) {
			break
		}

		if !fn(rowID, (*data)[start:end:end]) {
			break
		}
	}
}

// WriteTo writes the store to w in columnar format.
func (s *ColumnarStore) WriteTo(w io.Writer) (int64, error) {
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
		Count:      s.count,
		LiveCount:  s.live,
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
	currentDeleted := *s.deleted.Load()
	for _, block := range currentDeleted {
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
	for i := uint64(len(currentDeleted)); i < expectedBitmapBlocks; i++ {
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
func (s *ColumnarStore) ReadFrom(r io.Reader) (int64, error) {
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
	s.count = header.Count
	s.live = header.LiveCount

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
	deleted := make([]uint64, numBlocks)
	for i := range deleted {
		var buf [8]byte
		n3, err := io.ReadFull(tr, buf[:])
		read += int64(n3)
		if err != nil {
			return read, err
		}
		deleted[i] = binary.LittleEndian.Uint64(buf[:])
	}
	s.deleted.Store(&deleted)

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

var _ VectorStore = (*ColumnarStore)(nil)
