package segment

import (
	"sync"

	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/model"
)

// DefaultFetchArenaCap is the default capacity for FetchArena slices.
// Sized for typical batch sizes (64-128 rows).
const DefaultFetchArenaCap = 128

// FetchArena provides pre-allocated scratch space for Fetch operations.
// It eliminates per-batch allocations by reusing slices across queries.
//
// FetchArena is NOT thread-safe. It should be owned by a single goroutine
// (typically acquired from pool per query) and Reset() between uses.
//
// Memory footprint (default capacity 128):
//   - IDs: 128 * 8 = 1 KB
//   - Vectors: 128 * 8 = 1 KB (slice headers, actual data in VectorsBacking)
//   - VectorsBacking: 128 * 1536 * 4 = 768 KB (for 1536-dim vectors)
//   - Metadatas: 128 * 8 = 1 KB (slice headers)
//   - MetadataPool: 128 * ~64 = 8 KB (maps with avg 4 keys)
//   - Payloads: 128 * 8 = 1 KB (slice headers)
//   - PayloadBacking: 128 * 256 = 32 KB (avg 256 bytes per payload)
//   - Total: ~812 KB per arena (acceptable for pooled resource)
//
// Benchmark results (BatchSize=100, dim=128):
//   - With FetchArena: 0 allocs/op, 2113 ns/op
//   - Without (baseline): 204 allocs/op, 15855 ns/op, 137KB
//   - Improvement: 7.5x faster, 100% allocation reduction
type FetchArena struct {
	// Slice headers for RecordBatch
	IDs       []model.ID
	Vectors   [][]float32
	Metadatas []metadata.Document
	Payloads  [][]byte

	// Backing arrays to avoid per-row allocations
	VectorsBacking []float32 // Flat array: row i's vector is VectorsBacking[i*dim:(i+1)*dim]
	PayloadBacking []byte    // Pooled buffer for payload reads
	PayloadOffsets []int     // Offsets into PayloadBacking for each row

	// Pooled metadata documents for reuse
	metadataPool []*metadata.Metadata

	// Pooled batch to avoid allocation in BuildRecordBatch
	batch SimpleRecordBatch

	// Dimension (set on first use, validated on subsequent uses)
	dim int

	// Current batch size (for Reset validation)
	batchSize int
}

// NewFetchArena creates a FetchArena with default capacity.
func NewFetchArena() *FetchArena {
	return NewFetchArenaWithCapacity(DefaultFetchArenaCap, 0)
}

// NewFetchArenaWithCapacity creates a FetchArena with the specified capacity.
// dim can be 0 if unknown; it will be set on first use.
func NewFetchArenaWithCapacity(capacity, dim int) *FetchArena {
	a := &FetchArena{
		IDs:            make([]model.ID, 0, capacity),
		Vectors:        make([][]float32, 0, capacity),
		Metadatas:      make([]metadata.Document, 0, capacity),
		Payloads:       make([][]byte, 0, capacity),
		PayloadOffsets: make([]int, 0, capacity+1),
		metadataPool:   make([]*metadata.Metadata, 0, capacity),
		dim:            dim,
	}

	// Pre-allocate backing arrays if dimension is known
	if dim > 0 {
		a.VectorsBacking = make([]float32, 0, capacity*dim)
	}

	// Pre-allocate payload backing (256 bytes avg per payload)
	a.PayloadBacking = make([]byte, 0, capacity*256)

	// Pre-allocate metadata maps
	for i := 0; i < capacity; i++ {
		m := make(metadata.Metadata, 4) // Avg 4 keys per document
		a.metadataPool = append(a.metadataPool, &m)
	}

	return a
}

// Reset prepares the arena for a new batch of the given size.
// Slices are truncated to zero length but capacity is preserved.
func (a *FetchArena) Reset(batchSize int) {
	a.batchSize = batchSize

	// Reset slice lengths (preserve capacity)
	a.IDs = a.IDs[:0]
	a.Vectors = a.Vectors[:0]
	a.Metadatas = a.Metadatas[:0]
	a.Payloads = a.Payloads[:0]
	a.PayloadOffsets = a.PayloadOffsets[:0]
	a.PayloadBacking = a.PayloadBacking[:0]
	a.VectorsBacking = a.VectorsBacking[:0]

	// Clear metadata maps for reuse (don't delete keys - just reset)
	for i := 0; i < len(a.metadataPool) && i < batchSize; i++ {
		clear(*a.metadataPool[i])
	}
}

// EnsureCapacity grows the arena if needed to handle batchSize rows.
// Returns true if reallocation occurred.
func (a *FetchArena) EnsureCapacity(batchSize, dim int) bool {
	reallocated := false

	// Update dimension if not set
	if a.dim == 0 && dim > 0 {
		a.dim = dim
	}

	// Grow ID slice
	if cap(a.IDs) < batchSize {
		a.IDs = make([]model.ID, 0, batchSize)
		reallocated = true
	}

	// Grow Vectors slice headers
	if cap(a.Vectors) < batchSize {
		a.Vectors = make([][]float32, 0, batchSize)
		reallocated = true
	}

	// Grow VectorsBacking
	if dim > 0 && cap(a.VectorsBacking) < batchSize*dim {
		a.VectorsBacking = make([]float32, 0, batchSize*dim)
		reallocated = true
	}

	// Grow Metadatas slice
	if cap(a.Metadatas) < batchSize {
		a.Metadatas = make([]metadata.Document, 0, batchSize)
		reallocated = true
	}

	// Grow metadata pool
	for len(a.metadataPool) < batchSize {
		m := make(metadata.Metadata, 4)
		a.metadataPool = append(a.metadataPool, &m)
		reallocated = true
	}

	// Grow Payloads slice
	if cap(a.Payloads) < batchSize {
		a.Payloads = make([][]byte, 0, batchSize)
		a.PayloadOffsets = make([]int, 0, batchSize+1)
		reallocated = true
	}

	return reallocated
}

// AcquireMetadata returns a pooled Metadata map for row i.
// The map is cleared but retains its capacity.
// Caller must NOT retain the returned map beyond the current batch.
func (a *FetchArena) AcquireMetadata(i int) *metadata.Metadata {
	if i < len(a.metadataPool) {
		return a.metadataPool[i]
	}
	// Fallback: create new (should be rare after warmup)
	m := make(metadata.Metadata, 4)
	return &m
}

// AllocVectorSlice returns a slice from VectorsBacking for row i.
// The slice points into the backing array (zero-copy within batch).
func (a *FetchArena) AllocVectorSlice(i int) []float32 {
	if a.dim == 0 {
		return nil
	}
	start := i * a.dim
	end := start + a.dim

	// Ensure backing has capacity
	if cap(a.VectorsBacking) < end {
		// Grow backing array
		newCap := end * 2
		newBacking := make([]float32, len(a.VectorsBacking), newCap)
		copy(newBacking, a.VectorsBacking)
		a.VectorsBacking = newBacking
	}

	// Extend length if needed
	if len(a.VectorsBacking) < end {
		a.VectorsBacking = a.VectorsBacking[:end]
	}

	return a.VectorsBacking[start:end]
}

// AllocPayloadSlice returns a slice from PayloadBacking for a payload of given size.
// The slice points into the backing array (zero-copy within batch).
func (a *FetchArena) AllocPayloadSlice(size int) []byte {
	if size == 0 {
		return nil
	}

	start := len(a.PayloadBacking)
	end := start + size

	// Ensure backing has capacity
	if cap(a.PayloadBacking) < end {
		// Grow backing array (at least double)
		newCap := max(end*2, cap(a.PayloadBacking)*2)
		newBacking := make([]byte, start, newCap)
		copy(newBacking, a.PayloadBacking)
		a.PayloadBacking = newBacking
	}

	// Extend length
	a.PayloadBacking = a.PayloadBacking[:end]

	return a.PayloadBacking[start:end]
}

// SetDimension sets the vector dimension (must be called before AllocVectorSlice).
func (a *FetchArena) SetDimension(dim int) {
	a.dim = dim
}

// BuildRecordBatch configures and returns the arena's pooled SimpleRecordBatch.
// The returned batch shares memory with the arena - it's only valid until Reset().
//
// Zero allocations: the batch struct itself is pooled within the arena.
func (a *FetchArena) BuildRecordBatch(fetchVectors, fetchMetadata, fetchPayloads bool) *SimpleRecordBatch {
	a.batch.IDs = a.IDs
	a.batch.Vectors = nil
	a.batch.Metadatas = nil
	a.batch.Payloads = nil

	if fetchVectors {
		a.batch.Vectors = a.Vectors
	}
	if fetchMetadata {
		a.batch.Metadatas = a.Metadatas
	}
	if fetchPayloads {
		a.batch.Payloads = a.Payloads
	}
	return &a.batch
}

// fetchArenaPool provides pooled FetchArenas.
var fetchArenaPool = sync.Pool{
	New: func() any {
		return NewFetchArena()
	},
}

// GetFetchArena returns a FetchArena from the pool.
func GetFetchArena() *FetchArena {
	return fetchArenaPool.Get().(*FetchArena)
}

// PutFetchArena returns a FetchArena to the pool.
func PutFetchArena(a *FetchArena) {
	a.Reset(0)
	fetchArenaPool.Put(a)
}
