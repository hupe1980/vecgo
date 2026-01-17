package vectorstore

import (
	"runtime"
	"sync"
	_ "unsafe" // For go:linkname

	"github.com/hupe1980/vecgo/model"
)

// PrefetchHint provides prefetching capability for vector stores.
// Implementations can use this to hide memory latency by loading
// vectors into CPU cache before they're needed.
type PrefetchHint interface {
	// Prefetch hints the store to begin loading vectors for the given IDs.
	// This is a hint - implementations may ignore it.
	// The call should return immediately (non-blocking).
	Prefetch(ids []model.RowID)
}

// PrefetchBatch is a reusable batch of IDs to prefetch.
// Use GetPrefetchBatch/PutPrefetchBatch to avoid allocations.
type PrefetchBatch struct {
	IDs []model.RowID
}

// Reset clears the batch for reuse.
func (pb *PrefetchBatch) Reset() {
	pb.IDs = pb.IDs[:0]
}

// Add adds an ID to the batch.
func (pb *PrefetchBatch) Add(id model.RowID) {
	pb.IDs = append(pb.IDs, id)
}

// Len returns the number of IDs in the batch.
func (pb *PrefetchBatch) Len() int {
	return len(pb.IDs)
}

// prefetchBatchPool provides pooled PrefetchBatch instances.
var prefetchBatchPool = sync.Pool{
	New: func() any {
		return &PrefetchBatch{
			IDs: make([]model.RowID, 0, 32), // Typical HNSW neighbor count
		}
	},
}

// GetPrefetchBatch returns a pooled PrefetchBatch.
func GetPrefetchBatch() *PrefetchBatch {
	pb := prefetchBatchPool.Get().(*PrefetchBatch)
	pb.Reset()
	return pb
}

// PutPrefetchBatch returns a PrefetchBatch to the pool.
func PutPrefetchBatch(pb *PrefetchBatch) {
	if pb == nil {
		return
	}
	// Clear slice but keep capacity
	pb.IDs = pb.IDs[:0]
	prefetchBatchPool.Put(pb)
}

// Prefetch implements PrefetchHint for ColumnarStore.
// It uses CPU prefetch instructions to load vectors into L2 cache.
func (s *ColumnarStore) Prefetch(ids []model.RowID) {
	if len(ids) == 0 {
		return
	}

	dataPtr := s.data.Load()
	if dataPtr == nil {
		return
	}
	data := *dataPtr
	dim := int(s.dim)
	dataLen := len(data)

	// Prefetch each vector's memory into CPU cache
	// Each vector is dim*4 bytes (float32)
	// Cache line is typically 64 bytes
	for _, id := range ids {
		idx := int(id) * dim
		if idx >= 0 && idx+dim <= dataLen {
			// Prefetch the start of the vector
			// This will pull at least one cache line into L2
			prefetchData(&data[idx])
		}
	}
}

// prefetchData is a hint to the CPU to prefetch data.
// This is implemented differently per architecture:
// - x86/amd64: PREFETCHT0 instruction
// - ARM64: PRFM instruction
// For Go, we use a simple volatile read as a portable approximation.
//
//go:noinline
func prefetchData(ptr *float32) {
	// Volatile read - forces the memory to be accessed.
	// The compiler can't optimize this away.
	// On modern CPUs, this will trigger the hardware prefetcher.
	_ = *ptr

	// Also hint the Go runtime that we're accessing this memory.
	// This can help with GC and memory management.
	runtime.KeepAlive(ptr)
}

// PrefetchBatchFromStore prefetches a batch of vectors from a columnar store.
// This is a convenience function for HNSW neighbor prefetching.
func PrefetchBatchFromStore(store *ColumnarStore, ids []model.RowID) {
	if store == nil || len(ids) == 0 {
		return
	}
	store.Prefetch(ids)
}
