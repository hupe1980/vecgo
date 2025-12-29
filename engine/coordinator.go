// Package engine provides the coordination layer for vecgo.
//
// Vecgo routes all mutations through a Coordinator to provide atomic
// multi-subsystem semantics:
//   - (optional) durability prepare entry
//   - apply index mutation by explicit ID (deterministic)
//   - write payload + metadata stores + update meta index
//   - (optional) durability commit entry (durability boundary)
//
// Recovery ignores prepares without commits.
//
// # Architecture
//
// Coordinator is the single interface for mutation orchestration:
//   - Tx[T]: Single-shard mode (simple, low-overhead)
//   - ShardedCoordinator[T]: Multi-shard mode (parallel writes)
//
// Both implementations satisfy the Coordinator[T] interface, enabling
// transparent switching between modes.
package engine

import (
	"context"
	"fmt"
	"io"
	"iter"

	"github.com/hupe1980/vecgo/codec"
	"github.com/hupe1980/vecgo/index"
	"github.com/hupe1980/vecgo/metadata"
)

// HybridSearchOptions contains options for hybrid search (vector + metadata).
type HybridSearchOptions struct {
	// EF (Explore Factor) for HNSW search
	EF int

	// MetadataFilters are metadata conditions that must all match (AND logic)
	MetadataFilters *metadata.FilterSet

	// PreFilter applies metadata filtering before vector search (more efficient but may reduce recall)
	// PostFilter applies metadata filtering after vector search (maintains recall but less efficient)
	PreFilter bool
}

// Coordinator is the interface for all mutation orchestration in vecgo.
//
// Implementations:
//   - Tx[T]: Single-shard coordinator for simple deployments
//   - ShardedCoordinator[T]: Multi-shard coordinator for parallel write throughput
//
// All IDs returned by Insert/BatchInsert are global IDs that encode shard routing
// information when using ShardedCoordinator. Single-shard Tx returns local IDs directly.
type Coordinator[T any] interface {
	// Insert adds a vector with associated data and metadata atomically.
	// Returns the assigned ID (global ID in sharded mode).
	Insert(ctx context.Context, vector []float32, data T, meta metadata.Metadata) (uint32, error)

	// BatchInsert adds multiple vectors atomically.
	// Returns assigned IDs in the same order as input vectors.
	BatchInsert(ctx context.Context, vectors [][]float32, data []T, meta []metadata.Metadata) ([]uint32, error)

	// Update modifies an existing vector, data, and optionally metadata.
	// If meta is nil, existing metadata is preserved.
	Update(ctx context.Context, id uint32, vector []float32, data T, meta metadata.Metadata) error

	// Delete removes a vector and its associated data/metadata.
	Delete(ctx context.Context, id uint32) error

	// Get retrieves the data associated with an ID.
	// Returns the data and true if found, or zero value and false if not found.
	Get(id uint32) (T, bool)

	// GetMetadata retrieves the metadata associated with an ID.
	// Returns the metadata and true if found, or nil and false if not found.
	GetMetadata(id uint32) (metadata.Metadata, bool)

	// KNNSearch performs approximate K-nearest neighbor search.
	KNNSearch(ctx context.Context, query []float32, k int, opts *index.SearchOptions) ([]index.SearchResult, error)

	// BruteSearch performs exact brute-force search with optional filter.
	BruteSearch(ctx context.Context, query []float32, k int, filter func(id uint32) bool) ([]index.SearchResult, error)

	// HybridSearch performs a hybrid search combining vector similarity and metadata filtering.
	HybridSearch(ctx context.Context, query []float32, k int, opts *HybridSearchOptions) ([]index.SearchResult, error)

	// KNNSearchStream returns an iterator over K-nearest neighbor search results.
	KNNSearchStream(ctx context.Context, query []float32, k int, opts *index.SearchOptions) iter.Seq2[index.SearchResult, error]

	// EnableProductQuantization enables Product Quantization (PQ) on the underlying index(es).
	EnableProductQuantization(cfg index.ProductQuantizationConfig) error

	// DisableProductQuantization disables Product Quantization (PQ) on the underlying index(es).
	DisableProductQuantization()

	// SaveToWriter saves the database to an io.Writer.
	// Note: For sharded coordinators, this might not be supported or might save a combined format.
	SaveToWriter(w io.Writer) error

	// SaveToFile saves the database to a file (or directory for sharded).
	SaveToFile(path string) error

	// RecoverFromWAL replays the write-ahead log to recover from a crash.
	RecoverFromWAL(ctx context.Context) error

	// Stats returns statistics about the underlying index(es).
	Stats() index.Stats

	// Close releases all resources held by the coordinator (indexes, stores, WALs).
	Close() error
}

// Compile-time interface checks
var (
	_ Coordinator[any] = (*Tx[any])(nil)
	_ Coordinator[any] = (*ShardedCoordinator[any])(nil)
)

// New constructs a single-shard Coordinator (Tx).
//
// If d is nil, NoopDurability is used (same atomicity semantics, no persistence).
// The index MUST implement index.TransactionalIndex.
func New[T any](idx index.Index, dataStore Store[T], metaStore *metadata.UnifiedIndex, d Durability, c codec.Codec) (Coordinator[T], error) {
	if idx == nil {
		return nil, fmt.Errorf("coordinator: index is nil")
	}
	if dataStore == nil {
		return nil, fmt.Errorf("coordinator: data store is nil")
	}
	if metaStore == nil {
		return nil, fmt.Errorf("coordinator: metadata store is nil")
	}
	if c == nil {
		c = codec.Default
	}
	if d == nil {
		d = NoopDurability{}
	}

	// Require TransactionalIndex
	txIdx, ok := idx.(index.TransactionalIndex)
	if !ok {
		return nil, fmt.Errorf("coordinator: index type %T must implement index.TransactionalIndex", idx)
	}

	return &Tx[T]{
		txIndex:    txIdx,
		dataStore:  dataStore,
		metaStore:  metaStore,
		durability: d,
		codec:      c,
	}, nil
}
