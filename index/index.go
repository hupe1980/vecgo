// Package index provides interfaces and types for vector search indexes.
package index

import (
	"context"
	"errors"
	"fmt"
	"io"
	"iter"
	"sort"
	"strings"

	"github.com/hupe1980/vecgo/core"
	"github.com/hupe1980/vecgo/distance"
	"github.com/hupe1980/vecgo/searcher"
)

// Common sentinel errors for index operations
var (
	// ErrEmptyVector is returned when an empty vector is provided
	ErrEmptyVector = errors.New("vector cannot be empty")

	// ErrInvalidK is returned when k is not positive
	ErrInvalidK = errors.New("k must be positive")

	// ErrEntryPointDeleted is returned when the entry point has been deleted
	ErrEntryPointDeleted = errors.New("entry point has been deleted")

	// ErrInsufficientVectors is returned when there aren't enough vectors for an operation
	ErrInsufficientVectors = errors.New("not enough vectors for operation")
)

// ErrInvalidDimension is a typed error for invalid configured dimensions.
// It is returned when an index/DB is constructed with a non-positive dimension.
type ErrInvalidDimension struct {
	Dimension int
}

func (e *ErrInvalidDimension) Error() string {
	return fmt.Sprintf("invalid dimension: %d", e.Dimension)
}

// ErrInvalidDistanceType is a typed error for unsupported distance types.
type ErrInvalidDistanceType struct {
	DistanceType DistanceType
}

func (e *ErrInvalidDistanceType) Error() string {
	return fmt.Sprintf("invalid distance type: %s", e.DistanceType.String())
}

// ErrDimensionMismatch is a typed error for dimension mismatch
type ErrDimensionMismatch struct {
	Expected int // Expected dimensions
	Actual   int // Actual dimensions
}

// Error returns the error message for dimension mismatch
func (e *ErrDimensionMismatch) Error() string {
	return fmt.Sprintf("dimension mismatch: expected %d, got %d", e.Expected, e.Actual)
}

// ErrNodeNotFound is a typed error for node lookup failures
type ErrNodeNotFound struct {
	ID core.LocalID // Node ID that was not found
}

// Error returns the error message for node not found
func (e *ErrNodeNotFound) Error() string {
	return fmt.Sprintf("node %d not found", e.ID)
}

// ErrNodeDeleted is a typed error for operations on deleted nodes
type ErrNodeDeleted struct {
	ID core.LocalID // Node ID that was deleted
}

// Error returns the error message for deleted node
func (e *ErrNodeDeleted) Error() string {
	return fmt.Sprintf("node %d has been deleted", e.ID)
}

// DistanceFunc represents a function for calculating the distance between two vectors.
// Dimensions are assumed to be validated before calling this function.
type DistanceFunc func(v1, v2 []float32) float32

// DistanceType represents the type of distance function used for calculating distances between vectors.
type DistanceType int

// Constants representing different types of distance functions.
//
// Important: Vecgo indexes assume "lower is better" for the value returned by NewDistanceFunc.
// Some common similarities (cosine, dot product) are therefore represented as derived distances.
const (
	DistanceTypeSquaredL2 DistanceType = iota
	// DistanceTypeCosine is implemented using L2-normalized vectors (like most vector stores).
	// The resulting distance is equivalent to (1 - cosineSimilarity) for unit vectors.
	DistanceTypeCosine
	// DistanceTypeDotProduct is treated as a similarity (higher is better) and is exposed as a distance
	// by negating the dot product.
	DistanceTypeDotProduct
)

// String returns a string representation of the DistanceType.
func (t DistanceType) String() string {
	switch t {
	case DistanceTypeSquaredL2:
		return "SquaredL2"
	case DistanceTypeCosine:
		return "CosineDistance"
	case DistanceTypeDotProduct:
		return "DotProduct"
	default:
		return "Unknown"
	}
}

// NewDistanceFunc returns a distance function based on the specified distance type.
// The returned function assumes vectors have matching dimensions (validated at insert time).
func NewDistanceFunc(distanceType DistanceType) DistanceFunc {
	switch distanceType {
	case DistanceTypeCosine:
		// Cosine is typically implemented by L2-normalizing vectors and then using
		// an L2-derived distance. For unit vectors:
		//   1 - cos(a,b) = ||a-b||^2 / 2
		// Returning cosine *distance* (lower is better) keeps index ordering consistent.
		return func(v1, v2 []float32) float32 {
			return 0.5 * distance.SquaredL2(v1, v2)
		}
	case DistanceTypeDotProduct:
		// Convert dot product similarity (higher is better) into a distance (lower is better).
		return func(v1, v2 []float32) float32 {
			return -distance.Dot(v1, v2)
		}
	default:
		return distance.SquaredL2
	}
}

// ValidateBasicOptions validates common index options (dimension and distance type).
// This helper eliminates code duplication across Flat and HNSW constructors.
// Returns an error if validation fails, nil otherwise.
func ValidateBasicOptions(dimension int, distanceType DistanceType) error {
	if dimension <= 0 {
		return &ErrInvalidDimension{Dimension: dimension}
	}
	switch distanceType {
	case DistanceTypeSquaredL2, DistanceTypeCosine, DistanceTypeDotProduct:
		return nil
	default:
		return &ErrInvalidDistanceType{DistanceType: distanceType}
	}
}

// SearchResult represents a search result.
type SearchResult struct {
	// ID is the identifier of the search result.
	ID uint32

	// Distance is the distance between the query vector and the result vector.
	Distance float32
}

// BatchInsertResult represents the result of a batch insert operation
type BatchInsertResult struct {
	IDs    []core.LocalID // IDs of successfully inserted vectors
	Errors []error        // Errors for failed insertions (nil for successful)
}

// SearchOptions contains parameters for KNN search that are index-specific.
// Indexes can use or ignore options as appropriate.
type SearchOptions struct {
	// EFSearch is the exploration factor for HNSW search (HNSW-specific).
	// Ignored by other index types.
	EFSearch int

	// Filter function to exclude results during search.
	Filter func(id core.LocalID) bool
}

// LevelStats contains per-level index statistics.
// Only some index types populate this.
type LevelStats struct {
	Level          int
	Nodes          int
	Connections    int
	AvgConnections int
}

// Stats is a structured, non-printing representation of index statistics.
type Stats struct {
	Options     map[string]string
	Parameters  map[string]string
	Storage     map[string]string
	Concurrency map[string]string
	Levels      []LevelStats
}

func (s Stats) String() string {
	var b strings.Builder
	writeSection := func(title string, m map[string]string) {
		if len(m) == 0 {
			return
		}
		b.WriteString(title)
		b.WriteString(":\n")
		keys := make([]string, 0, len(m))
		for k := range m {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			b.WriteString("\t")
			b.WriteString(k)
			b.WriteString(" = ")
			b.WriteString(m[k])
			b.WriteString("\n")
		}
	}

	writeSection("Options", s.Options)
	writeSection("Parameters", s.Parameters)
	writeSection("Storage", s.Storage)
	writeSection("Concurrency", s.Concurrency)

	if len(s.Levels) > 0 {
		b.WriteString("Node Levels:\n")
		for _, lvl := range s.Levels {
			fmt.Fprintf(&b, "\tLevel %d:\n", lvl.Level)
			fmt.Fprintf(&b, "\t\tNumber of nodes: %d\n", lvl.Nodes)
			fmt.Fprintf(&b, "\t\tNumber of connections: %d\n", lvl.Connections)
			fmt.Fprintf(&b, "\t\tAverage connections per node: %d\n", lvl.AvgConnections)
		}
		fmt.Fprintf(&b, "\nTotal number of node levels = %d\n", len(s.Levels))
	}

	return b.String()
}

// WriteTo writes a human-readable stats representation.
func (s Stats) WriteTo(w io.Writer) (int64, error) {
	n, err := io.WriteString(w, s.String())
	return int64(n), err
}

// Index represents an index for vector search.
type Index interface {
	// Insert adds a vector to the index
	Insert(ctx context.Context, v []float32) (core.LocalID, error)

	// BatchInsert adds multiple vectors to the index in a single operation.
	// Returns IDs and errors for each vector. Errors slice will contain nil for successful insertions.
	BatchInsert(ctx context.Context, vectors [][]float32) BatchInsertResult

	// Delete removes a vector from the index
	Delete(ctx context.Context, id core.LocalID) error

	// Update updates a vector in the index
	Update(ctx context.Context, id core.LocalID, v []float32) error

	// KNNSearch performs a K-nearest neighbor search.
	// The opts parameter contains index-specific search options (can be nil for defaults).
	KNNSearch(ctx context.Context, q []float32, k int, opts *SearchOptions) ([]SearchResult, error)

	// KNNSearchWithBuffer performs a K-nearest neighbor search and appends results to the provided buffer.
	// This avoids allocating a new slice for results.
	KNNSearchWithBuffer(ctx context.Context, q []float32, k int, opts *SearchOptions, buf *[]SearchResult) error

	// KNNSearchWithContext performs a K-nearest neighbor search using the provided Searcher context.
	// The results are stored in s.Candidates (MaxHeap).
	// This is the zero-alloc path.
	KNNSearchWithContext(ctx context.Context, s *searcher.Searcher, q []float32, k int, opts *SearchOptions) error

	// KNNSearchStream returns an iterator over K-nearest neighbor search results.
	// Results are yielded in order from nearest to farthest.
	// The iterator supports early termination - stop iterating to cancel.
	// Returns (result, nil) for each result, or (zero, error) on failure.
	KNNSearchStream(ctx context.Context, q []float32, k int, opts *SearchOptions) iter.Seq2[SearchResult, error]

	// BruteSearch performs a brute-force search
	BruteSearch(ctx context.Context, query []float32, k int, filter func(id core.LocalID) bool) ([]SearchResult, error)

	// Stats returns statistics about the index.
	// This method must not write to stdout/stderr.
	Stats() Stats

	// Dimension returns the dimensionality of the vectors in the index.
	Dimension() int
}

// TransactionalIndex consolidates all interfaces required for transactional operations.
// This replaces the separate ApplyIndex, IDAllocator, and VectorGetter interfaces
// to provide a single, cohesive contract for transaction-enabled indexes.
//
// Indexes implementing this interface can participate in atomic multi-subsystem
// transactions coordinated by the engine layer (coordinator + durability).
type TransactionalIndex interface {
	Index

	// ID Allocation
	AllocateID() core.LocalID
	ReleaseID(id core.LocalID)

	// Apply operations (deterministic, used during recovery)
	ApplyInsert(ctx context.Context, id core.LocalID, vector []float32) error
	ApplyBatchInsert(ctx context.Context, ids []core.LocalID, vectors [][]float32) error
	ApplyUpdate(ctx context.Context, id core.LocalID, vector []float32) error
	ApplyDelete(ctx context.Context, id core.LocalID) error

	// Vector retrieval
	VectorByID(ctx context.Context, id core.LocalID) ([]float32, error)
}

// Shard represents an index partition in a sharded architecture.
//
// Sharding enables parallel write throughput by partitioning vectors across multiple
// independent indexes, each with its own lock. This eliminates the global coordinator
// lock bottleneck and provides near-linear scaling on multi-core systems.
//
// Design:
//   - Hash-based sharding: shard = hash(id) % numShards
//   - Each shard is a complete, independent index (HNSW/Flat)
//   - No cross-shard edges in HNSW graphs
//   - Search queries all shards in parallel and merges results
//
// Usage:
//   - Single-shard mode (numShards=1): Standard non-sharded index
//   - Multi-shard mode (numShards>1): Sharded for write scalability
type Shard interface {
	// VectorCount returns the number of vectors in THIS shard only.
	// Total vectors = sum of VectorCount() across all shards.
	VectorCount() int

	// ContainsID returns true if this shard owns the given ID.
	// Ownership is determined by: id % numShards == shardID
	ContainsID(id core.LocalID) bool

	// ShardID returns this shard's identifier (0-based index).
	// For non-sharded indexes, always returns 0.
	ShardID() int

	// NumShards returns the total number of shards in the system.
	// For non-sharded indexes, always returns 1.
	NumShards() int
}
