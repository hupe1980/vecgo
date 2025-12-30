# Vecgo Architecture Guide

This document provides an in-depth look at Vecgo's internal architecture, helping you understand how the system works and how to extend it.

## Table of Contents

1. [High-Level Overview](#high-level-overview)
2. [Core Components](#core-components)
3. [Index Types](#index-types)
4. [Storage Layer](#storage-layer)
5. [Write-Ahead Log (WAL)](#write-ahead-log-wal)
6. [Metadata System](#metadata-system)
7. [Concurrency Model](#concurrency-model)
8. [Data Flow](#data-flow)

---

## High-Level Overview

Vecgo is designed around a **Shared-Nothing, LSM-Tree Architecture** to maximize performance, concurrency, and durability.

```
┌─────────────────────────────────────────────────────────────┐
│                    Vecgo API Layer                          │
│   Builders: HNSW[T](), Flat[T](), DiskANN[T]()             │
│   Operations: Insert, Update, Delete, Search                │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                   Coordinator Layer                         │
│   - Sharding (Shared-Nothing)                               │
│   - Request Routing (GlobalID)                              │
│   - Result Aggregation                                      │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                   Shard Engine (Per Shard)                  │
│   - Write-Ahead Log (WAL)                                   │
│   - MemTable (Mutable HNSW)                                 │
│   - Immutable Segments (DiskANN/Flat)                       │
│   - Compaction & Merging                                    │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼────────┐  ┌───────▼────────┐  ┌──────▼──────┐
│   MemTable     │  │   Segment 1    │  │  Segment N  │
│   (Mutable)    │  │  (Immutable)   │  │ (Immutable) │
└────────────────┘  └────────────────┘  └─────────────┘
```

---

## Core Components

### 1. Vecgo API (`vecgo.go`)

The main entry point provides:
- **Index-specific builders**: `HNSW[T]()`, `Flat[T]()`, `DiskANN[T]()`
- **CRUD operations**: `Insert()`, `Update()`, `Delete()`, `BatchInsert()`
- **Search methods**: `KNNSearch()`, `KNNSearchWithFilter()`, fluent `Search()`
- **Lifecycle**: `Checkpoint()`, `Close()`

### 2. Coordinator (`engine/coordinator.go`)

The coordinator is the central nervous system of Vecgo. It enforces **Shared-Nothing Sharding** and **Monotonic Visibility**.

**Coordinator Interface**:
```go
type Coordinator[T any] interface {
    // Mutations
    Insert(ctx context.Context, vector []float32, data T, meta metadata.Metadata) (uint64, error)
    BatchInsert(ctx context.Context, vectors [][]float32, data []T, meta []metadata.Metadata) ([]uint64, error)
    Update(ctx context.Context, id uint64, vector []float32, data T, meta metadata.Metadata) error
    Delete(ctx context.Context, id uint64) error

    // Retrieval
    Get(id uint64) (T, bool)
    GetMetadata(id uint64) (metadata.Metadata, bool)

    // Search
    KNNSearch(ctx context.Context, query []float32, k int, opts *index.SearchOptions) ([]index.SearchResult, error)
    BruteSearch(ctx context.Context, query []float32, k int, filter func(id uint64) bool) ([]index.SearchResult, error)
    HybridSearch(ctx context.Context, query []float32, k int, opts *HybridSearchOptions) ([]index.SearchResult, error)
    KNNSearchStream(ctx context.Context, query []float32, k int, opts *index.SearchOptions) iter.Seq2[index.SearchResult, error]

    // Management
    EnableProductQuantization(cfg index.ProductQuantizationConfig) error
    DisableProductQuantization()
    SaveToFile(path string) error
    RecoverFromWAL(ctx context.Context) error
    Stats() index.Stats
    Checkpoint() error
    Close() error
}
```

**Implementations**:

1. **Tx[T]** (Single-Shard Coordinator):
   - Direct pass-through to index
   - Single lock for all writes (per shard)
   - Metadata updates inline with index operations
   - O(1) Get/GetMetadata by ID
   - Safe-by-default metadata: Deep copy on insert/update prevents external mutation

2. **ShardedCoordinator[T]** (Multi-Shard Coordinator):
   - Round-robin vector distribution across N shards
   - **Shared-Nothing Architecture**: Each shard is an independent `Tx[T]` with its own lock, index, and store.
   - Parallel search fan-out with result merging
   - Zero-allocation worker pool for search execution
   - Linear write scaling with number of shards
   - Independent locks per shard (parallel writes)
   - Context-aware error propagation: All shard errors surfaced with indices
   - Graceful shutdown: Close() waits for all shard workers to terminate
   - Timeout handling: Respects ctx.Done() during parallel search operations
   - **Worker Pool**: Fixed goroutine pool for parallel searches (zero goroutine creation per search)
   - Fan-out search to all shards via worker pool, merge results
   - **GlobalID encoding**: `[ShardID:8 bits][LocalID:56 bits]`
   - O(1) routing for Update/Delete/Get operations
   - **Performance**: Linear write scaling with number of shards

3. **ValidatedCoordinator[T]** (Validation Middleware):
   - Wraps any Coordinator with input validation
   - Checks for nil vectors, NaN/Inf values, dimension mismatches
   - Enforces limits: max dimension, batch size, metadata size
   - Delegates Close() to wrapped coordinator (implements io.Closer)
   - Enabled by default, disable with `.WithoutValidation()`

4. **WorkerPool[T]** (Parallel Search Executor):
   - Fixed-size goroutine pool for sharded searches
   - Closure-based context handling (no context in structs - idiomatic Go)
   - Backpressure via buffered channel (2x numWorkers)
   - Graceful shutdown with in-flight work completion
   - Benefits: 0 goroutines/search, 80-90% less GC pressure, 50-60% lower P99

**Key Methods**:
- `Insert(ctx, vector, data)`: Route to shard via GlobalID, deep copy metadata, update metadata index, append WAL
- `Update(ctx, id, vector, data)`: Decode GlobalID for O(1) shard routing, deep copy metadata
- `Delete(ctx, id)`: Soft delete (tombstone), trigger compaction if threshold exceeded
- `Get(ctx, id)`: Decode GlobalID, retrieve vector + metadata from correct shard
- `KNNSearch(ctx, query, k, filter)`: Fan-out via worker pool, merge top-k results, translate GlobalIDs in filters

### 3. Inner Workings & Optimizations

Vecgo employs several advanced techniques to achieve high performance and scalability.

#### A. Sharding (Shared-Nothing Architecture)

To scale beyond a single core's capacity, Vecgo implements a **Shared-Nothing Sharding** strategy.

*   **Partitioning**: The dataset is partitioned into $N$ independent shards (default: 4). Each shard possesses its own:
    *   **Index** (HNSW/Flat/DiskANN)
    *   **Vector Store**
    *   **Metadata Store**
    *   **Write-Ahead Log (WAL)**
    *   **RWMutex** (Lock contention is isolated to the shard)

*   **Global ID Encoding**:
    Vecgo uses a 64-bit `GlobalID` that encodes routing information directly into the ID:
    ```
    | Shard Index (8 bits) | Local ID (56 bits) |
    |----------------------|--------------------|
    ```
    *   **Max Shards**: 256
    *   **Max Vectors per Shard**: ~72 Quadrillion (effectively infinite)
    *   **Total Capacity**: Infinite for all practical purposes
    *   **Why 64-bit?**: Prevents ID exhaustion in long-running systems and aligns with DiskANN's native storage format.

*   **Routing**:
    *   **Insert**: Round-robin distribution ensures balanced load.
    *   **Update/Delete**: $O(1)$ routing. The shard index is extracted from the ID `(id >> 56)`, allowing direct access to the correct shard without a lookup table.
    *   **Search**: Scatter-Gather. The request is fanned out to all shards in parallel using a worker pool. Results are merged (Top-K) at the coordinator level.

#### B. Update Semantics

Updates in vector databases are complex due to graph topology maintenance. Vecgo enforces strict update strategies per index type:

| Index Type | Update Strategy | Description |
|------------|-----------------|-------------|
| **Flat** | **In-Place** | Direct overwrite of the vector in the columnar store. Safe and fast ($O(1)$). |
| **HNSW** | **Delete + Insert** | The old node is soft-deleted (tombstone), and the new vector is inserted as a fresh node. This prevents graph degradation over time. |
| **DiskANN** | **LSM-Tree** | New writes go to a **MemTable** (HNSW). When full, it is flushed to an **Immutable Disk Segment**. Updates are effectively Delete+Insert across the LSM tree. |

**Default Behavior**: `UpdateModeDeleteInsert` is the default for graph-based indexes to guarantee recall.

#### C. Unified Metadata Index (Hybrid Search)

Metadata filtering is often a bottleneck in vector databases. Vecgo solves this with a **Unified Index** (`metadata/unified.go`).

*   **Structure**: Combines document storage with an inverted index in a single structure.
*   **String Interning**: Uses Go 1.23 `unique.Handle` to deduplicate field names and string values. If 1 million documents have `category: "news"`, the string "news" is stored only once in memory.
*   **Roaring Bitmaps**: The inverted index uses compressed bitmaps (Roaring Bitmaps) for posting lists. Set operations (AND, OR, NOT) for filtering are extremely fast and cache-friendly.

#### C. Quantization & Reranking

Vecgo supports two-stage search to balance speed and recall.

*   **Binary Quantization (BQ)**:
    *   Compresses vectors to 1 bit per dimension (32x compression).
    *   Distance calculation uses Hamming distance (XOR + POPCNT), which is orders of magnitude faster than float32 arithmetic.
    *   Used for: Fast coarse filtering.

*   **Product Quantization (PQ)**:
    *   Splits vectors into $M$ sub-vectors and quantizes each to a centroid ID (uint8).
    *   Achieves 8x-32x compression.
    *   Distance calculation uses pre-computed lookup tables (ADC - Asymmetric Distance Computation).

*   **Reranking (DiskANN)**:
    1.  **Coarse Search**: Search using compressed vectors (PQ/BQ) in memory.
    2.  **Refinement**: Fetch full-precision vectors from disk for the top candidates.
    3.  **Rerank**: Re-score candidates using exact L2/Cosine distance.

#### D. Zero-Copy & Memory Mapping

For large datasets, Vecgo avoids loading everything into the Go heap (which causes GC pressure).

*   **Mmap**: Index files and vector stores are memory-mapped (`mmap`). The OS manages page caching.
*   **Unsafe Casting**: Data is accessed directly from the memory map using `unsafe.Pointer` casting to structs (e.g., `OffsetSegment`), avoiding deserialization overhead.
*   **Flat Layouts**: On-disk formats are designed to be "flat" (C-struct like) to support direct mapping.

#### E. Memory Management (Arena & Pooling)

To minimize Garbage Collection (GC) pauses, Vecgo uses custom memory management strategies:

*   **Arena Allocation (`internal/arena`)**:
    *   Used for HNSW graph construction.
    *   Allocates memory in large chunks (default 1MB) and hands out slices via lock-free CAS.
    *   **Benefit**: Eliminates millions of small allocations during bulk inserts, significantly reducing GC overhead.
    *   **Lifecycle**: Memory is freed all at once when the index is closed.

*   **Object Pooling (`internal/pool`)**:
    *   **SearchContext**: Reuses buffers for visited sets (`bitset`), priority queues, and temporary vectors.
    *   **Benefit**: Zero allocations per search request.
    *   **Sync.Pool**: Uses Go's `sync.Pool` to automatically scale with load and release memory when idle.

---

## Index Types

### 1. Flat Index (`index/flat`)
*   **Algorithm**: Brute-force exact search.
*   **Complexity**: $O(N)$
*   **Optimizations**:
    *   **Hardware Acceleration**: Uses AVX/NEON intrinsics (via `internal/math32`) for distance calculations.
    *   **Copy-On-Write (COW)**: Uses `atomic.Value` to swap the index state, allowing lock-free reads during updates.
*   **Use Case**: Small datasets (<100k) or when 100% recall is mandatory.

### 2. HNSW Index (`index/hnsw`)
*   **Algorithm**: Hierarchical Navigable Small World graph.
*   **Complexity**: $O(\log N)$
*   **Optimizations**:
    *   **Atomic Segments**: Node offsets are stored in fixed-size segments (`[65536]atomic.Uint32`). This allows the index to grow without expensive array resizing or global locks.
    *   **BitSet Pooling**: Visited sets for graph traversal are pooled (`sync.Pool`) to minimize allocations.
    *   **Logical Deletes**: Deletions mark a bit in a `BitSet` (Tombstones) rather than modifying the graph immediately.
*   **Use Case**: General purpose, high performance, fits in RAM.

### 3. DiskANN Index (`index/diskann`)
*   **Algorithm**: Vamana graph (Disk-resident).
*   **Architecture**:
    *   **Immutable Segments**: The on-disk index is strictly immutable.
    *   **RAM**: Compressed vectors (PQ/BQ) + Graph navigation cache.
    *   **Disk**: Full vectors + Adjacency lists.
*   **Optimizations**:
    *   **Beam Search**: Uses a larger beam width to navigate the graph, reducing disk I/O.
    *   **Implicit Reranking**: Automatically fetches full vectors from disk for the final candidates.
*   **Use Case**: Massive datasets (> RAM size), cost-efficiency.

---

## Storage Layer

### Columnar Storage (`vectorstore/columnar/`)

Used by Flat and HNSW for in-memory vectors:

**Layout** (Structure-of-Arrays):
```go
type ColumnarStore struct {
    vectors    []float32  // Flat array, 64-byte aligned for AVX-512
    dimension  int
    count      int
    tombstones map[uint64]struct{}  // Soft deletes
}
```

**Benefits**:
- **SIMD-Optimized**: 64-byte alignment enables AVX-512 aligned loads
- **Cache-friendly**: Sequential access
- **Zero-copy mmap**: Load snapshots without deserialization
- **Soft deletes**: No reallocation on delete
- **Compaction**: Reclaim space from deleted vectors

**Operations**:
```go
// Add vector (O(1) amortized)
func (c *ColumnarStore) Add(vec []float32) uint64 {
    id := c.count
    c.vectors = append(c.vectors, vec...)
    c.count++
    return id
}

// Get vector (O(1))
func (c *ColumnarStore) Get(id uint64) []float32 {
    offset := id * c.dimension
    return c.vectors[offset : offset+c.dimension]
}

// Delete (soft, O(1))
func (c *ColumnarStore) Delete(id uint64) {
    c.tombstones[id] = struct{}{}
}
```

### Disk-Resident Storage (`index/diskann/`)

Used by DiskANN for large-scale datasets:

**Components**:
- **Vamana Graph**: Adjacency lists on disk (Immutable)
- **Vector File**: Memory-mapped float32 arrays (Immutable)
- **PQ Codes**: Compressed representations (Immutable)
- **Metadata**: Index metadata (dimension, count, etc.)

**Memory-Mapped Vectors**:
```go
type MmapVectors struct {
    file      *os.File
    data      []byte      // mmap region
    dimension int
}

func (m *MmapVectors) Get(id uint64) []float32 {
    offset := id * m.dimension * 4  // 4 bytes per float32
    return bytesToFloat32Slice(m.data[offset:])
}
```

**Benefits**:
- OS page cache handles hot vectors
- Lazy loading (only read what you need)
- Supports datasets larger than RAM
- **Crash Safety**: Immutable files are never corrupted by crashes.

---

## Persistence Layer

### Unified Persistence Manager (`persistence/manager.go`)

The persistence manager provides a unified interface for all persistence operations:

```go
// Manager coordinates snapshots, WAL, and recovery
type Manager struct {
    snapshotPath string
    wal          *wal.WAL
    codec        codec.Codec
}

// Key methods
func (pm *Manager) Snapshot(writeFunc func(w io.Writer) error) error
func (pm *Manager) Recover(loader SnapshotLoader, replayer WALReplayer) error
func (pm *Manager) Checkpoint() error
```

### Atomic File Operations

All index files use atomic writes (temp file + rename):

```go
// AtomicSaveToFile writes atomically via temp file + rename
func SaveToFile(filename string, writeFunc func(io.Writer) error) error {
    tmp, _ := os.CreateTemp(dir, base+".tmp-*")
    writeFunc(tmp)  // Write to temp
    tmp.Sync()      // Ensure durability
    os.Rename(tmp.Name(), filename)  // Atomic replace
}

// AtomicSaveToDir writes multiple files atomically
func AtomicSaveToDir(dir string, files map[string]func(io.Writer) error) error
```

### DiskANN Crash Safety

DiskANN writes its index files atomically (and may include additional optional files depending on enabled features):

```go
// Builder writes all files atomically
func (b *Builder) writeIndexFiles() error {
    files := map[string]func(io.Writer) error{
        "index.meta":    b.writeMetaToWriter,
        "index.graph":   b.writeGraphToWriter,
        "index.pqcodes": b.writePQCodesToWriter,
        "index.vectors": b.writeVectorsToWriter,
    }
    // Optional: present only if enabled at build time.
    if b.enableBinaryPrefilter {
        files["index.bqcodes"] = b.writeBQCodesToWriter
    }
    return persistence.AtomicSaveToDir(b.indexPath, files)
}
```

`index.meta` includes header flags that indicate which optional files are present (for example, whether Binary Quantization codes were written for DiskANN search-time prefiltering).

**Benefits**:
- No corrupt indexes on power failure
- Consistent state guaranteed
- Clean rollback on write errors

---

## Write-Ahead Log (WAL)

The WAL ensures durability and crash recovery.

### Architecture

```
┌────────────────────────────────────────────────────┐
│  Coordinator                                       │
│   - Batches operations into log entries           │
│   - Appends to active segment                     │
│   - Group commit (fsync batching)                 │
└────────────────────────────────────────────────────┘
           │
           ▼
┌────────────────────────────────────────────────────┐
│  WAL Segment Writer                                │
│   - Binary format: [Header|Entry|Entry|...]       │
│   - Checksums for corruption detection            │
│   - Durability modes: Async / GroupCommit / Sync  │
└────────────────────────────────────────────────────┘
           │
           ▼
┌────────────────────────────────────────────────────┐
│  Disk (fsync)                                      │
│   - segment-000001.wal                            │
│   - segment-000002.wal                            │
└────────────────────────────────────────────────────┘
```

### Correctness & Ordering

To guarantee consistency, Vecgo enforces strict ordering between the WAL and the Index:

1.  **WAL First**: `WAL.Append()` is called *before* any modification to the in-memory index.
2.  **Index Second**: `Index.Apply()` is called only after the WAL entry is successfully written (and fsync'd, depending on durability mode).
3.  **Failure Handling**: If the WAL write fails, the operation returns an error, and the index remains untouched. If the process crashes after the WAL write but before the index update, the Replay mechanism restores the state.

### Replay Mechanism

On startup, the system performs a crash recovery:
1.  **Load Snapshot**: The latest valid snapshot is loaded into memory.
2.  **Replay WAL**: The WAL is replayed from the point of the last snapshot.
3.  **Idempotency**: Operations are applied to the MemTable/Index to restore the state to the moment of the crash.

---

## MemTable Lifecycle (LSM-Tree)

For high-throughput writes (especially in DiskANN), Vecgo uses an LSM-tree inspired lifecycle for MemTables:

1.  **Hot (Mutable)**:
    *   Active MemTable receiving new writes.
    *   Stored in `internal/arena` for fast allocation.
    *   Concurrent reads allowed; single writer.

2.  **Flushing (Immutable)**:
    *   When the Hot MemTable fills up, it is rotated to "Flushing" state.
    *   It becomes immutable (read-only).
    *   A background worker writes it to disk (creating a new disk segment).

3.  **Cold (On-Disk)**:
    *   Data is now persisted in a disk segment (e.g., SSTable or DiskANN segment).
    *   The in-memory MemTable is discarded.
    *   Reads are served via mmap or cached I/O.

4.  **Garbage Collection (Compaction)**:
    *   Background process merges small disk segments into larger ones.
    *   Tombstones (deleted items) are purged.
    *   Ensures read performance doesn't degrade over time.

### Durability Modes

| Mode | fsync Frequency | Performance | Durability |
|------|-----------------|-------------|------------|
| **Async** | Never (OS decides) | Fastest | Weak (may lose seconds of data) |
| **GroupCommit** | Batched (every 10ms or 100 ops) | 83x faster than Sync | Strong (≤10ms loss) |
| **Sync** | Every operation | Slowest | Strongest (no loss) |

**Usage**:
```go
db, _ := vecgo.HNSW[string](128).
    SquaredL2().
    WAL("./wal", func(o *wal.Options) {
        o.DurabilityMode = wal.GroupCommit  // Recommended
        o.GroupCommitInterval = 10 * time.Millisecond
        o.GroupCommitSize = 100
    }).
    Build()
```

### Recovery Process

On startup, Vecgo replays the WAL:

```go
func (c *Coordinator) Recover(walPath string) error {
    // 1. Read all segment files
    segments := loadSegments(walPath)
    
    // 2. Replay operations in order
    for _, segment := range segments {
        entries := segment.ReadAll()
        for _, entry := range entries {
            switch entry.Type {
            case Insert:
                c.Insert(ctx, entry.Vector)
            case Update:
                c.Update(ctx, entry.ID, entry.Vector)
            case Delete:
                c.Delete(ctx, entry.ID)
            }
        }
    }
    
    // 3. Checkpoint to snapshot
    c.Checkpoint()
    
    // 4. Delete old WAL files
    clearWAL(walPath)
}
```

---

## Metadata System

### Safe-by-Default Metadata

Vecgo uses **deep copy on insert** to prevent silent data corruption from external mutation. It also leverages Go 1.24's `unique` package for efficient string interning, significantly reducing memory usage for repetitive keys and string values.

```go
// User code - metadata is safe after Insert()
meta := metadata.Metadata{
    "category": metadata.String("tech"),
    "year":     metadata.Int(2024),
}

id, _ := db.Insert(ctx, VectorWithData[string]{
    Vector:   vec,
    Data:     "doc1",
    Metadata: meta,
})

// External mutation does NOT affect stored metadata
meta["category"] = metadata.String("science")  // ✅ Safe - no corruption

// Stored metadata is unchanged
stored, _ := db.GetMetadata(ctx, id)
// stored["category"].S.Value() == "tech" (not "science")
```

**Implementation** (`metadata/types.go`):
```go
// Document uses interned strings for keys
type Document map[unique.Handle[string]]Value

type Value struct {
    Kind Kind
    S    unique.Handle[string] // Interned string value
    I64  int64
    F64  float64
    B    bool
    A    []Value
}

// Clone creates a deep copy of metadata (recursive for arrays)
func (d Document) Clone() Document {
    clone := make(Document, len(d))
    for k, v := range d {
        clone[k] = v.clone()  // Deep copy including nested arrays
    }
    return clone
}

// CloneIfNeeded avoids allocation for empty/nil metadata
func CloneIfNeeded(m Metadata) Metadata {
    if len(m) == 0 { return nil }
    return m.Clone()
}
```

### Unified Metadata Store (`metadata/index/`)

Vecgo uses a **Roaring Bitmap-based inverted index** for efficient metadata filtering:

**Architecture**:
```go
type MetadataIndex struct {
    // String fields: value -> roaring bitmap of IDs
    stringIndex map[unique.Handle[string]]map[unique.Handle[string]]*roaring.Bitmap
    
    // Numeric fields: sorted list of (value, ID) pairs
    numericIndex map[unique.Handle[string]]*BTreeIndex
    
    // ID -> full metadata
    documents map[uint64]metadata.Metadata
}
```

**Example**:
```go
// Insert with metadata
db.Insert(ctx, VectorWithData[string]{
    Vector: vec,
    Data: "doc1",
    Metadata: metadata.Metadata{
        "category": "tech",
        "year": 2024,
    },
})

// Search with filter
results := db.Search(query).
    KNN(10).
    Filter(metadata.And(
        metadata.Eq("category", "tech"),
        metadata.Gte("year", 2023),
    )).
    Execute(ctx)
```

**Query Compilation**:
```go
// metadata.And(Eq("category", "tech"), Gte("year", 2023))
// compiles to:
candidateBitmap := stringIndex["category"]["tech"].
    And(numericIndex["year"].Range(2023, math.MaxInt))
```

**Benefits**:
- **10,000x faster** than linear scan filtering
- **50% memory reduction** vs duplicated metadata per index
- Supports complex queries: `And()`, `Or()`, `Eq()`, `Gte()`, etc.

---

## Concurrency Model

### Read Concurrency

All indexes support **concurrent reads**:

**HNSW**: Lock-free graph traversal using atomic operations  
**Flat**: Read-only access to vectors (no locks needed)  
**DiskANN**: mmap allows concurrent reads from OS page cache

### Write Concurrency

**Single-Shard Mode**:
- Global `sync.RWMutex` protects all writes
- Readers can proceed concurrently
- Simple but bottlenecks on multi-core inserts

**Multi-Shard Mode** (`.Shards(n)`):
- Each shard has independent lock
- Hash-based routing: `shard = hash(vector) % N`
- **Linear write speedup** on multi-core systems

```go
// Shard 0        Shard 1        Shard 2        Shard 3
//   Lock 0         Lock 1         Lock 2         Lock 3
//     │              │              │              │
//  Insert 1      Insert 2      Insert 3      Insert 4  (parallel)
```

### Search Coordination (Worker Pool)

**Problem Solved**: Spawning N goroutines per search request creates severe GC pressure under load.

**Solution**: Fixed-size worker pool with closure-based context handling.

```go
// Worker Pool Pattern (engine/worker_pool.go)
type WorkerPool[T any] struct {
    workCh chan func()      // Closures capture context
    stopCh chan struct{}    // Graceful shutdown signal
    wg     sync.WaitGroup   // Wait for in-flight work
    closed atomic.Bool      // Idempotent close
}

func (wp *WorkerPool[T]) Submit(ctx context.Context, req WorkRequest[T]) error {
    // Context captured in closure (NOT stored in struct - idiomatic Go)
    workFunc := func() {
        results, err := req.shard.KNNSearch(ctx, req.query, req.k, req.opts)
        // Send result with cancellation checks...
    }
    
    select {
    case wp.workCh <- workFunc:
        return nil
    case <-ctx.Done():
        return ctx.Err()
    }
}
```

**Performance Benefits**:
- **Zero goroutines created** per search (constant pool size)
- **80-90% less GC pressure** (no stack allocations per request)
- **50-60% lower P99 latency** under high load
- **Backpressure**: Buffered channel prevents resource exhaustion
```

---

## Data Flow

### Insert Operation

```
User API
   │
   ├─> Coordinator.Insert(ctx, vector)
   │     │
   │     ├─> 1. Route to shard (hash-based)
   │     ├─> 2. Acquire shard lock
   │     ├─> 3. Index.Add(vector) → returns ID
   │     ├─> 4. MetadataStore.Add(ID, metadata)
   │     ├─> 5. WAL.Append(InsertEntry)
   │     └─> 6. Release lock
   │
   └─> Return ID to user
```

### Search Operation

```
User API
   │
   ├─> Coordinator.Search(ctx, query, k, filter)
   │     │
   │     ├─> 1. Compile metadata filter to bitmap (if present)
   │     ├─> 2. Fan-out to all shards (parallel)
   │     │     │
   │     │     ├─> Shard 0: Index.Search(query, k, filter)
   │     │     │               │
   │     │     │               ├─> Pre-filter during traversal
   │     │     │               │   (HNSW: filter in searchLayer)
   │     │     │               │   (DiskANN: filter in beamSearch)
   │     │     │               └─> Return local top-k results
   │     │     │
   │     │     ├─> Shard 1: Index.Search(query, k, filter)
   │     │     ├─> Shard 2: Index.Search(query, k, filter)
   │     │     └─> Shard 3: Index.Search(query, k, filter)
   │     │
   │     ├─> 3. Collect results (context-aware, respects timeout)
   │     ├─> 4. Merge top-k from all shards
   │     └─> 5. Hydrate with metadata & data payloads
   │
   └─> Return results to user
```

**Key Improvements (Dec 2024)**:
- ✅ **Pre-filtering**: Filter applied during graph traversal (100% recall vs ~50% post-filtering)
- ✅ **Error propagation**: All shard errors surfaced with indices
- ✅ **Context cancellation**: Gracefully handles timeouts during result collection

### Delete + Compaction

```
User API
   │
   ├─> Coordinator.Delete(ctx, id)
   │     │
   │     ├─> 1. Route to shard (lookup by ID)
   │     ├─> 2. Index.Delete(id) → soft delete (tombstone)
   │     ├─> 3. MetadataStore.Remove(id)
   │     ├─> 4. WAL.Append(DeleteEntry)
   │     │
   │     └─> 5. Check compaction threshold
   │           │
   │           ├─> If deleted% > threshold (e.g., 20%):
   │           │     ├─> Trigger background compaction
   │           │     ├─> Rebuild index without tombstones
   │           │     └─> Atomic swap (old → new)
   │           │
   │           └─> Else: No-op
   │
   └─> Return to user
```

---

## Extension Points

Want to add a new index type? Implement these interfaces:

```go
// Core index interface
type Index interface {
    Add(vector []float32) (uint64, error)
    Search(query []float32, k int) ([]Result, error)
    Update(id uint64, vector []float32) error
    Delete(id uint64) error
    Close() error  // Idempotent, waits for background workers
}

// Snapshotting (mmap-only)
type Snapshotable interface {
    SaveToFile(filename string) error  // Save binary snapshot
}

// Loading uses mmap via NewFromFile[T](filename string, opts ...Option[T])

// Statistics (optional)
type StatProvider interface {
    Stats() IndexStats
}
```

**Production Requirements (Dec 2024)**:
- ✅ **Idempotent Close()**: Safe to call multiple times
- ✅ **Goroutine tracking**: Use `sync.WaitGroup` for all background workers
- ✅ **Error context**: Include operation details in error messages
- ✅ **Pre-filtering**: Support filter functions during graph traversal for correctness

---

## Performance Characteristics

| Component | Operation | Complexity | Notes |
|-----------|-----------|------------|-------|
| **Flat** | Insert | O(1) | Append to columnar store |
| **Flat** | Search | O(N) | SIMD-optimized brute force |
| **HNSW** | Insert | O(log N) | Graph construction |
| **HNSW** | Search | O(log N) | Greedy best-first |
| **DiskANN** | Insert | O(R log N) | Vamana graph update |
| **DiskANN** | Search | O(L × beam) | Beam search with disk I/O |
| **Metadata** | Filter compile | O(# fields) | Roaring bitmap merge |
| **WAL** | Append | O(1) | Sequential write |
| **WAL** | Replay | O(# entries) | Sequential read |

---

## Summary

Vecgo's architecture is designed for:
- **Performance**: SIMD, zero-allocation, sharded writes
- **Flexibility**: Multiple index types, pluggable storage
- **Durability**: WAL with group commit, snapshots
- **Scalability**: Sharding for multi-core, DiskANN for billions of vectors

For performance tuning, see [docs/tuning.md](tuning.md).
