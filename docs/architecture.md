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

Vecgo is organized in layers, from high-level API down to low-level storage:

```
┌─────────────────────────────────────────────────────────────┐
│                    Vecgo API Layer                          │
│   Builders: HNSW[T](), Flat[T](), DiskANN[T]()             │
│   Operations: Insert, Update, Delete, Search                │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                   Coordinator Layer                         │
│   - Transaction coordination (optional sharding)            │
│   - Metadata integration                                    │
│   - WAL integration                                         │
│   - Search result aggregation (sharded mode)                │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼────────┐  ┌───────▼────────┐  ┌──────▼──────┐
│  Flat Index    │  │  HNSW Index    │  │ DiskANN     │
│  (exact)       │  │  (approximate) │  │ (disk-based)│
└────────────────┘  └────────────────┘  └─────────────┘
        │                   │                   │
┌───────▼───────────────────▼───────────────────▼──────┐
│              Storage Layer                            │
│   - Columnar (in-memory with mmap support)           │
│   - Disk-resident (DiskANN)                          │
│   - Soft deletes + compaction                        │
└───────────────────────────────────────────────────────┘
        │
┌───────▼───────────────────────────────────────────────┐
│         Write-Ahead Log (WAL) - Optional              │
│   - Group commit (batched fsync)                      │
│   - Async / GroupCommit / Sync modes                  │
│   - Auto-checkpoint                                   │
│   - Crash recovery with replay                        │
└───────────────────────────────────────────────────────┘
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

The coordinator is a **proper interface** that orchestrates all mutation operations. This provides clean abstraction and compile-time verification.

**Coordinator Interface**:
```go
type Coordinator[T any] interface {
    Insert(ctx context.Context, vector []float32, data T) (uint64, error)
    BatchInsert(ctx context.Context, vectors [][]float32, data []T) ([]uint64, error)
    Update(ctx context.Context, id uint64, vector []float32, data T) error
    Delete(ctx context.Context, id uint64) error
    Get(ctx context.Context, id uint64) ([]float32, T, error)
    GetMetadata(ctx context.Context, id uint64) (T, error)
    KNNSearch(ctx context.Context, query []float32, k int, filter metadata.Filter) ([]search.Result[T], error)
    BruteSearch(ctx context.Context, query []float32, k int, filter metadata.Filter) ([]search.Result[T], error)
}
```

**Implementations**:

1. **Tx[T]** (Single-Shard Coordinator):
   - Direct pass-through to index
   - Single lock for all writes
   - Metadata updates inline with index operations
   - O(1) Get/GetMetadata by ID

2. **ShardedCoordinator[T]** (Multi-Shard Coordinator):
   - Hash-based vector distribution across N shards
   - Independent locks per shard (parallel writes)
   - Fan-out search to all shards, merge results
   - **GlobalID encoding**: `[ShardID:8 bits][LocalID:24 bits]`
   - O(1) routing for Update/Delete/Get operations
   - **Performance**: 2.7x-3.4x write throughput on multi-core systems

3. **ValidatedCoordinator[T]** (Validation Middleware):
   - Wraps any Coordinator with input validation
   - Checks for nil vectors, NaN/Inf values, dimension mismatches
   - Enforces limits: max dimension, batch size, metadata size
   - Enabled by default, disable with `.WithoutValidation()`

**Key Methods**:
- `Insert(ctx, vector, data)`: Route to shard via GlobalID, update metadata, append WAL
- `Update(ctx, id, vector, data)`: Decode GlobalID for O(1) shard routing
- `Delete(ctx, id)`: Soft delete (tombstone), trigger compaction if threshold exceeded
- `Get(ctx, id)`: Decode GlobalID, retrieve vector + metadata from correct shard
- `KNNSearch(ctx, query, k, filter)`: Fan-out to all shards, merge top-k results, translate GlobalIDs in filters

### 3. Index Implementations

**Flat** (`index/flat/`):
- Brute-force exact search
- Columnar vector storage (SOA layout)
- SIMD-optimized distance computations
- Best for: <100K vectors, 100% recall required

**HNSW** (`index/hnsw/`):
- Hierarchical Navigable Small World graph
- Columnar vector storage
- Arena-allocated connection slices
- Lock-free concurrent reads
- Best for: 100K - 10M vectors, >95% recall acceptable

**DiskANN** (`index/diskann/`):
- Vamana graph with disk-resident vectors
- Product Quantization (PQ) for memory compression
- Beam search for approximate neighbors
- Background compaction for deleted vectors
- Best for: 10M+ vectors, disk-constrained environments

---

## Index Types

### Flat Index

**Storage**:
```go
type Flat[T any] struct {
    store   vectorstore.VectorStore  // Columnar storage
    vectors [][]float32              // Dense vectors (SOA)
    data    []T                       // Associated data
    tombstones map[uint64]struct{}   // Deleted IDs
}
```

**Search Algorithm**:
```go
func (f *Flat) Search(query []float32, k int) []Result {
    // 1. Compute distances to ALL vectors (SIMD-optimized)
    distances := make([]float32, len(f.vectors))
    for i, vec := range f.vectors {
        if _, deleted := f.tombstones[i]; deleted {
            continue  // Skip deleted
        }
        distances[i] = distance.SquaredL2(query, vec)
    }
    
    // 2. Find top-k smallest distances
    return topK(distances, k)
}
```

**Pros**: 100% recall, simple, fast for small datasets  
**Cons**: O(N) search complexity, memory-bound

### HNSW Index

**Storage**:
```go
type HNSW[T any] struct {
    store   vectorstore.VectorStore  // Columnar vectors
    nodes   []*Node                   // Graph nodes
    arena   *arena.Arena              // Connection allocator
}

type Node struct {
    level       int
    connections [][]uint64  // Per-level connection lists
    mu          sync.RWMutex
}
```

**Search Algorithm** (Greedy Best-First):
```go
func (h *HNSW) Search(query []float32, k int, ef int) []Result {
    // 1. Start at entry point
    current := h.entryPoint
    
    // 2. Descend layers (greedy search)
    for level := h.maxLevel; level > 0; level-- {
        current = h.searchLayer(query, current, 1, level)
    }
    
    // 3. Search layer 0 with larger candidate pool (ef)
    candidates := h.searchLayer(query, current, ef, 0)
    
    // 4. Return top-k from candidates
    return topK(candidates, k)
}
```

**Pros**: Sub-linear search (log N), high recall (>95%), fast inserts  
**Cons**: Approximate (not 100% recall), memory overhead for graph

### DiskANN Index

**Storage**:
```go
type DiskANN[T any] struct {
    graph      *VamanaGraph     // Disk-resident graph
    pq         *PQCompressor    // Product Quantization
    vectors    *MmapVectors     // Memory-mapped vectors
    deleted    map[uint64]bool  // Tombstones
    compactor  *BackgroundCompactor
}
```

**Search Algorithm** (Beam Search):
```go
func (d *DiskANN) Search(query []float32, k int, beamWidth int) []Result {
    // 1. Start with random entry points
    beams := randomEntryPoints(beamWidth)
    
    // 2. Beam search iteration
    for iter := 0; iter < L; iter++ {
        // For each beam, expand to neighbors
        for _, beam := range beams {
            neighbors := d.graph.Neighbors(beam.id)
            for _, nid := range neighbors {
                // Disk read (cached)
                vec := d.vectors.Get(nid)
                dist := distance.SquaredL2(query, vec)
                beams = mergeKeepBest(beams, nid, dist, beamWidth)
            }
        }
    }
    
    // 3. Refine with full-precision vectors
    return refineTopK(beams, k)
}
```

**Pros**: Scales to billions of vectors, disk-efficient, compaction support  
**Cons**: Higher latency than HNSW, requires disk I/O

---

## Storage Layer

### Columnar Storage (`vectorstore/columnar/`)

Used by Flat and HNSW for in-memory vectors:

**Layout** (Structure-of-Arrays):
```go
type ColumnarStore struct {
    vectors    []float32  // Flat array: [v0_d0, v0_d1, ..., v1_d0, v1_d1, ...]
    dimension  int
    count      int
    tombstones map[uint64]struct{}  // Soft deletes
}
```

**Benefits**:
- **Cache-friendly**: Sequential access for SIMD
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
- **Vamana Graph**: Adjacency lists on disk
- **Vector File**: Memory-mapped float32 arrays
- **PQ Codes**: Compressed representations
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

### Unified Metadata Store (`metadata/index/`)

Vecgo uses a **Roaring Bitmap-based inverted index** for efficient metadata filtering:

**Architecture**:
```go
type MetadataIndex struct {
    // String fields: value -> roaring bitmap of IDs
    stringIndex map[string]map[string]*roaring.Bitmap
    
    // Numeric fields: sorted list of (value, ID) pairs
    numericIndex map[string]*BTreeIndex
    
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
- **2.7x-3.4x write speedup** on 4-8 cores

```go
// Shard 0        Shard 1        Shard 2        Shard 3
//   Lock 0         Lock 1         Lock 2         Lock 3
//     │              │              │              │
//  Insert 1      Insert 2      Insert 3      Insert 4  (parallel)
```

### Search Coordination

```go
func (c *Coordinator) Search(query []float32, k int) []Result {
    // Fan-out to all shards
    resultChan := make(chan []Result, len(c.shards))
    for _, shard := range c.shards {
        go func(s *Shard) {
            resultChan <- s.Search(query, k)
        }(shard)
    }
    
    // Merge top-k from all shards
    allResults := collectResults(resultChan)
    return mergeTopK(allResults, k)
}
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
   ├─> Coordinator.Search(ctx, query, k)
   │     │
   │     ├─> 1. Fan-out to all shards (parallel)
   │     │     ├─> Shard 0: Index.Search(query, k)
   │     │     ├─> Shard 1: Index.Search(query, k)
   │     │     └─> Shard N: Index.Search(query, k)
   │     │
   │     ├─> 2. Merge results from all shards
   │     ├─> 3. Apply metadata filter (if any)
   │     └─> 4. Return top-k
   │
   └─> Return results to user
```

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
    Close() error
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
