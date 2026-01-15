# Vecgo Architecture Guide

This document provides an in-depth look at Vecgo's internal architecture, helping you understand how the system works and how to extend it.

## Table of Contents

1. [High-Level Overview](#high-level-overview)
2. [Core Components](#core-components)
3. [Index Types](#index-types)
4. [Storage Layer](#storage-layer)
5. [Durability Model](#durability-model)
6. [Metadata System](#metadata-system)
7. [Concurrency Model](#concurrency-model)
8. [Data Flow](#data-flow)

---

## High-Level Overview

Vecgo is designed around a **Shared-Nothing, LSM-Tree Architecture** to maximize performance, concurrency, and durability.

**Current technical concept:** Vecgo is a single **tiered engine** (`engine`) with an in-memory L0 hot tier and immutable on-disk segments. External identity uses a persistent `PK -> Location(SegmentID, RowID)` index. Payload/content is stored separately from vectors and can be read via a BlobStore abstraction. Payload interchange is an *edge concern* and must not introduce a core hot-path dependency.

**Cutover plan:** the tiered engine is the default implementation behind the public `vecgo` facade.

**Caching (vNext):** caching is a first-class subsystem, not an afterthought. For local mmap segments, the OS page cache does most of the work; for non-mmap readers (e.g., BlobStore/cloud), vNext needs an explicit bounded segment block cache and snapshot-aware cache keys.

**Code layout (current):**
- `model`: Shared types (`PrimaryKey`, `SegmentID`, `RowID`, `Location`, `SearchOptions`).
- `blobstore`: Abstraction for immutable data segments (supports Local/S3).
- `cache`: Cache primitives/implementations (used where applicable).
- `engine`: Tiered engine orchestrator (snapshots, commit/flush, compaction scheduling).
- `internal/segment`: Segment interfaces and implementations (internal-only).
    - Engine-integrated segment types: `memtable` (L0), `flat` (L1), `diskann` (larger compactions).
- `blobstore`: Blob IO abstraction used by segments/payload readers (local mmap implementation provided).
- `vectorstore`: Internal vector storage interface + columnar implementation (including mmap-backed load).
- `manifest`: Manifest schema and atomic publication.
- `pk`: Persistent Primary Key index.
- `resource`: ResourceController.

Note: legacy execution paths have been removed; the tree reflects the current engine-first design.

```
┌─────────────────────────────────────────────────────────────┐
│                    Vecgo API Layer                          │
│   Entry Point: vecgo.Open(ctx, ...)                         │
│   Operations: Insert, BatchInsert, Delete, Search,          │
│               BatchSearch, SearchThreshold, HybridSearch    │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                   Engine Layer                              │
│   - MemTable (Mutable, L0)                                  │
│   - Immutable Segments (L1..Ln)                             │
│   - Commit-Oriented Durability                              │
│   - Compaction & Merging                                    │
│   - Snapshot Isolation                                      │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                   Searcher Context                          │
│   - Reusable Scratch Memory (Heaps, BitSets, Buffers)       │
│   - Zero-Allocation Execution Path                          │
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

### 1. Identity & Addressing (Auto-Increment Model)

Vecgo uses a high-performance **Auto-Increment Primary Key** system to maximize insert throughput and cache locality.

1.  **Internal ID (`uint64`)**:
    -   **Primary Key**: A globally unique, monotonically increasing 64-bit integer assigned by Vecgo.
    -   **Persistence**: The strict ordering allows the primary key index to be a **Paged MVCC Array**, eliminating hashing overhead and providing O(1) access.
    -   **Validation**: The index supports Multi-Version Concurrency Control (MVCC) with time-travel queries (Snapshots).
    -   **Lifecycle**: Durable for the lifetime of the record.
    -   **Usage**: All internal references (graph edges, bitmaps) and public APIs (Search results, Deletes) use this ID.

2.  **External ID Mapping (Optional layer equivalent)**:
    -   If users require external IDs (e.g., UUIDs), they should store them in the `Metadata` blob.
    -   Lookups by external ID use the standard inverted index (metadata filter).

3.  **Physical Addressing (`Location`)**:
    -   The system maintains a lightweight in-memory `ID -> Location(SegmentID, RowID)` mapping with version history.
    -   **SegmentID**: Identifies the immutable segment (or MemTable).
    -   **RowID**: The offset within that segment.
    -   **Optimization**: Because IDs are sequential, this mapping is highly compressible and cache-friendly. The Paged Array structure ensures no re-hashing or massive copy overhead during growth.

### 1.1 Segment Publication (Manifest)

Segmented persistence relies on an atomic publish protocol:

*   **Manifest**: a small file describing live segments + schema/codec/quantizer metadata.
*   **Atomic publish**: write new segment(s) + updated manifest to temporary paths, then `rename()` into place.
*   **Recovery**: load last valid manifest; ignore orphan temp files. No WAL replay needed.

### 2. Vecgo API (`vecgo.go`)

The main entry point provides:
- **Entry Point**: `vecgo.Open(ctx, backend, ...opts)`
- **CRUD operations**: `Insert()`, `Delete()`
- **Search methods**: `Search()`
- `Get(pk)` point lookup by primary key (**Implemented**; returns vector, metadata, and payload.)
- **Lifecycle**: `Close()`

### 3. Engine (`engine/engine.go`)

The Engine is the central nervous system of vNext. It manages the lifecycle of data across memory and disk.

**Responsibilities**:
- **Write Buffer (L0)**: Manages the in-memory `MemTable` for fast writes.
- **Segment Management**: Manages immutable segments (L1..Ln) on disk.
- **Compaction**: Background merging of segments to reclaim space and optimize search.
- **Snapshot Isolation**: Provides consistent views for readers.
- **Commit-Oriented Durability**: `Commit()` writes immutable segments atomically.

**Key Components**:
- `Engine`: The main struct that coordinates everything.
- `Snapshot`: A point-in-time view of the database (Segments + Tombstones + Active MemTable).
- `Compactor`: Background worker that merges segments.

#### Visibility & Versioning (Production Contract)

For production workloads, the system must have an unambiguous “which version is visible?” rule.

Current design principle:

- External identity is `PK`.
- Physical storage is `Location(SegmentID, RowID)`.

V2 target (breaking changes allowed):

- The published read view becomes a single immutable snapshot object, and includes the PK index view.
- A candidate is visible iff `pkIndex[PK] == (SegmentID, RowID)` in the same snapshot (and it is not deleted).

This rule prevents duplicate PKs in results and prevents “resurrection” of stale versions after compaction.

### 3. SIMD Compute Layer (`internal/simd`)

Vecgo centralizes all performance-critical math in a dedicated layer to ensure "best-in-class" speed on modern hardware.

*   **Single Choke Point**: All vector operations (Dot Product, L2 Distance, Quantized Lookup) route through `internal/simd`.
*   **Runtime Dispatch**: Detects CPU features (AVX-512, AVX, NEON) at startup and hot-swaps function pointers.
*   **Batch Kernels**:
    *   `SquaredL2Batch`: Computes distance from one query to N vectors in a single call, amortizing function overhead.
    *   `PqAdcLookup`: Optimized lookup table generation for Product Quantization.
*   **Safety**:
    *   **Bounds Checks**: Eliminated inside the kernel (unsafe) for speed.
    *   **Validation**: Callers (Segments) must validate lengths before calling kernels.
    *   **Fallback**: Pure Go implementations provided for `noasm` builds or unsupported architectures.

Float16 note: vNext will treat float16 as a *storage* dtype. SIMD batch conversion (`F16ToF32`) exists in `internal/simd` for fast float16→float32 decoding on AVX2/AVX-512/NEON. Storage integration is planned for a future release.

## Searcher Context (Internal Optimization)

To achieve maximum performance and zero allocations in the steady state, Vecgo uses the `Searcher` context internally.

**Problem**: Traditional search implementations allocate memory for:
- Priority queues (candidates)
- Visited sets (bitsets/maps)
- Scratch vectors (decompression)
- IO buffers (disk reads)

**Solution**: The `Searcher` struct owns all these resources.
- **Reusable**: Managed by the engine's worker pool.
- **Typed**: Uses value-based heaps (`PriorityQueueItem`) to avoid pointer overhead.
- **Sized**: Pre-allocated to the maximum index size.

```go
type Searcher struct {
    Visited        *VisitedSet
    Candidates     *PriorityQueue
    ScratchCandidates *PriorityQueue
    ScratchVec     []float32
    IOBuffer       []byte
    Heap           *CandidateHeap
    Results        []model.Candidate
}
```

The vNext `engine` acquires and releases `Searcher` contexts from a pool during execution, ensuring near-zero-allocation behavior for the search path without exposing complexity to the user.

### SIMD Compute Layer

Vecgo’s hot-path vector math (dot products, squared L2, normalization, and PQ ADC lookup) is centralized in `internal/simd` and used by the `distance` package.

Key properties:

- **Runtime dispatch**: selects AVX/AVX-512 (amd64) or NEON (arm64) when available via `golang.org/x/sys/cpu`, otherwise falls back to a generic implementation.
- **Build control**: compile with `-tags noasm` to disable assembly and force the generic path.
- **Safety model**: SIMD kernels assume inputs are well-formed (e.g., matching lengths); callers validate dimensions at API boundaries.

### 3. Inner Workings & Optimizations

Vecgo employs several advanced techniques to achieve high performance and scalability.

#### A. Legacy: Sharding (Coordinator-era)

This section described the pre-vNext coordinator-based sharding architecture.

**Current reality (vNext)**: the default stack (`vecgo` → `engine` → `segment/*`) does not implement internal sharding. If you want multi-process or multi-shard behavior, run multiple engine instances and shard/rout at the application level.

#### B. Update Semantics

Updates in vector databases are complex due to graph topology maintenance. Vecgo enforces strict update strategies per index type:

| Index Type | Update Strategy | Description |
|------------|-----------------|-------------|
| **Flat** | **Copy-On-Write (Chunk Swap)** | Updates must not tear under lock-free readers. Write the new bytes into a new buffer (vector- or chunk-granularity) and atomically publish the swap. |
| **HNSW** | **Delete + Insert** | The old node is soft-deleted (tombstone), and the new vector is inserted as a fresh node. This prevents graph degradation over time. |
| **DiskANN** | **LSM-Tree** | New writes go to a **MemTable** (HNSW). When full, it is flushed to an **Immutable Disk Segment**. Updates are effectively Delete+Insert across the LSM tree. |

**Default Behavior**: `UpdateModeDeleteInsert` is the default for graph-based indexes to guarantee recall.

Note: “In-place overwrite” is only safe if reads are synchronized with the write (e.g. via a lock). Under lock-free reads, Flat updates must be treated as copy-on-write publication (or delete+insert) to avoid torn vectors.

Concurrency contract (must be upheld everywhere atomic publication is used):

- Fully initialize data first (write all bytes).
- Publish the pointer/length only after initialization.
- After publish, treat the published snapshot as immutable.

#### C. Unified Metadata Index (Hybrid Search)

Metadata filtering is often a bottleneck in vector databases. Vecgo solves this with a **Unified Index** (`metadata/unified.go`).

*   **Structure**: Combines document storage with an inverted index in a single structure.
*   **String Interning**: Uses Go 1.24 `unique.Handle` to deduplicate field names and string values. If 1 million documents have `category: "news"`, the string "news" is stored only once in memory.
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

*   **Mmap (`internal/mmap`)**: Uses a custom, platform-agnostic `mmap` abstraction.
    *   **Mapping**: Represents a memory-mapped file.
    *   **Region**: Safe views into subsections of the mapping.
    *   **Access Patterns**: Uses `Advise` (e.g., `AccessSequential`) to optimize OS page caching.
*   **Unsafe Casting**: Data is accessed directly from the memory map using `unsafe.Pointer` casting to structs (e.g., `OffsetSegment`), avoiding deserialization overhead.
*   **Flat Layouts**: On-disk formats are designed to be "flat" (C-struct like) to support direct mapping.

#### E. Memory Management (Arena & Pooling)

To minimize Garbage Collection (GC) pauses, Vecgo uses custom memory management strategies:

*   **Arena Allocation (`internal/arena`)**:
    *   Used for HNSW graph construction.
    *   Allocates memory in large chunks (default 4MB) and hands out slices via lock-free CAS.
    *   **Hot-Path APIs**: `Alloc` (~4.2ns/op) provides efficient allocation for HNSW inner loops with proper error handling (no panics).
    *   **Stats Control**: `arenaStatsEnabled` compile-time flag to disable stats tracking in production.
    *   **Zero Panics**: All allocation functions return errors instead of panicking - production code never crashes.
    *   **Benefit**: Eliminates millions of small allocations during bulk inserts, significantly reducing GC overhead.
    *   **Lifecycle**: Memory is freed all at once when the index is closed.

*   **Object Pooling (`sync.Pool` + engine pools)**:
    *   **SearchContext**: Reuses buffers for visited sets (`bitset`), priority queues, and temporary vectors.
    *   **Benefit**: Drives search toward near-zero allocations in steady state (verify via benchmarks; treat allocs/op as a regression metric).
    *   **Sync.Pool**: Uses Go's `sync.Pool` to automatically scale with load and release memory when idle.

*   **Aligned Memory (`internal/mem`)**:
    *   **SIMD Alignment**: Vectors in `MemTable` and other critical paths are allocated with 64-byte alignment (AVX-512 friendly).
    *   **Custom Allocator**: Uses `mem.AllocAlignedFloat32` to ensure backing arrays start at aligned addresses, enabling efficient SIMD loads.

#### F. Resource Controller (Governance)

To ensure stability under load, Vecgo uses a **ResourceController** to manage global limits:

*   **Memory Budget**: Tracks estimated usage of L0 arenas, caches, and mmap overhead.
    *   Current implementation: a hard limit (`MemoryLimitBytes`) enforced via a weighted semaphore. The controller is wired into the MemTable allocator and the block cache.
    *   Note: the engine has automatic commit triggers (MemTable size thresholds), but the commit signal is best-effort (buffered size 1, dropped when already pending). This is not a complete admission-control or backpressure policy.
*   **Concurrency Budget**: Limits the number of active goroutines for background tasks (compaction, build) using a weighted semaphore.
*   **IO Budget**: (Optional) Token bucket for disk IOPS to prevent compaction from starving search.

#### G. Memory Lifetime & Safety Rules

Since Vecgo uses manual memory management (Arenas, mmap, unsafe pointers), strict rules are enforced to prevent segfaults and corruption:

1.  **Arena Generations**:
    -   Each Arena has a monotonically increasing `GenerationID`.
    -   Any stored reference must carry `(GenerationID, Offset)`.
    -   On access: `if ref.Gen != arena.Gen { panic("stale arena reference") }`

2.  **Compaction Barrier / Reclamation**:
    -   Compaction MUST NOT reclaim/close memory that can still be observed by readers.
    -   Enforce via either:
        -   **Ref-counted snapshots/segments** (simple, correct; current vNext approach), or
        -   **Epoch barrier (RCU-style)** (lower per-read overhead).

3.  **GC Visibility Rule**:
    -   Clearing the last Go reference to arena-backed memory does not imply safety.
    -   Offsets are invisible to GC and must be treated as raw pointers.

4.  **Arena Boundary Rule**:
    -   No arena-backed pointer, slice, or offset may cross a public API boundary.
    -   All arena-backed data must be translated to stable representations (PK + Location, copied vectors, metadata) before returning to the caller.

---

## Segment Types (vNext)

Note: The refactoring plan trends toward a **disk-first segmented engine**, but this does **not** mean “disk-only”. In-memory indexes (Flat/HNSW) remain critical as the L0 write buffer/hot tier and for small datasets where RAM fits.

### 1. Flat Segment (`internal/segment/flat`)
*   **Algorithm**: Brute-force exact search.
*   **Complexity**: $O(N)$
*   **Optimizations**:
    *   **Hardware Acceleration**: Uses AVX/NEON intrinsics (via `internal/simd`) for distance calculations.
*   **Use Case**: Small datasets (<100k) or when 100% recall is mandatory.

### 2. MemTable (L0) + HNSW (`internal/segment/memtable` + `internal/hnsw`)
*   **Algorithm**: Hierarchical Navigable Small World graph.
*   **Complexity**: $O(\log N)$
*   **Optimizations**:
    *   **Atomic Segments**: Node offsets are stored in fixed-size segments (`[65536]atomic.Uint32`). This allows the index to grow without expensive array resizing or global locks.
    *   **BitSet Pooling**: Visited sets for graph traversal are pooled (`sync.Pool`) to minimize allocations.
    *   **Logical Deletes**: Deletions mark a bit in a `BitSet` (Tombstones) rather than modifying the graph immediately.
*   **Use Case**: General purpose, high performance, fits in RAM.

### 3. DiskANN Segment (`internal/segment/diskann`)
*   **Algorithm**: Vamana graph (Disk-resident).
*   **Architecture**:
    *   **Immutable Segments**: The on-disk index is strictly immutable.
    *   **RAM**: Compressed vectors (PQ/BQ) + Graph navigation cache.
    *   **Disk**: Full vectors + Adjacency lists.
*   **Optimizations**:
    *   **Beam Search**: Uses a larger beam width to navigate the graph, reducing disk I/O.
    *   **Implicit Reranking**: Automatically fetches full vectors from disk for the final candidates.
    *   **Dual Compression**: LZ4 (hot data) or ZSTD (cold data) with 256KB blocks for optimal I/O.
        *   **LZ4**: ~6.1μs compress, ~5.3μs decompress - balanced for hot segments.
        *   **ZSTD**: ~5.2μs compress, ~13μs decompress - better ratio for archival/cold data.
        *   Compression skipped when ratio > 0.9 (incompressible data).
        *   Block-based I/O enables efficient random access and streaming decompression.
*   **Use Case**: Massive datasets (> RAM size), cost-efficiency.

### 4. Partitioned Flat (IVF)
*   **Algorithm**: Inverted File Index with K-Means clustering.
*   **Architecture**:
    *   **Centroids**: Stored in segment header.
    *   **Partitioning**: Vectors are physically reordered by partition on disk.
    *   **Probing**: Search finds top `nprobes` centroids and scans only those partitions.
    *   **Quantization**: Optional SQ8 (Scalar Quantization) for compressed candidate generation (4x smaller).
*   **Use Case**: Large disk-resident segments where full scan is too slow. Produced by compaction.

---

## Storage Layer

### Many Files vs Single-File Bundles (Packfile)

Vecgo’s storage model is **immutable segments + manifest publication**. Whether segments are stored as separate files or bundled is an implementation detail, as long as atomic publication and recovery semantics remain intact.

Decision: do **not** use generic archive formats as the segment storage format.

If we add a single-file option, the production-grade version is a **purpose-built packfile**:

- footer/superblock index mapping entry name/type -> (offset, length)
- per-entry checksums + strict bounds validation
- alignment for efficient mmap/range reads
- optional per-entry compression only for cold payload blobs

Planned direction: keep multi-file mode as default, and optionally support bundle mode for operational simplicity and object-store backends.

### Columnar Storage (`vectorstore/`)

Used by Flat and HNSW for in-memory vectors:

**Layout** (Structure-of-Arrays):
```go
type ColumnarStore struct {
    vectors    []float32  // Flat array, 64-byte aligned for AVX-512
    dimension  int
    count      int
    tombstones map[model.RowID]struct{}  // Soft deletes
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
func (c *ColumnarStore) Add(vec []float32) model.RowID {
    id := model.RowID(c.count)
    c.vectors = append(c.vectors, vec...)
    c.count++
    return id
}

// Get vector (O(1))
func (c *ColumnarStore) Get(id model.RowID) []float32 {
    offset := int(id) * c.dimension
    return c.vectors[offset : offset+c.dimension]
}

// Delete (soft, O(1))
func (c *ColumnarStore) Delete(id model.RowID) {
    c.tombstones[id] = struct{}{}
}
```

### Disk-Resident Storage (legacy path reference)

Note: historical sections below may mention legacy `index/*` paths. The engine-first implementation uses `internal/segment/diskann` for DiskANN segments.

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

Vecgo persists state via a small set of purpose-built components:

- `manifest/`: atomic publication of the durable checkpoint (including `MaxLSN`) and the set of immutable segments.
- `pk/`: persistent PK index (`PK -> Location(SegmentID, RowID)`).
- `internal/segment/*`: immutable segment files produced by flush/compaction.

> **Note**: WAL has been removed. Vecgo uses commit-oriented durability with append-only versioned commits (like LanceDB/Git).

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

`index.meta` includes header flags that indicate which optional files are present (for example, whether Binary Quantization codes were written for DiskANN search-time prefiltering). The format version (v2+) also records `CompressionType` to enable LZ4 block-compressed segment data.

**Benefits**:
- No corrupt indexes on power failure
- Consistent state guaranteed
- Clean rollback on write errors

---

## Durability Model

Vecgo uses a **Commit-Oriented** architecture for durability — the same pattern used by LanceDB and Git (append-only versioned commits, not WAL-based).

### Architecture

```
┌────────────────────────────────────────────────────┐
│  Engine                                            │
│   - Buffers operations in MemTable (memory)        │
│   - Commit() writes immutable segment to disk      │
│   - Manifest tracks committed segments             │
└────────────────────────────────────────────────────┘
           │
           ▼ Commit()
┌────────────────────────────────────────────────────┐
│  Immutable Segment                                 │
│   - Atomic file write (temp + rename)             │
│   - Binary format: [Header|Vectors|Index|Meta]    │
│   - Checksums for corruption detection            │
└────────────────────────────────────────────────────┘
           │
           ▼
┌────────────────────────────────────────────────────┐
│  Manifest Update                                   │
│   - Atomic PUT of new manifest                    │
│   - CURRENT pointer updated atomically            │
└────────────────────────────────────────────────────┘
```

### Durability Contract

| State | Survives Crash? | Description |
|-------|-----------------|-------------|
| Before `Insert()` | N/A | - |
| After `Insert()`, before `Commit()` | ❌ No | Data buffered in MemTable |
| After `Commit()` | ✅ Yes | Data written to immutable segment |
| After `Close()` | ✅ Yes | `Close()` auto-commits pending data |

This is the same contract as SQLite's explicit transaction mode or Git commits.

### Why No WAL?

Traditional databases use Write-Ahead Logging for single-row transaction durability.
Vector databases have different workload characteristics:

| Workload | Pattern | Needs WAL? |
|----------|---------|------------|
| **RAG pipelines** | Batch embed → batch insert → commit | ❌ No |
| **Semantic search** | Build index offline → deploy → query | ❌ No |
| **Recommendations** | Periodic batch updates → serve queries | ❌ No |
| **ML embeddings** | Checkpoint after training epoch | ❌ No |

The commit-oriented model eliminates:
- WAL rotation complexity
- Crash recovery replay time  
- Checkpointing overhead
- ~500+ lines of recovery code

### Storage Abstraction (Cloud-Native Backend)

To enable serverless and cloud-native deployments, Vecgo uses a **Pluggable Storage Backend** that abstracts the file system.

**Interface Definition**:

```go
type StorageBackend interface {
    // Core Object Storage
    Get(ctx context.Context, key string) ([]byte, error)
    Put(ctx context.Context, key string, data []byte) error
    Delete(ctx context.Context, key string) error
    List(ctx context.Context, prefix string) ([]string, error)

    // Atomic Primitives (Critical for concurrency)
    // CompareAndSwap is required for manifest updates to ensure linearizable history.
    CompareAndSwap(ctx context.Context, key string, oldData, newData []byte) (bool, error)

    // Streaming (Optimization)
    GetReader(ctx context.Context, key string) (io.ReadCloser, error)
    PutWriter(ctx context.Context, key string) (io.WriteCloser, error)
}
```

**Implementations**:
1.  **LocalFS**: Default for embedded/local use. Uses `rename()` for atomicity and `os.File` for streaming.
2.  **S3/GCS/Azure**: Uses object storage with conditional writes (If-Match) to implement CAS.
3.  **InMemory**: For testing and ephemeral workloads.

**Caching Strategy**:
Cloud backends are high-latency. Vecgo implements a **Segment Block Cache**:
- **Immutable Segments**: Cached locally (LRU) or memory-mapped if using a specialized FUSE/Mount adapter.
- **Manifest**: Always fetched fresh (with CAS) to ensure strong consistency.
- **WAL**: Buffered in memory and flushed transactionally (segmented upload).

**Cloud-Native mmap Approximation (LanceDB Pattern)**:

Object stores (S3/GCS) don't support mmap, but Vecgo approximates mmap semantics via:

1. **Columnar Layout**: Segments are offset-addressable with fixed-size blocks
2. **HTTP Range Reads**: S3 blob uses `Range: bytes=start-end` header (see `blobstore/s3/s3_store.go`)
3. **Lazy Loading**: Headers + offsets are resident (~8KB); vector blocks fetched on-demand
4. **Block Cache**: Shared LRU cache across segments (`blobstore/caching_store.go`)
5. **Index-Constrained Access**: Block skipping via `MatchesBlock()` avoids full scans

```
Cloud Segment Access Pattern:
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Open()     │ →   │ Range GET   │ →   │ Header +    │  (8KB resident)
│  segment    │     │ bytes=0-256 │     │ Offsets     │
└─────────────┘     └─────────────┘     └─────────────┘
                                              │
┌─────────────┐     ┌─────────────┐     ┌─────▼───────┐
│  Search()   │ →   │ BlockCache  │ ←→  │ Lazy Block  │  (256KB per block)
│  k vectors  │     │ Hit/Miss    │     │ Range GETs  │
└─────────────┘     └─────────────┘     └─────────────┘
```

**Memory Impact** (1GB segment on S3):
- Full download (old): 1GB resident
- Lazy + cache (new): 8KB + 100MB cache = ~100MB resident (10x reduction)

**Persistence Model**:
- **Segments**: Immutable blobs (e.g., `seg/001-c8f2.data`).
- **Manifest**: Single mutable object (`MANIFEST-NNNNNN.bin`) updated atomically.
- **CURRENT**: Pointer file to the active manifest.


---

## MemTable Lifecycle (LSM-Tree)

For high-throughput writes (especially in DiskANN), Vecgo uses an LSM-tree inspired lifecycle for MemTables:

1.  **Hot (Mutable)**:
    *   Active MemTable receiving new writes.
    *   Stored in `internal/arena` for fast allocation.
    *   Concurrent reads allowed; single writer.

2.  **Flushing (Immutable Queue)**:
    *   When the Hot MemTable crosses configured thresholds, a flush is triggered via a buffered channel.
    *   The background loop calls `Commit()` when signaled.
    *   There is no explicit flush queue today; multiple triggers can coalesce, and signals can be dropped if one is already pending.
    *   Admission control/backpressure comes from the ResourceController (hard limits), not from a flush-queue growth policy.

3.  **Cold (On-Disk)**:
    *   Data is now persisted in a disk segment (e.g., SSTable or DiskANN segment).
    *   The in-memory MemTable is discarded.
    *   Reads are served via mmap or cached I/O.

4.  **Garbage Collection (Compaction)**:
    *   Background process merges small disk segments into larger ones.
    *   Tombstones (deleted items) are purged.
    *   Ensures read performance doesn't degrade over time.

### Durability

Vecgo uses **commit-oriented durability**:

| Operation | Durable? | Description |
|-----------|----------|-------------|
| `Insert()` | ❌ No | Buffered in MemTable (memory) |
| `Commit()` | ✅ Yes | Writes segment + updates manifest |
| `Close()` | ✅ Yes | Auto-commits pending data |

This model matches LanceDB (Databricks) and Git — optimized for batch vector workloads. Unlike WAL-based databases, committed segments are immutable and self-describing.

### Recovery Process

On startup, the engine performs recovery:

1. Load the manifest (published segments).
2. Open segment files.
3. Rebuild PK index from segments (or load checkpoint).

No WAL replay needed — committed segments are the source of truth.

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

### Unified Metadata Store (metadata filtering)

Vecgo contains a `metadata` package with a **dual-index architecture** for fast predicate evaluation during search.

**Architecture (Jan 2026)**:

The `UnifiedIndex` maintains three synchronized structures:

1. **Document Store**: `map[RowID]InternedDocument` - Primary storage for full metadata documents
2. **BitmapIndex (Inverted)**: `map[field]map[value]*Bitmap` - Pre-materialized bitmaps per unique value
3. **NumericIndex (Columnar)**: Sorted `(value, rowID)` pairs for O(log n) range queries

```
UnifiedIndex
├── documents map[RowID]InternedDocument  (primary storage)
├── inverted  map[field]map[value]*Bitmap (BitmapIndex - low cardinality)
└── numeric   *NumericIndex               (ColumnIndex - high cardinality)
```

**Cost-Based Query Dispatch**:

For numeric comparisons (`<`, `>`, `<=`, `>=`, `!=`), the index uses adaptive dispatch based on field cardinality:

```go
const lowCardinalityThreshold = 512  // Aligned with roaring container boundaries

if distinctValues <= 512 {
    // BitmapIndex: O(cardinality) but with fast bitmap OR
    // Wins for status (5 values), category (50 values), etc.
    result = evaluateNumericFilterScan(filter)
} else {
    // ColumnIndex: O(log n + matches) binary search
    // Wins for timestamps, prices, sensor readings, etc.
    result = numericIndex.EvaluateFilter(filter)
}
```

**Performance Characteristics**:

| Field Type | Cardinality | Strategy | Complexity |
|-----------|-------------|----------|------------|
| `status` | ~5 | BitmapIndex | O(5) = O(1) |
| `category` | ~100 | BitmapIndex | O(100) = O(1) |
| `timestamp` | ~10K | ColumnIndex | O(log 10K) = O(14) |
| `price` | ~50K | ColumnIndex | O(log 50K) = O(16) |

**Benchmark Results** (Apple M4 Pro, 100K documents):

```
# High cardinality (10K distinct values) - "timestamp > X"
NumericIndex:  117μs  (binary search)
InvertedScan:  716μs  (scan all values)
Winner: NumericIndex 6.1x faster ✓

# Low cardinality (100 distinct values) - "category >= X"
NumericIndex:  118μs  (binary search)
InvertedScan:  8.3μs  (OR 50 bitmaps)
Winner: InvertedScan 14x faster ✓
```

**Optimized Operations**:

- `OpEqual`, `OpIn`: Direct bitmap lookup O(1)
- `OpLessThan`, `OpGreaterThan`, etc.: Adaptive dispatch as above
- `OpContains`, `OpStartsWith`: Filter scan (not yet optimized)

**Example Usage**:

```go
fs := metadata.NewFilterSet(
    metadata.Filter{Key: "category", Operator: metadata.OpEqual, Value: metadata.String("tech")},
    metadata.Filter{Key: "year", Operator: metadata.OpGreaterEqual, Value: metadata.Int(2023)},
)
```

**NumericIndex Implementation Details**:

- **Columnar Layout**: Separate `values []float64` and `rowIDs []uint32` arrays for cache-optimal binary search
- **Lazy Sorting**: Online inserts append unsorted; `sortField()` called on first query
- **Batch Operations**: Uses `AddMany()` for efficient bitmap population from range results
- **Persistence**: Delta-encoded for compression; lazy rebuilt on load

---

## Concurrency Model

### Read Concurrency

All indexes support **concurrent reads**:

**HNSW**: Lock-free graph traversal using atomic operations  
**Flat**: Read-only access to vectors (no locks needed)  
**DiskANN**: mmap allows concurrent reads from OS page cache  
**PK Index**: Wait-free O(1) reads via atomic pointer chains

### Global MVCC & Lock-Free Search

As of Jan 2026, Vecgo uses a **Global MVCC** architecture optimized for high-throughput search workloads.

1.  **Lock-Free Primary Key Index**:
    -   Address mapping (`ID -> Location`) uses **Paged MVCC** with per-entry atomic pointer linked-lists.
    -   **Wait-Free Reads**: `Get()` uses `atomic.Load()` to traverse the version chain—no locks acquired.
    -   **Lock-Free Writes**: `Upsert()`/`Delete()` use CAS loops with exponential backoff for contention.
    -   **Version Pooling**: `sync.Pool` reduces allocation overhead for version nodes.
    -   **Memory Safety**: Atomic operations ensure safe concurrent access without data races (verified with `-race`).

2.  **Lock-Free Versioned Tombstones**:
    -   **Problem**: Standard deletion maps require locks or O(N) copy-on-write overhead.
    -   **Solution**: `VersionedTombstones` uses a **Paged/Chunked Copy-On-Write** structure accessed via an `atomic.Pointer`.
    -   **Reads (IsDeleted)**: Wait-free O(1) atomic load.
    -   **Writes (MarkDeleted)**: Optimistic Copy-On-Write of a single 4KB chunk. Write complexity is O(ChunkSize) rather than O(TotalRows), eliminating memory bandwidth saturation during heavy deletes.

3.  **Non-Blocking Search**:
    -   The hot `Search` path acquires no long-held locks.
    -   It uses light-weight Reference Counting (`TryIncRef`) to secure the current snapshot structure.
    -   It uses atomic loads for Tombstone filtering.
    -   Result: Linearly scalable search throughput with core count.

### HNSW Concurrency

The HNSW implementation uses fine-grained locking and atomic operations to support high concurrency:

- **Insertions**:
    - Uses `atomic.Pointer` for node storage (lock-free reads).
    - Uses sharded `sync.RWMutex` for connection updates (minimizes contention).
    - Handles concurrent entry point updates via atomic CAS and retry loops.
    - **Robustness**: Automatically retries if the entry point is deleted concurrently (`ErrEntryPointDeleted`).

- **Searches**:
    - Completely lock-free traversal (except for `visited` set pooling).
    - Safe against concurrent deletions (tombstones are checked atomically).
    - Uses `sync.Pool` for scratch buffers to avoid allocation.
    - **Dynamic EF (ACORN-lite)**: Automatically expands search breadth based on filter selectivity.

### Write Concurrency

**Single-Shard Mode**:
- Global `sync.RWMutex` protects all writes
- Readers can proceed concurrently
- Simple but bottlenecks on multi-core inserts

Legacy note: previous versions discussed multi-shard coordinator routing (`.Shards(n)`). The current vNext engine does not implement internal sharding; scale reads via segment-level parallelism and scale writes via the LSM pipeline (fast L0 + async flush/compaction).

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
        results, err := req.exec(ctx)
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

**Design Notes**:
- This worker-pool sketch is a vNext idea; it is not currently implemented in the engine.
- A bounded queue can apply backpressure to callers (submit blocks / respects context) but does not, by itself, prevent memory exhaustion.
```

---

## Data Flow

### Insert Operation

```
User API
   │
    ├─> Engine.Insert(ctx, pk, vector)
    │     │
    │     ├─> 1. Apply to active MemTable (L0) — NOT durable yet
    │     ├─> 2. Update PK -> Location
    │     └─> 3. Commit() writes immutable segment — DURABLE
    │
    └─> Return PK to user
```

### Search Operation (vNext)

```
User API
   │
    ├─> Engine.Search(ctx, query, k, filter)
   │     │
    │     ├─> 1. Acquire snapshot (atomic)
    │     ├─> 2. Compile filter (bitmap if available, else predicate)
    │     ├─> 3. Search across (L0 + immutable segments)
    │     ├─> 4. Exact rerank and merge to global top-k
    │     └─> 5. Fetch PKs and requested columns
   │
   └─> Return results to user
```

**Key Improvements (Jan 2026)**:
- ✅ **Lock-Free PK Index**: MVCC implemented with atomic pointer linked-lists (CAS) for wait-free reads and lock-free writes.
- ✅ **Dynamic EF (ACORN-lite)**: HNSW search automatically adapts expansion factor (`ef`) based on filter selectivity to maintain recall.
- ✅ **Zero-Overhead Distance**: Mmap store uses direct function pointers, bypassing interface overhead for L2/Cosine/Dot.
- ✅ **Pre-filtering**: Filter applied during graph traversal (100% recall vs ~50% post-filtering)
- ✅ **Vectorized predicate evaluation (vNext)**: Prefer batch checks (e.g., `MatchesBatch`) to reduce branchy per-row overhead
- ✅ **Error propagation**: All shard errors surfaced with indices
- ✅ **Context cancellation**: Gracefully handles timeouts during result collection

### Delete + Compaction

```
User API
   │
    ├─> Engine.Delete(ctx, pk)
   │     │
    │     ├─> 1. Update PK index (remove mapping)
    │     ├─> 2. Mark tombstone (memtable or immutable segment bitmap)
    │     ├─> 3. Commit() persists tombstones — DURABLE
    │     └─> 4. Background compaction materializes deletes
   │
   └─> Return to user
```

---

## Extension Points

Want to add a new segment type? Implement the vNext segment contract (`segment.Segment`).

High-level rule: segments are immutable once published; the engine owns lifecycle and concurrency.

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
| **Commit** | Segment write | O(N) | Atomic file write |

---

## Design Principles & Anti-Patterns

To maintain high performance, Vecgo adheres to strict design principles.

### Core Principles
1.  **Zero-Allocation Steady State**: Hot paths (Search) must not allocate on the heap.
2.  **Data-Oriented Design**: Memory layouts are chosen for cache locality, not OOP convenience.
3.  **Explicit Ownership**: Buffers and IO resources are owned by execution contexts (`Searcher`), not the GC.

### Anti-Patterns (Strictly Forbidden)

1.  **The "Generic Interface" Trap**:
    -   *Bad*: `type Vector interface { Dist(other Vector) float32 }`
    -   *Good*: `func Dist(a, b []float32) float32` (Concrete types, inlinable).

2.  **The "Channel Iterator"**:
    -   *Bad*: `func Search() <-chan Result` (Goroutine leak risk, slow).
    -   *Good*: `func Search(yield func(Result) bool)` (Go 1.23 iterators) or `[]Result`.

3.  **The "Hidden Allocation"**:
    -   *Bad*: `fmt.Sprintf` in hot paths, `interface{}` conversion, closure capture.
    -   *Good*: Zero-alloc logging, typed errors, explicit context passing.

4.  **The "Global Lock"**:
    -   *Bad*: `sync.RWMutex` protecting the whole DB.
    -   *Good*: Shard-level locks + MVCC for lock-free reads.

5.  **The "Interface Boxing" Trap**:
    -   *Bad*: `heap.Push(h, item)` where `h` is `interface{}`.
    -   *Good*: Use typed heaps (generics) or specialized heap implementations.

---

## Summary

Vecgo's architecture is designed for:
- **Performance**: SIMD, zero-allocation, sharded writes
- **Flexibility**: Multiple index types, pluggable storage
- **Durability**: Commit-oriented (like LanceDB/Git) — append-only versioned commits, no WAL
- **Scalability**: Sharding for multi-core, DiskANN for billions of vectors

For performance tuning, see [docs/tuning.md](tuning.md).

## Observability

Vecgo is designed with production observability as a first-class citizen. It uses a push-based model via the `MetricsObserver` interface, allowing integration with any metrics backend (Prometheus, StatsD, OpenTelemetry, etc.) without creating hard dependencies.

### Metrics Observer

The `MetricsObserver` interface (defined in `engine/metrics.go`) provides hooks for all critical critical operations:

- **Write Path**: `OnInsert`, `OnDelete`, `OnCommit`
- **Read Path**: `OnSearch`
- **Background**: `OnFlush`, `OnCompaction`
- **State**: `OnMemTableStatus`

### Zero-Overhead Logging

The Engine accepts an optional `*slog.Logger`. If provided, it logs operational events (flushes, compactions, errors) with structured context. If nil, logging is disabled with zero overhead.

### Usage

To enable observability, inject an implementation of `MetricsObserver` into `engine.Options`:

```go
opts := engine.Options{
    Metrics: &MyPrometheusObserver{}, // Your implementation
    Logger:  slog.Default(),
}
```
