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

**Current technical concept (see `REFACTORING.md`):** Vecgo is a single **tiered engine** (`engine`) with an in-memory L0 hot tier and immutable on-disk segments. External identity uses a persistent `PK -> Location(SegmentID, RowID)` index. Payload/content is stored separately from vectors and can be read via a BlobStore abstraction. Payload interchange is an *edge concern* and must not introduce a core hot-path dependency.

**Cutover plan:** the tiered engine is the default implementation behind the public `vecgo` facade.

**Caching (vNext):** caching is a first-class subsystem, not an afterthought. For local mmap segments, the OS page cache does most of the work; for non-mmap readers (e.g., BlobStore/cloud), vNext needs an explicit bounded segment block cache and snapshot-aware cache keys.

**Code layout (current):**
- `model`: Shared types (`PrimaryKey`, `SegmentID`, `RowID`, `Location`, `SearchOptions`).
- `cache`: Cache primitives/implementations (used where applicable).
- `engine`: Tiered engine orchestrator (snapshots, WAL replay, flush/compaction scheduling).
- `internal/segment`: Segment interfaces and implementations (internal-only).
    - Engine-integrated segment types: `memtable` (L0), `flat` (L1), `diskann` (larger compactions).
- `blobstore`: Blob IO abstraction used by segments/payload readers (local mmap implementation provided).
- `vectorstore`: Internal vector storage interface + columnar implementation (including mmap-backed load).
- `manifest`: Manifest schema and atomic publication.
- `pk`: Persistent Primary Key index.
- `internal/wal`: Write-Ahead Log (internal-only).
- `resource`: ResourceController.

Note: legacy execution paths have been removed; the tree reflects the current engine-first design.

For the current roadmap and planned work (e.g. FlatBuffers headers), see the phase plan in `REFACTORING.md`.

```
┌─────────────────────────────────────────────────────────────┐
│                    Vecgo API Layer                          │
│   Entry Point: vecgo.Open(...)                              │
│   Operations: Insert, BatchInsert, Delete, Search,          │
│               BatchSearch, SearchThreshold, HybridSearch    │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                   Engine Layer                              │
│   - Write-Ahead Log (WAL)                                   │
│   - MemTable (Mutable, L0)                                  │
│   - Immutable Segments (L1..Ln)                             │
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

### 1. Identity & Addressing (vNext Model)

Vecgo separates **External Identity** from **Internal Execution** to optimize for cache density and performance.

1.  **User Primary Key (PK)**:
    -   Stable, user-provided identifier (e.g., `uint64` or `string`).
    -   Never changes for the lifetime of a record.
    -   Used in all public APIs (Insert, Delete, Search results).

2.  **Internal RowID (`uint32`)**:
    -   Dense, transient, internal index used for **all** hot-path structures (graph adjacency, bitmaps).
    -   **Scope**: Local to a segment.
    -   **Lifecycle**: RowIDs may change during compaction (e.g., when merging segments).
    -   **Benefit**: Reduces graph edge size by 50% (4 bytes vs 8 bytes) and doubles cache density compared to 64-bit IDs.

3.  **Translation Layer**:
    -   **PK Index**: A persistent, crash-safe index (LSM or B-Tree) maps `PK -> Location(SegmentID, RowID)`.
    -   **Reverse Mapping**: Within a segment, store a `RowID -> PK` column for result reconstruction (the `SegmentID` selects the segment; the `RowID` is segment-local).

### 1.1 Segment Publication (Manifest)

Segmented persistence relies on an atomic publish protocol:

*   **Manifest**: a small file describing live segments + schema/codec/quantizer metadata.
*   **Atomic publish**: write new segment(s) + updated manifest to temporary paths, then `rename()` into place.
*   **Recovery**: load last valid manifest; replay committed WAL entries; ignore orphan temp files.

### 2. Vecgo API (`vecgo.go`)

The main entry point provides:
- **Entry Point**: `vecgo.Open(dir, dim, metric, ...opts)`
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
- **WAL**: Ensures durability of writes.

**Key Components**:
- `Engine`: The main struct that coordinates everything.
- `Snapshot`: A point-in-time view of the database (Manifest + Segments).
- `Compactor`: Background worker that merges segments.

### 3. SIMD Compute Layer (`internal/simd`)

Vecgo centralizes all performance-critical math in a dedicated layer to ensure "best-in-class" speed on modern hardware.

*   **Single Choke Point**: All vector operations (Dot Product, L2 Distance, Quantized Lookup) route through `internal/simd`.
*   **Runtime Dispatch**: Detects CPU features (AVX-512, AVX2, NEON) at startup and hot-swaps function pointers.
*   **Batch Kernels**:
    *   `SquaredL2Batch`: Computes distance from one query to N vectors in a single call, amortizing function overhead.
    *   `PqAdcLookup`: Optimized lookup table generation for Product Quantization.
*   **Safety**:
    *   **Bounds Checks**: Eliminated inside the kernel (unsafe) for speed.
    *   **Validation**: Callers (Segments) must validate lengths before calling kernels.
    *   **Fallback**: Pure Go implementations provided for `noasm` builds or unsupported architectures.

Float16 note: vNext treats float16 as a *storage* dtype. A spec-based binary16 codec exists in `internal/f16` to decode to float32 for indexing/reranking; any SIMD/batch acceleration should layer on top of that.

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
    *   Allocates memory in large chunks (default 1MB) and hands out slices via lock-free CAS.
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
    *   Current implementation (Jan 2026): a hard limit only (`MemoryLimitBytes`) enforced via a weighted semaphore; there is no distinct `SoftLimit` signal and no automatic flush/backpressure policy wired into the engine.
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
- `internal/wal/`: WAL append + replay for crash recovery.
- `internal/segment/*`: immutable segment files produced by flush/compaction.

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

### Architecture (vNext)

```
┌────────────────────────────────────────────────────┐
│  Engine                                            │
│   - Appends operations to WAL                      │
│   - Applies to active MemTable                     │
│   - GroupCommit (fsync batching)                   │
└────────────────────────────────────────────────────┘
           │
           ▼
┌────────────────────────────────────────────────────┐
│  WAL Segment Writer                                │
│   - Binary format: [Header|Entry|Entry|...]       │
│   - Checksums for corruption detection            │
│   - Durability modes: Async / Sync                │
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

### Replay Mechanism (vNext)

On startup, the system performs a crash recovery:
1.  **Load Snapshot**: The latest valid snapshot is loaded into memory.
2.  **Replay WAL**: The WAL is replayed from the point of the last snapshot.
3.  **Idempotency**: Operations are applied to the MemTable/segments to restore the state to the moment of the crash.

### Storage Abstraction (BlobStore)

To support future cloud-native deployments (S3/GCS) without rewriting the engine, vNext may access segment files via a `BlobStore` abstraction.

Current implementation (Jan 2026): there is no `BlobStore` abstraction in code yet; segments open local files directly (mmap + `os.File`). Treat `BlobStore` as a design target / TODO.

*   **Reader**: Returns a reader for a specific blob/file. For local disk, this supports mmap. For cloud, it might buffer.
*   **Writer**: Returns a writer for a specific blob/file.
*   **Delete**: Deletes a blob/file.

While vNext is "disk-first" (local SSD), defining this interface early prevents "local file path" assumptions from leaking into the Segment logic.

---

## MemTable Lifecycle (LSM-Tree)

For high-throughput writes (especially in DiskANN), Vecgo uses an LSM-tree inspired lifecycle for MemTables:

1.  **Hot (Mutable)**:
    *   Active MemTable receiving new writes.
    *   Stored in `internal/arena` for fast allocation.
    *   Concurrent reads allowed; single writer.

2.  **Flushing (Immutable Queue)**:
    *   When the Hot MemTable fills up, it is rotated and appended to an immutable flush queue.
    *   A background worker consumes the queue and flushes tables to the main index (or disk segment).
    *   This decouples ingestion from flushing, preventing write stalls and deadlocks.
    *   Backpressure is applied if the queue grows too large.

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
| **Sync** | Every operation | Slowest | Strongest (no loss) |

Configuration note (Jan 2026): `engine.WALOptions` supports Async/Sync. `engine.Open` uses Sync by default, but can be configured via `WithWALOptions` (or `vecgo.WithWALOptions`).

GroupCommit (batch fsync) is implemented to improve throughput in Sync mode.

### Recovery Process (vNext)

On startup, the engine performs recovery roughly as:

1. Load the manifest (published segments).
2. Open segment files.
3. Replay WAL records after the last durable point into the active MemTable.

See `engine/engine.go` and the recovery notes in `REFACTORING.md` for the up-to-date behavior.

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

Vecgo contains a `metadata` package with a **bitmap-based inverted index** intended for fast predicate evaluation during search.

**Current implementation (Reality snapshot)**:

- `metadata.UnifiedIndex` maintains interned documents and per-field/per-value bitmaps.
- Bitmap compilation is optimized for:
  - `OpEqual` (equality)
  - `OpIn` (membership)
- Other operators (range compares, not-equals, contains, etc.) currently fall back to per-document checks.

Example: build a conjunction (AND) filter with typed values:

```go
fs := metadata.NewFilterSet(
    metadata.Filter{Key: "category", Operator: metadata.OpEqual, Value: metadata.String("tech")},
    metadata.Filter{Key: "year", Operator: metadata.OpGreaterEqual, Value: metadata.Int(2023)},
)
```

**Planned**:

- End-to-end filter pushdown through the engine/segment search APIs.
- Better support for numeric range filters (e.g., binned bitmaps or ordered indexes) so predicates like `price < 100` avoid scans.

---

## Concurrency Model

### Read Concurrency

All indexes support **concurrent reads**:

**HNSW**: Lock-free graph traversal using atomic operations  
**Flat**: Read-only access to vectors (no locks needed)  
**DiskANN**: mmap allows concurrent reads from OS page cache

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

**Performance Benefits**:
- **Zero goroutines created** per search (constant pool size)
- **80-90% less GC pressure** (no stack allocations per request)
- **50-60% lower P99 latency** under high load
- **Backpressure**: Buffered channel prevents resource exhaustion
```

---

## Data Flow

### Insert Operation (vNext)

```
User API
   │
    ├─> Engine.Insert(ctx, pk, vector)
    │     │
    │     ├─> 1. WAL.Append(Upsert)
    │     ├─> 2. Apply to active MemTable (L0)
    │     └─> 3. Update PK -> Location
    │
    └─> Return PK (and/or Location) to user
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

**Key Improvements (Dec 2024)**:
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
    │     ├─> 1. WAL.Append(Delete)
    │     ├─> 2. Update PK index (remove mapping)
    │     ├─> 3. Mark tombstone (memtable or immutable segment bitmap)
    │     └─> 4. Background flush/compaction materializes deletes
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
| **WAL** | Append | O(1) | Sequential write |
| **WAL** | Replay | O(# entries) | Sequential read |

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
- **Durability**: WAL (Async/Sync) + snapshots; GroupCommit is a TODO
- **Scalability**: Sharding for multi-core, DiskANN for billions of vectors

For performance tuning, see [docs/tuning.md](tuning.md).
