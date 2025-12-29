# Vecgo Performance Tuning Guide

This guide helps you optimize Vecgo for your specific workload and requirements.

## Table of Contents

1. [Index Selection](#index-selection)
2. [HNSW Parameters](#hnsw-parameters)
3. [Sharding for Multi-Core](#sharding-for-multi-core)
4. [Quantization](#quantization)
5. [DiskANN Configuration](#diskann-configuration)
6. [WAL Durability Modes](#wal-durability-modes)
7. [Memory Optimization](#memory-optimization)
8. [Search Performance](#search-performance)
9. [Benchmarking](#benchmarking)

---

## Index Selection

Choose the right index for your dataset size and accuracy requirements:

### Decision Tree

```
Dataset Size?
│
├─ <100K vectors
│   └─> Use Flat (exact search)
│       - 100% recall
│       - Fastest for small datasets
│       - Simple, no tuning required
│
├─ 100K - 10M vectors
│   └─> Use HNSW (approximate)
│       - 95-99% recall (tunable)
│       - Fast in-memory search
│       - Good insert performance
│
└─ >10M vectors or limited RAM
    └─> Use DiskANN (disk-resident)
        - 90-95% recall (tunable)
        - Scales to billions
        - Slower than HNSW but disk-efficient
```

### Quick Comparison

| Index | Dataset Size | Recall | Insert Speed | Search Speed | Memory |
|-------|-------------|--------|--------------|--------------|--------|
| **Flat** | <100K | 100% | Fastest | O(N) brute force | Lowest |
| **HNSW** | 100K-10M | 95-99% | Fast | O(log N) | Medium |
| **DiskANN** | 10M+ | 90-95% | Medium | O(L × beam) | Lowest (disk) |

### Example Usage

```go
// Small dataset: Flat
db := vecgo.Flat[string](128).
    SquaredL2().
    Build()

// Medium dataset: HNSW
db := vecgo.HNSW[string](128).
    SquaredL2().
    M(32).
    EF(200).
    Build()

// Large dataset: DiskANN
db := vecgo.DiskANN[string]("./data", 128).
    SquaredL2().
    R(64).
    L(100).
    Build()
```

---

## HNSW Parameters

HNSW has three key parameters that control the accuracy/speed trade-off:

### M (Number of Connections per Layer)

**What it does**: Controls graph connectivity. Higher M = more connections = better recall.

**Trade-offs**:
- ↑ M → ↑ Recall, ↑ Memory, ↓ Search speed
- ↓ M → ↓ Recall, ↓ Memory, ↑ Search speed

**Recommended values**:

| Use Case | M | Memory Impact | Recall |
|----------|---|---------------|--------|
| **Low memory** | 8-16 | Minimal | 90-95% |
| **Balanced** (default) | 16-32 | Moderate | 95-98% |
| **High recall** | 32-64 | High | 98-99.5% |
| **Maximum** | 64-128 | Very high | 99.5%+ |

**Example**:
```go
// Default (balanced)
db := vecgo.HNSW[string](128).M(16).Build()

// High recall (more memory)
db := vecgo.HNSW[string](128).M(64).Build()
```

### EF (Exploration Factor during Search)

**What it does**: Controls search beam width. Higher EF = more candidates explored = better recall.

**Trade-offs**:
- ↑ EF → ↑ Recall, ↓ Search speed
- ↓ EF → ↓ Recall, ↑ Search speed

**Recommended values**:

| Use Case | EF | Search Speed | Recall |
|----------|-----|--------------|--------|
| **Fast search** | 50-100 | Fastest | 85-92% |
| **Balanced** (default) | 100-200 | Fast | 92-97% |
| **High recall** | 200-400 | Medium | 97-99% |
| **Maximum** | 400-800 | Slow | 99%+ |

**Dynamic EF**: You can override EF at search time:
```go
db := vecgo.HNSW[string](128).EFConstruction(200).Build()  // Build-time EF

// Override at search time
results := db.Search(query).
    KNN(10).
    EF(400).  // Higher EF for this query
    Execute(ctx)
```

### EFConstruction (Build-Time Exploration)

**What it does**: Controls exploration during graph construction. Higher EFConstruction = better graph quality.

**Trade-offs**:
- ↑ EFConstruction → ↑ Build time, ↑ Index quality, ↑ Search recall
- ↓ EFConstruction → ↓ Build time, ↓ Index quality, ↓ Search recall

**Recommended values**:

| Use Case | EFConstruction | Build Time | Index Quality |
|----------|----------------|------------|---------------|
| **Fast build** | 100-200 | Fastest | Good |
| **Balanced** (default) | 200-400 | Fast | Better |
| **High quality** | 400-800 | Slow | Best |

**Note**: EFConstruction is set during index creation via `EFConstruction()` and cannot be changed later.

### Heuristic (Neighbor Selection)

**What it does**: Enables a more selective neighbor selection algorithm.

**Trade-offs**:
- `true` (default): Better recall, slightly slower inserts
- `false`: Faster inserts, slightly lower recall

**Recommended**: Keep default (`true`) unless you need maximum insert speed.

```go
db := vecgo.HNSW[string](128).
    M(32).
    EFConstruction(200).
    Heuristic(true).  // Default, recommended
    Build()
```

### HNSW Configuration Examples

```go
// Fast and memory-efficient (90-95% recall)
db := vecgo.HNSW[string](128).
    SquaredL2().
    M(12).
    EF(100).
    Build()

// Balanced (default, 95-98% recall)
db := vecgo.HNSW[string](128).
    SquaredL2().
    M(16).
    EF(200).
    Build()

// High recall (98-99.5% recall)
db := vecgo.HNSW[string](128).
    SquaredL2().
    M(48).
    EF(400).
    Build()

// Maximum recall (99.5%+ recall)
db := vecgo.HNSW[string](128).
    SquaredL2().
    M(96).
    EF(800).
    Build()
```

---

## Sharding for Multi-Core

Sharding eliminates the global lock bottleneck, enabling parallel writes. Vecgo uses a **Shared-Nothing** architecture where each shard manages its own Index, Store, and Write-Ahead Log (WAL).

### When to Use Sharding

**Use sharding when**:
- Multi-core CPU (4+ cores)
- High write throughput required
- Concurrent inserts from multiple goroutines

**Don't use sharding when**:
- Single-core CPU
- Read-heavy workload (no benefit)
- Small datasets (<10K vectors)

### Performance Impact

| Cores | Shards | Speedup (Insert) | Notes |
|-------|--------|------------------|-------|
| 1 | 1 | 1.0x (baseline) | No benefit from sharding |
| 4 | 1 | 1.0x | Global lock bottleneck |
| 4 | 4 | 2.7-3.2x | Near-linear scaling |
| 8 | 4 | 3.0-3.4x | Diminishing returns |
| 8 | 8 | 3.4-3.8x | Best scaling |

**Search performance**: Sharding has **no negative impact** on search (fan-out is parallel).

**Durability**: Each shard has its own dedicated WAL. This means recovery is also parallelized, significantly reducing startup time after a crash.

### Configuration

```go
// No sharding (default)
db := vecgo.HNSW[string](128).
    SquaredL2().
    Build()

// 4 shards (recommended for 4-8 cores)
db := vecgo.HNSW[string](128).
    SquaredL2().
    Shards(4).
    Build()

// Match CPU core count
numCores := runtime.NumCPU()
db := vecgo.HNSW[string](128).
    SquaredL2().
    Shards(numCores).
    Build()
```

### Shard Count Guidelines

**Rule of thumb**: Use `Shards(runtime.NumCPU())` for write-heavy workloads.

| CPU Cores | Recommended Shards | Expected Speedup |
|-----------|-------------------|------------------|
| 1-2 | 1 | - |
| 4 | 4 | 2.7-3.2x |
| 8 | 4-8 | 3.0-3.8x |
| 16+ | 8-16 | 3.5-4.0x |

**Note**: Sharding beyond 16 shards has diminishing returns.

---

## Quantization

Quantization reduces memory usage at the cost of slight accuracy loss.

### Binary Quantization

**Compression**: 32x (1 bit per dimension)  
**Use case**: Pre-filtering, coarse search  
**Accuracy**: Lower (Hamming distance approximation)

```go
// Binary quantization for Flat index
db := vecgo.Flat[string](128).
    SquaredL2().
    BinaryQuantization().  // 32x compression
    Build()
```

**Performance**:
- Memory: 128-dim float32 (512 bytes) → 16 bytes (32x reduction)
- Distance: 0.68ns for 128-dim Hamming distance (ultra-fast)
- Recall: 70-85% (suitable for filtering)

#### DiskANN: Binary Prefilter (Search-Only)

DiskANN can optionally use Binary Quantization as a **coarse traversal-time prefilter** during search. This is intentionally **not** used for graph construction, updates, or final reranking.

**What it does**:
- Encodes the query and candidate nodes into binary codes
- Skips expanding candidates whose normalized Hamming distance is above a threshold
- Preserves the primary DiskANN scoring pipeline (PQ traversal + float32 rerank)

```go
// DiskANN with optional BQ traversal prefilter.
// The threshold is a normalized Hamming distance in [0, 1].
db := vecgo.DiskANN[string]("./data", 128).
    SquaredL2().
    PQSubvectors(8).
    BinaryPrefilter(0.25).
    Build()
```

**Notes**:
- Enabling this writes an additional on-disk file (`index.bqcodes`).
- You must enable it at build time; opening an existing index without BQ codes cannot enable it later.

### Product Quantization (PQ)

**Compression**: 8-32x (configurable)  
**Use case**: Large datasets, memory-constrained  
**Accuracy**: High (90-95% recall)

```go
// Product quantization (Flat only)
db := vecgo.Flat[string](128).
    SquaredL2().
    PQ(8, 256).  // 8 subvectors, 256 centroids
    Build()
```

**Parameters**:
- `subvectors`: Number of splits (typically 8-16)
- `centroids`: Codebook size (typically 256)

**Memory reduction**:
- 128-dim float32 = 512 bytes
- PQ(8, 256) = 8 bytes (64x reduction)
- PQ(16, 256) = 16 bytes (32x reduction)

### Optimized Product Quantization (OPQ)

**Compression**: 8-32x (same as PQ)  
**Use case**: Better recall than PQ with same compression  
**Accuracy**: Very high (93-97% recall, 20-30% better than PQ)

```go
// OPQ for improved accuracy
db := vecgo.Flat[string](128).
    SquaredL2().
    OPQ(8, 256).  // Learned rotation + PQ
    Build()
```

**When to use**:
- You need high recall but low memory
- Acceptable to pay for rotation overhead

### Quantization Comparison

| Method | Compression | Recall | Speed | Use Case |
|--------|-------------|--------|-------|----------|
| **None** | 1x | 100% | Baseline | Default |
| **Binary** | 32x | 70-85% | Fastest | Pre-filtering |
| **PQ** | 8-64x | 90-95% | Fast | Memory-constrained |
| **OPQ** | 8-64x | 93-97% | Medium | High recall + low memory |

---

## DiskANN Configuration

DiskANN has several parameters that control memory/disk trade-offs:

### R (Graph Degree)

**What it does**: Number of neighbors per node in Vamana graph.

**Trade-offs**:
- ↑ R → ↑ Recall, ↑ Disk I/O, ↑ Memory
- ↓ R → ↓ Recall, ↓ Disk I/O, ↓ Memory

**Recommended values**:

| Use Case | R | Memory | Recall |
|----------|---|--------|--------|
| **Low memory** | 32 | Minimal | 85-90% |
| **Balanced** (default) | 64 | Moderate | 90-95% |
| **High recall** | 96-128 | High | 95-98% |

```go
db := vecgo.DiskANN[string]("./data", 128).
    SquaredL2().
    R(64).  // Balanced
    Build()
```

### L (Search List Size)

**What it does**: Beam search exploration depth.

**Trade-offs**:
- ↑ L → ↑ Recall, ↑ Search time, ↑ Disk I/O
- ↓ L → ↓ Recall, ↓ Search time, ↓ Disk I/O

**Recommended values**:

| Use Case | L | Search Speed | Recall |
|----------|---|--------------|--------|
| **Fast search** | 50-75 | Fastest | 85-90% |
| **Balanced** (default) | 100-150 | Fast | 90-95% |
| **High recall** | 150-300 | Slow | 95-98% |

```go
db := vecgo.DiskANN[string]("./data", 128).
    SquaredL2().
    L(100).  // Balanced
    Build()
```

### BeamWidth (Parallel Search Beams)

**What it does**: Number of parallel search paths.

**Trade-offs**:
- ↑ BeamWidth → ↑ Recall, ↑ Search time
- ↓ BeamWidth → ↓ Recall, ↓ Search time

**Recommended**: 2-8 (default 4)

```go
db := vecgo.DiskANN[string]("./data", 128).
    SquaredL2().
    BeamWidth(4).  // Default
    Build()
```

### Product Quantization (Memory Compression)

**What it does**: Compress vectors in memory using PQ codes.

**Recommended for**: Datasets >10M vectors

```go
db := vecgo.DiskANN[string]("./data", 128).
    SquaredL2().
    PQSubvectors(8).  // Compress to 8 bytes
    Build()
```

### Binary Prefilter (Traversal Prefilter)

**What it does**: Optional, **search-only** traversal prefilter using normalized Hamming distance on Binary Quantization codes.

**When it helps**: Reduce disk I/O / candidate expansions by cheaply ruling out obviously-wrong nodes.

```go
db := vecgo.DiskANN[string]("./data", 128).
    SquaredL2().
    PQSubvectors(8).
    BinaryPrefilter(0.25).
    Build()
```

**Parameter**:
- `BinaryPrefilter(maxNormalizedHamming float32)` where $0 \le maxNormalizedHamming \le 1$.

### Compaction (Reclaim Deleted Space)

**What it does**: Rebuild index to remove deleted vectors.

**Auto-compaction**:
```go
db := vecgo.DiskANN[string]("./data", 128).
    SquaredL2().
    EnableAutoCompaction(true).
    CompactionThreshold(0.2).  // Trigger at 20% deleted
    Build()
```

**Manual compaction**:
```go
db.Compact(ctx)  // Force compaction
```

### DiskANN Configuration Examples

```go
// Fast and memory-efficient
db := vecgo.DiskANN[string]("./data", 128).
    SquaredL2().
    R(32).
    L(50).
    BeamWidth(2).
    PQSubvectors(8).
    Build()

// Balanced (default)
db := vecgo.DiskANN[string]("./data", 128).
    SquaredL2().
    R(64).
    L(100).
    BeamWidth(4).
    EnableAutoCompaction(true).
    CompactionThreshold(0.2).
    Build()

// High recall
db := vecgo.DiskANN[string]("./data", 128).
    SquaredL2().
    R(96).
    L(200).
    BeamWidth(8).
    Build()
```

---

## WAL Durability Modes

The Write-Ahead Log (WAL) provides crash recovery with configurable durability trade-offs.

### Durability Modes

| Mode | fsync Frequency | Throughput | Durability | Data Loss Risk |
|------|-----------------|------------|------------|----------------|
| **Async** | Never (OS decides) | Highest (1.0x baseline) | Weakest | Seconds |
| **GroupCommit** | Batched (10ms or 100 ops) | High (0.83x vs Async) | Strong | ≤10ms |
| **Sync** | Every operation | Lowest (0.01x vs Async) | Strongest | None |

### Configuration

```go
// Async (fastest, weakest durability)
db, _ := vecgo.HNSW[string](128).
    SquaredL2().
    WAL("./wal", func(o *wal.Options) {
        o.DurabilityMode = wal.Async
    }).
    Build()

// GroupCommit (recommended: 83x faster than Sync)
db, _ := vecgo.HNSW[string](128).
    SquaredL2().
    WAL("./wal", func(o *wal.Options) {
        o.DurabilityMode = wal.GroupCommit
        o.GroupCommitInterval = 10 * time.Millisecond  // Max latency
        o.GroupCommitSize = 100                        // Batch size
    }).
    Build()

// Sync (slowest, strongest durability)
db, _ := vecgo.HNSW[string](128).
    SquaredL2().
    WAL("./wal", func(o *wal.Options) {
        o.DurabilityMode = wal.Sync
    }).
    Build()
```

### Auto-Checkpoint (Delta-Based Mmap Architecture)

WAL auto-checkpoint prevents unbounded log growth and enables the **delta-based mmap architecture** for optimal production performance:

```go
db, _ := vecgo.HNSW[string](128).
    SquaredL2().
    WAL("./wal", func(o *wal.Options) {
        o.AutoCheckpointOps = 10000                    // After 10K operations
        o.AutoCheckpointMB = 100                       // Or 100MB WAL size
    }).
    SnapshotPath("./data/snapshot.bin").              // Enable auto-save
    Build()
```

**How Delta-Based Mmap Works**:
1. **Writes**: Go to in-memory delta + WAL (1.4µs per insert)
2. **Auto-checkpoint**: When WAL thresholds hit, snapshot saves to `SnapshotPath`
3. **Restart**: Load from mmap snapshot (zero-copy, 760x faster) + replay WAL delta
4. **Reads**: Zero-copy mmap (3.3ms for 10K vectors, 2.9MB memory)

**Benefits**:
- **Fast writes**: In-memory delta operations
- **Instant startup**: Zero-copy mmap loading
- **Durability**: WAL ensures no data loss
- **Bounded memory**: OS page cache handles hot data

When either threshold is reached, Vecgo:
1. Writes a snapshot to the configured `SnapshotPath`
2. Deletes old WAL segments
3. Starts a fresh WAL

### Recommendations

| Use Case | Mode | Why |
|----------|------|-----|
| **Development** | Async | Fastest, data loss acceptable |
| **Production (default)** | GroupCommit | 83x faster than Sync, ≤10ms loss |
| **Financial/Critical** | Sync | Zero data loss |

---

## Memory Optimization

### Columnar Storage

Vecgo uses **Structure-of-Arrays (SOA)** layout for cache efficiency:

```
// Row-oriented (bad for SIMD):
vectors = [{x:1, y:2}, {x:3, y:4}, ...]

// Column-oriented (good for SIMD):
vectors = [1, 2, 3, 4, ...]  // Sequential access
```

**No configuration needed** — automatic in Flat and HNSW.

### Memory-Mapped Snapshots

Load snapshots without deserialization using zero-copy mmap:

```go
// Load from snapshot (zero-copy mmap)
db, err := vecgo.NewFromFile[string]("snapshot.bin")
if err != nil {
    log.Fatal(err)
}
defer db.Close()

// Use normally
results, _ := db.Search(query).KNN(10).Execute(ctx)
```

**Benefits**:
- **Instant startup** (<10ms): No deserialization, just pointer arithmetic.
- **OS page cache**: Hot parts of the graph stay in RAM; cold parts are swapped out.
- **Zero GC overhead**: The graph structure is off-heap.

### Soft Deletes + Compaction

Vecgo uses **soft deletes** (tombstones) to avoid reallocation:

```go
db.Delete(ctx, id)  // Marks as deleted, doesn't free memory
```

**Compaction** reclaims space:
```go
// Auto-compaction at 20% deleted
db := vecgo.HNSW[string](128).
    CompactionThreshold(0.2).
    Build()

// Manual compaction
db.Compact(ctx)
```

### Metadata Interning

Vecgo automatically uses **string interning** (via Go 1.24's `unique` package) for metadata keys and values. This significantly reduces memory usage when storing repetitive metadata (e.g., "category": "news", "status": "active").

**Benefits**:
- **Reduced Heap Usage**: Identical strings share the same underlying memory.
- **Zero Configuration**: Enabled automatically for all metadata.
- **Performance**: ~40% memory reduction for highly repetitive datasets.

**Benchmark Results (100k docs, 5 keys/doc)**:
- Naive Storage: 111 MB
- Vecgo Interning: 68 MB (**38% reduction**)

---

## Search Performance

### Batch Search

Amortize setup costs by searching for multiple queries:

```go
queries := [][]float32{query1, query2, query3}
results := db.BatchSearch(ctx, queries, 10)
```

### Streaming Search

Early termination for first-k results:

```go
for result, err := range db.Search(query).KNN(100).Stream(ctx) {
    if err != nil {
        log.Fatal(err)
    }
    if result.Distance > threshold {
        break  // Stop early
    }
    process(result)
}
```

### Metadata Filtering

Pre-compile filters for reuse:

```go
// Compile once
filter := metadata.And(
    metadata.Eq("category", "tech"),
    metadata.Gte("year", 2023),
)

// Reuse in multiple searches
results1 := db.Search(query1).KNN(10).Filter(filter).Execute(ctx)
results2 := db.Search(query2).KNN(10).Filter(filter).Execute(ctx)
```

---

## Benchmarking

### Running Benchmarks

```bash
# All benchmarks
go test -bench=. -benchmem ./...

# Specific index
go test -bench=BenchmarkHNSW -benchmem ./index/hnsw

# With profiling
go test -bench=. -cpuprofile=cpu.prof -memprofile=mem.prof
```

### Interpreting Results

```
BenchmarkHNSW/Insert-8         50000    25000 ns/op    1024 B/op    10 allocs/op
```

- `Insert-8`: 8 CPU cores used
- `50000`: Iterations run
- `25000 ns/op`: 25μs per operation
- `1024 B/op`: Bytes allocated per operation
- `10 allocs/op`: Allocations per operation

**Goals**:
- Search: 0 allocs/op (pooled buffers)
- Insert: <10 allocs/op
- Throughput: >100K ops/sec (HNSW on modern CPU)

### Custom Benchmarks

```go
func BenchmarkMyWorkload(b *testing.B) {
    db := vecgo.HNSW[string](128).SquaredL2().Build()
    query := randomVector(128)
    
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        db.KNNSearch(context.Background(), query, 10)
    }
}
```

---

## Summary

### Quick Reference

| Goal | Configuration |
|------|---------------|
| **Fastest search (small dataset)** | Flat index |
| **Balanced (medium dataset)** | HNSW with M=16, EF=200 |
| **Large dataset (>10M)** | DiskANN with R=64, L=100 |
| **Multi-core scaling** | .Shards(runtime.NumCPU()) |
| **Low memory** | PQ or OPQ quantization |
| **High recall** | HNSW with M=64, EF=400 |
| **Fast writes** | Sharding + GroupCommit WAL |
| **Zero data loss** | Sync WAL mode |

### Performance Checklist

- [ ] Choose correct index type (Flat/HNSW/DiskANN)
- [ ] Enable sharding for multi-core (HNSW/Flat only)
- [ ] Tune HNSW M/EF for accuracy/speed
- [ ] Enable GroupCommit WAL for durability
- [ ] Use quantization for memory reduction
- [ ] Enable auto-compaction for DiskANN
- [ ] Use mmap loading for fast startup
- [ ] Benchmark your specific workload

---

For architecture details, see [docs/architecture.md](architecture.md).
