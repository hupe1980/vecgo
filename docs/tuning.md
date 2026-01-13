# Vecgo Performance Tuning Guide

This guide covers performance tuning for Vecgo's tiered engine architecture.

## API Overview

```go
import (
    "context"
    "github.com/hupe1980/vecgo"
)

ctx := context.Background()

// Create a new index
db, err := vecgo.Open(vecgo.Local("./data"), vecgo.Create(128, vecgo.MetricL2))
if err != nil {
    panic(err)
}
defer db.Close()

// Insert vectors
_, _ = db.Insert(ctx, make([]float32, 128), nil, nil)

// Commit to disk (durability point)
if err := db.Commit(ctx); err != nil {
    panic(err)
}

// Search
results, _ := db.Search(ctx, make([]float32, 128), 10)
```

The engine supports **automatic flush triggers** based on MemTable size. However, call `Commit(ctx)` explicitly at batch boundaries for deterministic persistence.

---

## SIMD and CPU Features

Distance computation is SIMD-optimized with runtime CPU feature detection:

| Platform | Features |
|----------|----------|
| amd64 | AVX-512, AVX2, SSE4.2 |
| arm64 | NEON, SVE2 |

Build with `-tags noasm` to force the generic fallback (useful for debugging).

---

## HNSW Parameters (L0 MemTable)

The in-memory L0 layer uses [HNSW](https://arxiv.org/abs/1603.09320) for approximate nearest neighbor search.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `M` | 32 | Max connections per node. Higher = better recall, more memory |
| `EF` | 300 | Construction queue size. Higher = better graph quality, slower insert |
| `EFSearch` | max(k+100, 200) | Search queue size. Higher = better recall, slower search |

---

## Compaction

Compaction runs in the background after commits:

1. **Commit** writes a new Flat segment to disk
2. **Compaction** merges segments based on size-tiered policy
3. **Output format** depends on row count:
   - Small: Flat segment (exact search, mmap)
   - Large: [DiskANN](https://papers.nips.cc/paper/2019/hash/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Abstract.html) segment (graph-based, quantized)

### Configuration

```go
db, err := vecgo.Open(
    vecgo.Local("./data"),
    vecgo.Create(128, vecgo.MetricL2),
    vecgo.WithCompactionThreshold(8), // Trigger after 8 segments
)
```

---

## Caching

### Local Storage (mmap)

For local disk with mmap segments, the OS page cache handles caching. Focus on:
- Contiguous column access patterns
- Minimizing random seeks

### Remote Storage (S3/GCS)

For cloud storage, configure the block cache:

```go
db, err := vecgo.Open(
    vecgo.Remote(s3Store),
    vecgo.WithCacheDir("/fast/nvme"),           // Disk cache location
    vecgo.WithBlockCacheSize(64 * 1024 * 1024), // 64MB memory cache
)
```

---

## Segment Types

| Segment | Storage | Use Case |
|---------|---------|----------|
| **MemTable** | Memory | L0 hot tier (HNSW index) |
| **Flat** | Disk (mmap) | Small segments, exact search |
| **DiskANN** | Disk | Large segments, graph-based ANN |

---

## Durability Model

Vecgo uses **commit-oriented durability** — no Write-Ahead Log (WAL).

```go
// Insert operations accumulate in memory
db.Insert(ctx, vec1, nil, nil)
db.Insert(ctx, vec2, nil, nil)

// Commit writes immutable segment + updates manifest atomically
err := db.Commit(ctx)
// Data is durable after Commit() returns
```

**Why no WAL?** Vector workloads are batch-oriented:
- RAG pipelines: embed → insert batch → commit
- ML training: checkpoint after epoch
- Search: build offline → serve queries

**Pattern: Batch insert with explicit commit**

```go
for _, batch := range batches {
    for _, item := range batch {
        db.Insert(ctx, item.Vector, item.Metadata, item.Payload)
    }
    db.Commit(ctx) // Data is now durable
}
```

---

## Memory Management

### Arena Allocator

HNSW uses a custom arena allocator for stable heap and reduced GC pressure:
- 4MB chunk size (fewer syscalls)
- Off-heap allocation via mmap
- Zero-allocation search path

### Resource Limits

```go
db, err := vecgo.Open(
    vecgo.Local("./data"),
    vecgo.Create(128, vecgo.MetricL2),
    vecgo.WithMemoryLimit(0), // Unlimited (for bulk load)
)
```

---

## Quantization

Configure quantization for DiskANN segments:

| Method | Compression | Accuracy | Use Case |
|--------|-------------|----------|----------|
| None | 1× | 100% | Small datasets |
| SQ8 | 4× | ~99% | Balanced |
| PQ | 8-64× | ~95% | Large scale |
| INT4 | 8× | ~97% | Memory constrained |

---

## Benchmarking

Run benchmarks with allocation tracking:

```bash
go test -bench=. -benchmem ./benchmark_test/
```

Search benchmarks report `recall@k` against an exact baseline.
