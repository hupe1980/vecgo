---
layout: default
title: Configuration
parent: Guides
nav_order: 1
---

# Configuration

Complete reference for all Vecgo configuration options.
{: .fs-6 .fw-300 }

---

## Opening a Database

```go
db, err := vecgo.Open(ctx, backend, options...)
```

### Backends

```go
// Local filesystem
backend := vecgo.NewLocalBackend("./data")

// In-memory (testing)
backend := vecgo.NewMemoryBackend()

// S3
backend := vecgo.NewS3Backend(s3Client, "bucket", "prefix")

// MinIO
backend := vecgo.NewMinIOBackend(minioClient, "bucket", "prefix")
```

---

## Core Options

### Dimension

**Required.** The dimensionality of vectors.

```go
vecgo.WithDimension(1536)
```

{: .warning }
All vectors must have exactly this dimension. Cannot be changed after creation.

### Distance Metric

```go
vecgo.WithDistance(vecgo.DistanceCosine)  // Default
vecgo.WithDistance(vecgo.DistanceL2)
vecgo.WithDistance(vecgo.DistanceDot)
```

| Metric | Best For | Range |
|:-------|:---------|:------|
| Cosine | Normalized embeddings | [-1, 1] |
| L2 (Euclidean) | Spatial data | [0, ∞) |
| Dot Product | MIPS | (-∞, ∞) |

---

## Index Configuration

### Index Type

```go
vecgo.WithIndexType(vecgo.IndexHNSW)     // Default
vecgo.WithIndexType(vecgo.IndexIVFFlat)
vecgo.WithIndexType(vecgo.IndexFlat)
```

### HNSW Options

```go
vecgo.WithHNSWConfig(vecgo.HNSWConfig{
    M:              16,   // Connections per node (default: 16)
    EfConstruction: 200,  // Build quality (default: 200)
    EfSearch:       100,  // Search quality (default: 100)
})
```

| Parameter | Effect | Tradeoff |
|:----------|:-------|:---------|
| `M` | Graph connectivity | Memory ↔ Recall |
| `EfConstruction` | Build quality | Build time ↔ Recall |
| `EfSearch` | Search quality | Latency ↔ Recall |

{: .tip }
Start with defaults. Increase `EfSearch` if recall is too low.

### IVF Options

```go
vecgo.WithIVFConfig(vecgo.IVFConfig{
    NClusters: 1024,  // Number of clusters
    NProbe:    32,    // Clusters to search
})
```

| Parameter | Effect | Guideline |
|:----------|:-------|:----------|
| `NClusters` | Partitioning | √n to 4√n vectors |
| `NProbe` | Search breadth | 1-10% of clusters |

---

## Quantization

### Product Quantization (PQ)

```go
vecgo.WithQuantization(vecgo.QuantizationPQ)
vecgo.WithPQConfig(vecgo.PQConfig{
    M:         8,     // Subvectors (dimension must be divisible)
    Bits:      8,     // Bits per code (4 or 8)
    TrainSize: 10000, // Vectors for training
})
```

**Memory reduction:** `dim × 4 bytes` → `M bytes`

### Scalar Quantization (SQ8)

```go
vecgo.WithQuantization(vecgo.QuantizationSQ8)
```

**Memory reduction:** 4x (float32 → uint8)

### INT4 Quantization

```go
vecgo.WithQuantization(vecgo.QuantizationINT4)
```

**Memory reduction:** 8x (float32 → 4-bit)

---

## Storage Options

### MemTable Size

```go
vecgo.WithMemTableSize(64 * 1024 * 1024)  // 64 MB (default)
```

Larger = fewer flushes, more memory usage.

### Block Cache

```go
vecgo.WithBlockCacheSize(256 * 1024 * 1024)  // 256 MB
```

Caches frequently accessed data blocks.

### Write Buffer

```go
vecgo.WithWriteBufferSize(4 * 1024 * 1024)  // 4 MB
```

Buffers writes before committing.

---

## Durability Options

### Auto-Commit

```go
vecgo.WithAutoCommit(true)
vecgo.WithAutoCommitInterval(time.Second)
```

Automatically commits at specified interval.

### Sync Mode

```go
vecgo.WithSyncWrites(true)  // fsync after each write (slower, safer)
```

### WAL Options

```go
vecgo.WithWALEnabled(true)   // Default: true
vecgo.WithWALSyncMode(vecgo.WALSyncBatch)  // Batch syncs
```

---

## Concurrency

### Read Parallelism

```go
vecgo.WithMaxConcurrentReads(8)
```

### Write Parallelism

```go
vecgo.WithMaxConcurrentWrites(2)
```

### Background Workers

```go
vecgo.WithCompactionWorkers(2)
vecgo.WithFlushWorkers(1)
```

---

## Search Options

Per-query options passed to `Search()`:

```go
results, err := db.Search(ctx, query, k,
    vecgo.WithFilter(filter),           // Metadata filter
    vecgo.WithEfSearch(200),            // Override HNSW ef
    vecgo.WithNProbe(64),               // Override IVF nprobe
    vecgo.WithIncludeMetadata(true),    // Return metadata
    vecgo.WithIncludeVectors(false),    // Don't return vectors
    vecgo.WithMinScore(0.7),            // Minimum similarity
)
```

---

## Environment Variables

```bash
# SIMD override
VECGO_SIMD=avx2      # Force specific instruction set
VECGO_SIMD=none      # Disable SIMD

# Debug
VECGO_DEBUG=1        # Enable debug logging
```

---

## Full Example

```go
db, err := vecgo.Open(ctx, vecgo.NewLocalBackend("./production-db"),
    // Core
    vecgo.WithDimension(1536),
    vecgo.WithDistance(vecgo.DistanceCosine),
    
    // Index
    vecgo.WithIndexType(vecgo.IndexHNSW),
    vecgo.WithHNSWConfig(vecgo.HNSWConfig{
        M:              32,
        EfConstruction: 400,
        EfSearch:       200,
    }),
    
    // Quantization
    vecgo.WithQuantization(vecgo.QuantizationSQ8),
    
    // Storage
    vecgo.WithMemTableSize(128 * 1024 * 1024),
    vecgo.WithBlockCacheSize(512 * 1024 * 1024),
    
    // Durability
    vecgo.WithAutoCommit(true),
    vecgo.WithAutoCommitInterval(5 * time.Second),
    
    // Concurrency
    vecgo.WithMaxConcurrentReads(16),
)
```
