# 🧬🔍 Vecgo

![Build Status](https://github.com/hupe1980/vecgo/workflows/build/badge.svg)
[![Go Reference](https://pkg.go.dev/badge/github.com/hupe1980/vecgo.svg)](https://pkg.go.dev/github.com/hupe1980/vecgo)
[![goreportcard](https://goreportcard.com/badge/github.com/hupe1980/vecgo)](https://goreportcard.com/report/github.com/hupe1980/vecgo)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**Vecgo** is a high-performance vector database library for Go, designed for production workloads requiring billion-scale similarity search with typed metadata filtering and enterprise-grade durability.

## ⚡ Key Features

### 🎯 Index Types
- **Flat**: Exact nearest-neighbor search with SIMD-optimized distance computation
- **HNSW**: Graph-based approximate search with sub-millisecond latency
- **DiskANN**: Billion-scale disk-resident search with Vamana algorithm

### 🗄️ Enterprise Features
- **Full CRUD**: Insert, update, delete, and batch operations
- **Native Pre-Filtering**: Filter during graph traversal for 100% recall (vs ~50% post-filtering)
- **Hybrid Search**: Combine vector similarity with attribute filters
- **Streaming Results**: Iterator-based search with early termination
- **Atomic Persistence**: Binary snapshots with atomic writes (temp file + rename)
- **Write-Ahead Log**: Crash recovery with group commit (83x faster than sync)
- **Auto-Compaction**: Background cleanup of deleted vectors
- **Production-Ready**: Zero goroutine leaks, comprehensive error propagation, idempotent Close()
- **Metadata Safety**: Deep copy on insert prevents silent mutation bugs
- **DiskANN Crash Safety**: All index files written atomically

### 🚀 Performance
- **SIMD Kernels**: AVX/AVX512 (x86_64) and NEON (ARM64) acceleration
- **Zero-Allocation Search**: Pooled buffers eliminate GC pressure
- **Columnar Storage**: SOA layout for optimal cache locality
- **Sharded Writes**: 2.7-3.4x speedup with parallel write coordinators
- **Worker Pool**: Fixed goroutine pool for sharded searches (0 goroutines created per search)
- **Lock-Free Reads**: Concurrent search without contention
- **Smart Filtering**: Pre-filtering reduces distance computations by ~40% for filtered searches

### 🗜️ Compression
- **Binary Quantization**: 32x compression (0.68ns/op for 128-dim Hamming)
- **Scalar Quantization**: 4x compression with 8-bit encoding
- **Product Quantization**: Learned codebooks for 8-64x compression
- **Optimized PQ (OPQ)**: 20-30% better reconstruction vs standard PQ

### 📊 Observability
- **Structured Logging**: `log/slog` integration with contextual attributes
- **Metrics**: Prometheus-compatible instrumentation
- **Production-Ready**: Built-in monitoring and alerting support

## 🚀 Quick Start

### Basic Usage

```go
import "github.com/hupe1980/vecgo"

// Create an HNSW index
db, err := vecgo.HNSW[string](128).  // 128-dimensional vectors
    SquaredL2().                      // Distance function
    M(16).                            // Graph connectivity
    EFConstruction(200).              // Build-time search quality
    Build()
if err != nil {
    log.Fatal(err)
}
defer db.Close()

// Insert vectors
id, _ := db.Insert(ctx, vecgo.VectorWithData[string]{
    Vector: []float32{1.0, 2.0, 3.0, /* ... */},
    Data:   "my document",
})

// Search
results, _ := db.Search(query).KNN(10).Execute(ctx)

// Streaming search with early termination
for result, err := range db.Search(query).KNN(100).Stream(ctx) {
    if err != nil {
        log.Fatal(err)
    }
    if result.Distance > threshold {
        break // Stop early
    }
    process(result)
}
```

### Index Types

**Flat (Exact Search)**
```go
db, err := vecgo.Flat[string](128).
    Cosine().
    Build()
```

**HNSW (Approximate Search)**
```go
db, err := vecgo.HNSW[string](128).
    SquaredL2().
    M(32).               // Graph degree (16-64)
    EFConstruction(200). // Build quality (100-400)
    Shards(4).           // Parallel writes (2-8 cores)
    Build()
```

**DiskANN (Billion-Scale)**
```go
db, err := vecgo.DiskANN[string]("./data", 128).
    SquaredL2().
    R(64).               // Graph degree
    L(100).              // Build list size
    BeamWidth(4).        // Search width
    Build()
```

## 🔍 Advanced Features

### Metadata Filtering

**Native pre-filtering** achieves 100% recall by filtering during graph traversal (not after):

```go
import "github.com/hupe1980/vecgo/metadata"

// Insert with metadata
db.Insert(ctx, vecgo.VectorWithData[string]{
    Vector: []float32{1, 2, 3},
    Data:   "Document 1",
    Metadata: metadata.Metadata{
        "category": metadata.String("technology"),
        "year":     metadata.Int(2024),
    },
})

// Hybrid search (vector + metadata)
results, _ := db.Search(query).
    KNN(10).
    Filter(metadata.Eq("category", "technology")).
    Filter(metadata.Gt("year", 2020)).
    Execute(ctx)
```

**Performance**: Pre-filtering is ~30% faster than post-filtering and guarantees k results when available.

### Streaming Search

```go
// Early termination
for result, err := range db.Search(query).KNN(1000).Stream(ctx) {
    if err != nil {
        log.Fatal(err)
    }
    if result.Distance > threshold {
        break
    }
    process(result)
}

// Find first match
first, _ := db.Search(query).KNN(100).First(ctx)
```

### Persistence

```go
// Save snapshot
db.SaveToFile("vectors.bin")

// Load snapshot (zero-copy mmap)
db, _ := vecgo.NewFromFile[string]("vectors.bin")
defer db.Close()
```

### Write-Ahead Log (Durability)

**Three Durability Modes**:
- **DurabilityAsync**: No fsync (~7µs/op) - fastest, data loss risk
- **DurabilityGroupCommit**: Batched fsync (~48µs/op) - **83x faster than sync**
- **DurabilitySync**: Per-op fsync (~4ms/op) - maximum durability

```go
import "github.com/hupe1980/vecgo/wal"

// Enable WAL with group commit (recommended)
db, _ := vecgo.HNSW[string](128).
    SquaredL2().
    WAL("./data", func(o *wal.Options) {
        o.DurabilityMode = wal.DurabilityGroupCommit
        o.GroupCommitInterval = 10 * time.Millisecond
        o.GroupCommitMaxOps = 100
        o.Compress = true // zstd compression
    }).
    Build()

// Crash recovery
db, _ := vecgo.NewFromFile[string]("vectors.bin", 
    vecgo.WithWAL("./data", func(o *wal.Options) {
        o.Compress = true
    }))
db.RecoverFromWAL(context.Background())
```

### Input Validation (Production Safety)

**Automatic validation** prevents crashes and DoS attacks:

```go
import "github.com/hupe1980/vecgo/engine"

// Validation is enabled by default with safe limits
db, _ := vecgo.HNSW[string](128).Build()

// Customize validation limits
db, _ := vecgo.HNSW[string](128).
    Options(vecgo.WithValidationLimits(&engine.ValidationLimits{
        MaxDimension:     1024,         // Max vector dimension
        MaxVectors:       10_000_000,   // Max total vectors
        MaxK:             1000,          // Max search results
        MaxMetadataBytes: 16 * 1024,    // Max 16KB metadata per vector
        MaxBatchSize:     5000,          // Max batch insert size
    })).
    Build()

// Disable for pre-validated inputs (not recommended)
db, _ := vecgo.HNSW[string](128).
    Options(vecgo.WithoutValidation()).
    Build()
```

**Protected operations**:
- ✅ Nil vector detection
- ✅ Dimension mismatch
- ✅ NaN/Inf value checks
- ✅ Metadata size limits
- ✅ Batch size limits
- ✅ Total vector count limits

## 🗜️ Quantization

### Binary Quantization (32x Compression)

```go
import "github.com/hupe1980/vecgo/quantization"

bq := quantization.NewBinaryQuantizer(128)
bq.Train(trainingVectors)

encoded := bq.EncodeUint64(vector)
distance := quantization.HammingDistance(encoded1, encoded2)
```

**Performance**: 0.68ns/op for 128-dim Hamming distance

### Product Quantization

```go
// Standard PQ (Flat index only)
db.EnableProductQuantization(index.ProductQuantizationConfig{
    NumSubvectors: 8,
    NumCentroids:  256,
})

// Optimized PQ (20-30% better reconstruction)
opq, _ := quantization.NewOptimizedProductQuantizer(
    dimension, 8, 256, 10, // dim, M, K, iterations
)
opq.Train(trainingVectors)
codes := opq.Encode(vector)
distance := opq.ComputeAsymmetricDistance(query, codes)
```

## 📦 Storage Architecture

Vecgo uses a **columnar SOA (Structure-of-Arrays)** layout for optimal SIMD performance:

```go
import "github.com/hupe1980/vecgo/vectorstore/columnar"

store := columnar.New(128)

// Append and access
id, _ := store.Append(vec)
vec, ok := store.GetVector(id)

// Soft deletes with compaction
store.DeleteVector(id)
idMap := store.Compact()

// Persistence
store.WriteTo(file)
store.ReadFrom(file)

// Zero-copy mmap
mmapStore, closer, _ := columnar.OpenMmap("vectors.col")
defer closer.Close()
```

**File Format**: 64-byte header + contiguous vectors + delete bitmap + CRC32 checksum

## 🌐 DiskANN - Billion-Scale Search

DiskANN provides disk-resident approximate search using Microsoft Research's Vamana algorithm with full CRUD support.

```go
db, _ := vecgo.DiskANN[string]("./data", 128).
    SquaredL2().
    R(64).               // Graph degree
    L(100).              // Build list size
    BeamWidth(4).        // Search width
    EnableAutoCompaction(true).
    CompactionThreshold(0.2).
    Build()

// Full Vecgo API support
db.Insert(ctx, vecgo.VectorWithData[string]{...})
db.Search(query).KNN(10).Execute(ctx)
db.Delete(ctx, id)
```

**Architecture**:
- **RAM**: Vamana graph + PQ codebooks
- **SSD**: Memory-mapped full vectors for reranking

**Offline Bulk Loading**:
```go
import "github.com/hupe1980/vecgo/index/diskann"

builder, _ := diskann.NewBuilder(128, index.DistanceTypeSquaredL2, "./index", nil)
for _, vec := range vectors {
    builder.Add(vec)
}
builder.Build(ctx)

idx, _ := diskann.Open("./index", nil)
results, _ := idx.KNNSearch(ctx, query, 10, nil)
```

**When to Use**: Datasets > 10M vectors, cost-optimized deployments, billion-scale requirements

## 📊 Observability

### Structured Logging

```go
import "log/slog"

// JSON logging
logger := vecgo.NewJSONLogger(slog.LevelInfo)
db, _ := vecgo.HNSW[string](128).
    SquaredL2().
    Logger(logger).
    Build()

// Debug logging
db, _ := vecgo.HNSW[string](128).
    SquaredL2().
    LogLevel(slog.LevelDebug).
    Build()
```

### Metrics Collection

```go
metrics := &vecgo.BasicMetricsCollector{}
db, _ := vecgo.HNSW[string](128).
    SquaredL2().
    MetricsCollector(metrics).
    Build()

stats := metrics.GetStats()
fmt.Printf("Searches: %d, Avg latency: %dns\n", 
    stats.SearchCount, stats.SearchAvgNanos)
```

**Prometheus Integration**: Implement `MetricsCollector` for custom exporters.

## 🧪 Examples

See [`_examples/`](./_examples/) for complete working code:
- [Flat index](./_examples/flat/main.go)
- [HNSW index](./_examples/hnsw/main.go)
- [DiskANN index](./_examples/diskann/main.go)
- [Persistence](./_examples/persistence/main.go)
- [Quantization](./_examples/quantization/main.go)
- [Metrics & logging](./_examples/metrics/main.go)
- [Streaming search](./_examples/streaming/main.go)

## ��️ Architecture

```
┌────────────────────┐
│    Vecgo[T] API     │  Typed public interface
├────────────────────┤
│  Coordinator/Tx     │  Coordinated mutations with rollback
├────────────────────┤
│  Index Layer        │  Flat / HNSW / DiskANN
├────────────────────┤
│  Metadata Index     │  Roaring Bitmap inverted index
├────────────────────┤
│  Vector Storage     │  Columnar SOA with soft deletes
├────────────────────┤
│  Persistence        │  Snapshots + zero-copy mmap
├────────────────────┤
│  Durability (WAL)   │  Crash recovery + group commit
├────────────────────┤
│  Quantization       │  Binary / Scalar / PQ / OPQ
├────────────────────┤
│  SIMD Kernels       │  AVX/AVX512/NEON distance functions
└────────────────────┘
```

## 🎯 Performance Tips

**Index Selection**:
- **Flat**: < 100K vectors, exact results required
- **HNSW**: 100K-10M vectors, sub-millisecond latency
- **DiskANN**: > 10M vectors, cost-optimized deployments

**HNSW Tuning**:
- `M=16-32`: Balanced (32 for high recall)
- `EF=100-400`: Higher = better quality, slower build
- `Shards=2-8`: Multi-core write scaling

**DiskANN Tuning**:
- `R=32-128`: Graph degree (higher = better recall)
- `L=75-200`: Build quality (higher = slower build)
- `BeamWidth=2-8`: Search paths (higher = better recall)

**Distance Functions**:
- **SquaredL2**: General-purpose, no normalization
- **Cosine**: Normalized vectors (auto-normalized on insert/search)
- **DotProduct**: Maximum inner product (negated for distance)

## 📚 Documentation

- **API Reference**: [pkg.go.dev/github.com/hupe1980/vecgo](https://pkg.go.dev/github.com/hupe1980/vecgo)
- **Architecture**: [docs/architecture.md](docs/architecture.md)
- **Performance Tuning**: [docs/tuning.md](docs/tuning.md)

## 🤝 Contributing

Contributions welcome! Please open an issue or pull request.

## 📄 License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
