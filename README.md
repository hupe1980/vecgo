# 🧬🔍 Vecgo

![Build Status](https://github.com/hupe1980/vecgo/workflows/build/badge.svg)
[![Go Reference](https://pkg.go.dev/badge/github.com/hupe1980/vecgo.svg)](https://pkg.go.dev/github.com/hupe1980/vecgo)
[![goreportcard](https://goreportcard.com/badge/github.com/hupe1980/vecgo)](https://goreportcard.com/report/github.com/hupe1980/vecgo)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**Vecgo** is a **pure Go, embeddable, hybrid vector database** designed for high-performance production workloads.

It occupies a unique niche:
- **Faster & lighter than external services** (no network overhead, no sidecar).
- **More capable than simple libraries** (provides durability, concurrency, and hybrid search).
- **Simpler than CGO wrappers** (pure Go toolchain, static binaries, cross-compilation).

⚠️ This is experimental and subject to breaking changes.

## ⚡ Key Features

### 🎯 Index Types
- **Flat**: Exact nearest-neighbor search with SIMD-optimized distance computation
- **DiskANN**: Disk-resident ANN segments built by compaction (Vamana-style)
- **HNSW**: Segment implementation exists, but is not yet wired into the current tiered engine

### 🗄️ Enterprise Features
- **Full CRUD**: Insert, update, delete, and batch operations
- **Typed Metadata Filtering**: Metadata filters are supported in the tiered engine (MemTable + Flat segments). (DiskANN metadata filtering is currently limited.)
- **Hybrid Search**: Combine vector similarity with attribute filters
- **Atomic Persistence**: Binary snapshots with atomic writes (temp file + rename)
- **Write-Ahead Log**: Crash recovery with group commit
- **Auto-Compaction**: Background cleanup of deleted vectors
- **Async Indexing**: LSM-style write path (WAL + MemTable) for sub-millisecond write latency
- **Production-Ready**: Zero goroutine leaks, comprehensive error propagation, idempotent Close()
- **Metadata Safety**: Deep copy on insert prevents silent mutation bugs. Uses Go 1.24 `unique` package for efficient string interning.
- **DiskANN Crash Safety**: All index files written atomically

### 🚀 Performance
- **SIMD Kernels**: AVX/AVX512 (x86_64) and NEON (ARM64) acceleration
- **Instant Startup**: Mmap-able HNSW and DiskANN for <10ms load times
- **Zero-Allocation Search**: Generation-based visited sets and pooled buffers eliminate GC pressure
- **Columnar Storage**: SOA layout for optimal cache locality
- **Shared-Nothing Architecture**: Linear write scaling with independent shards and WALs
- **Worker Pool**: Fixed goroutine pool for sharded searches (0 goroutines created per search)
- **Lock-Free Reads**: Concurrent search without contention (Atomic Pointer Swapping)
- **Cache-Optimized**: Cached vector access in heuristic search
- **Smart Filtering**: Pre-filtering reduces distance computations by ~40% for filtered searches

### 🗜️ Compression
- **Binary Quantization**: 32x compression (0.68ns/op for 128-dim Hamming)
- **Scalar Quantization**: 4x compression with 8-bit encoding
- **Product Quantization**: Learned codebooks for 8-64x compression
- **Optimized PQ (OPQ)**: 20-30% better reconstruction vs standard PQ

> Note: DiskANN can optionally use Binary Quantization as a **search-only traversal prefilter** via `BinaryPrefilter(...)` (it does not replace PQ traversal or float32 reranking).

### 📊 Observability
- **Structured Logging**: `log/slog` integration with contextual attributes
- **Metrics**: Prometheus-compatible instrumentation
- **Production-Ready**: Built-in monitoring and alerting support

## 🚀 Quick Start

### Basic Usage

```go
import "github.com/hupe1980/vecgo"

// Current public API (Jan 2026): open the tiered engine directly.
db, err := vecgo.Open("./data", 128, vecgo.MetricL2)
if err != nil {
	log.Fatal(err)
}
defer db.Close()

// Insert vectors
if err := db.Insert(1, make([]float32, 128)); err != nil {
	log.Fatal(err)
}

// Persist the active MemTable into an immutable Flat segment.
if err := db.Flush(); err != nil {
	log.Fatal(err)
}

// Search
results, err := db.Search(ctx, make([]float32, 128), 10, vecgo.WithRefineFactor(2.0))
if err != nil {
	log.Fatal(err)
}
_ = results
```

For tuning and current segment integration status, see `docs/tuning.md`.

## 📚 Documentation

- **API Reference**: [pkg.go.dev/github.com/hupe1980/vecgo](https://pkg.go.dev/github.com/hupe1980/vecgo)
- **Architecture**: [docs/architecture.md](docs/architecture.md)
- **Performance Tuning**: [docs/tuning.md](docs/tuning.md)

## 🤝 Contributing

Contributions welcome! Please open an issue or pull request.

## 📄 License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
