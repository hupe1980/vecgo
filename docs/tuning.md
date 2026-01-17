# Vecgo Performance Tuning Guide

This guide covers performance tuning for Vecgo's tiered engine architecture.

## Quick Reference

```mermaid
flowchart LR
    subgraph Writes["✍️ Write Path"]
        Insert[Insert] --> MT[MemTable<br/>HNSW M=32]
        MT -->|Commit| Seg[Segment]
    end
    
    subgraph Reads["🔍 Read Path"]
        Query[Query] --> L0[L0 Search<br/>EF=200+]
        L0 --> Merge[Merge Results]
        SegSearch[Segment Search] --> Merge
    end
    
    subgraph Tuning["⚙️ Key Parameters"]
        M["M: Graph connectivity"]
        EF["EF: Search quality"]
        Cache["Cache: Memory budget"]
    end
    
    style Writes fill:#e8f5e9
    style Reads fill:#e3f2fd
    style Tuning fill:#fff3e0
```

## API Overview

```go
import (
    "context"
    "github.com/hupe1980/vecgo"
)

ctx := context.Background()

// Create a new index
db, err := vecgo.Open(ctx, vecgo.Local("./data"), vecgo.Create(128, vecgo.MetricL2))
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

## Insert Modes

Vecgo provides three insert modes optimized for different workloads:

```mermaid
flowchart LR
    subgraph Single["Single Insert"]
        S1[Insert] --> HNSW1[HNSW Graph<br/>Immediate]
    end
    
    subgraph Batch["Batch Insert"]
        B1[BatchInsert] --> HNSW2[HNSW Graph<br/>Immediate]
    end
    
    subgraph Deferred["Deferred Insert"]
        D1[BatchInsertDeferred] --> Store[Vector Store<br/>No Graph]
        Store -->|Commit| Disk[DiskANN<br/>After Flush]
    end
    
    style Single fill:#e8f5e9
    style Batch fill:#e3f2fd
    style Deferred fill:#fff3e0
```

### Performance Comparison (768 dimensions, Apple M4 Pro)

| Mode | Method | Throughput | Allocs | Searchable |
|------|--------|------------|--------|------------|
| **Single** | `Insert()` | 625 vec/s | 2 | ✅ Immediately |
| **Batch (10)** | `BatchInsert()` | 1,200 vec/s | 93 | ✅ Immediately |
| **Batch (100)** | `BatchInsert()` | 3,300 vec/s | 302 | ✅ Immediately |
| **Deferred (1K)** | `BatchInsertDeferred()` | 577,000 vec/s | 2,215 | ❌ After flush |
| **Deferred (10K)** | `BatchInsertDeferred()` | 2,030,000 vec/s | 20,534 | ❌ After flush |

### Single Insert

Use for **real-time applications** where vectors must be searchable immediately:

```go
// Each insert builds into the HNSW graph
id, err := db.Insert(ctx, vector, metadata, payload)
// Vector is searchable NOW
```

**Characteristics:**
- HNSW graph is updated immediately
- ~625 vectors/sec (768 dim)
- 2 allocations per insert
- Thread-safe (sharded HNSW with 16 shards)

### Batch Insert

Use for **medium batches** when you have multiple vectors and need immediate search:

```go
// Batch of vectors, all indexed into HNSW
ids, err := db.BatchInsert(ctx, vectors, metadatas, payloads)
// All vectors searchable NOW
```

**Characteristics:**
- HNSW graph is updated for each vector
- ~3,000 vectors/sec at batch=100
- Amortizes lock overhead
- Good for RAG pipelines with small document batches

### Deferred Insert (Bulk Loading)

Use for **bulk loading** when you're ingesting large datasets and don't need immediate search:

```go
// Vectors stored but NOT indexed into HNSW
ids, err := db.BatchInsertDeferred(ctx, vectors, metadatas, payloads)
// Vectors NOT searchable yet!

// Commit triggers flush - now vectors are in DiskANN segment
db.Commit(ctx)
// Vectors searchable via DiskANN
```

**Characteristics:**
- **~1000x faster** than indexed insert
- Skips HNSW graph construction entirely
- Vectors only stored in columnar format
- Searchable after flush creates DiskANN segment
- Ideal for initial data loading, migrations, nightly reindex

### Choosing the Right Mode

| Scenario | Recommended Mode | Why |
|----------|------------------|-----|
| Real-time RAG | `Insert()` | Immediate searchability required |
| Document ingestion batch | `BatchInsert()` | Balance of speed and immediate search |
| Initial corpus loading | `BatchInsertDeferred()` | 1000x faster, search can wait |
| Database migration | `BatchInsertDeferred()` | Bulk transfer, rebuild index once |
| Nightly embedding update | `BatchInsertDeferred()` | Batch window, rebuild overnight |
| Streaming embeddings | `Insert()` | Each item must be searchable |

---

## SIMD and CPU Features

Distance computation is SIMD-optimized with runtime CPU feature detection:

```mermaid
flowchart LR
    subgraph x86["x86-64"]
        AVX512[AVX-512<br/>Best]
        AVX2[AVX2<br/>Good]
    end
    
    subgraph ARM["ARM64"]
        SVE2[SVE2<br/>Best]
        NEON[NEON<br/>Good]
    end
    
    subgraph Fallback["Fallback"]
        Go[Pure Go<br/>-tags noasm]
    end
    
    style x86 fill:#e3f2fd
    style ARM fill:#e8f5e9
    style Fallback fill:#fff3e0
```

| Platform | Features | Speedup |
|----------|----------|---------|
| amd64 | AVX-512, AVX2 | 4-8× |
| arm64 | NEON, SVE2 | 3-6× |
| noasm | Pure Go | 1× (baseline) |

Build with `-tags noasm` to force the generic fallback (useful for debugging).

---

## HNSW Parameters (L0 MemTable)

The in-memory L0 layer uses [HNSW](https://arxiv.org/abs/1603.09320) for approximate nearest neighbor search.

| Parameter | Default | Description | Trade-off |
|-----------|---------|-------------|-----------|
| `M` | 32 | Max connections per node | ↑ recall, ↑ memory |
| `EF` | 300 | Construction queue size | ↑ quality, ↓ insert speed |
| `EFSearch` | max(k+100, 200) | Search queue size | ↑ recall, ↓ search speed |

> **Tip:** For high-recall workloads, increase `EFSearch`. For high-throughput insert, reduce `EF`.

---

## Compaction

```mermaid
flowchart LR
    C1[Commit] --> S1[Flat Segment]
    C2[Commit] --> S2[Flat Segment]
    C3[Commit] --> S3[Flat Segment]
    
    S1 & S2 & S3 -->|Merge| Comp[Compaction]
    
    Comp -->|Small| Flat[Flat Segment<br/>Exact Search]
    Comp -->|Large| DiskANN[DiskANN Segment<br/>Graph + PQ]
    
    style Comp fill:#fff3e0
    style Flat fill:#e8f5e9
    style DiskANN fill:#e3f2fd
```

Compaction runs in the background after commits:

1. **Commit** writes a new Flat segment to disk
2. **Compaction** merges segments based on size-tiered policy
3. **Output format** depends on row count:
   - Small: Flat segment (exact search, mmap)
   - Large: [DiskANN](https://papers.nips.cc/paper/2019/hash/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Abstract.html) segment (graph-based, quantized)

### Configuration

```go
db, err := vecgo.Open(
    ctx,
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
    ctx,
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
    ctx,
    vecgo.Local("./data"),
    vecgo.Create(128, vecgo.MetricL2),
    vecgo.WithMemoryLimit(0), // Unlimited (for bulk load)
)
```

---

## Quantization

Vecgo supports six quantization methods for memory reduction and faster distance computation.

```mermaid
flowchart LR
    subgraph Lossy["Lossy Compression"]
        BQ["Binary (BQ)<br/>32× compression<br/>70-85% recall"]
        RaBitQ["RaBitQ<br/>~30× compression<br/>80-90% recall"]
        PQ["Product (PQ)<br/>8-64× compression<br/>90-95% recall"]
        OPQ["Optimized PQ<br/>8-64× compression<br/>93-97% recall"]
    end
    
    subgraph Scalar["Scalar Compression"]
        SQ8["SQ8<br/>4× compression<br/>95-99% recall"]
        INT4["INT4<br/>8× compression<br/>90-95% recall"]
    end
    
    style BQ fill:#ffebee
    style RaBitQ fill:#fff3e0
    style SQ8 fill:#e8f5e9
    style INT4 fill:#e3f2fd
    style PQ fill:#f3e5f5
    style OPQ fill:#fce4ec
```

### Quantization Comparison

| Method | Compression | Recall | Speed | Best For |
|--------|-------------|--------|-------|----------|
| **None** | 1× | 100% | Baseline | Small datasets, exact search |
| **SQ8** | 4× | 95-99% | Fast | General purpose, balanced |
| **INT4** | 8× | 90-95% | Fast | Memory-constrained |
| **BQ** | 32× | 70-85% | Fastest | Pre-filtering, coarse search |
| **RaBitQ** | ~30× | 80-90% | Fast | Better BQ alternative |
| **PQ** | 8-64× | 90-95% | Medium | Large-scale, high compression |
| **OPQ** | 8-64× | 93-97% | Slower | Best recall with compression |

### When to Use Each Method

```mermaid
flowchart TD
    Start{Dataset Size?} -->|< 100K| None["Use: None<br/>100% recall"]
    Start -->|100K - 1M| SQ8["Use: SQ8<br/>4× smaller, ~99% recall"]
    Start -->|1M - 10M| PQ["Use: PQ/OPQ<br/>8-32× smaller"]
    Start -->|> 10M| DiskANN["Use: DiskANN + PQ<br/>Graph + compression"]
    
    SQ8 -->|Need more compression?| INT4["Try: INT4<br/>8× smaller"]
    PQ -->|Need better recall?| OPQ["Try: OPQ<br/>+5% recall"]
    
    style None fill:#e8f5e9
    style SQ8 fill:#e3f2fd
    style PQ fill:#f3e5f5
    style DiskANN fill:#fff3e0
```

### Configuration Examples

#### Scalar Quantization (SQ8)

Best for general-purpose use with minimal recall loss:

```go
db, err := vecgo.Open(ctx, vecgo.Local("./data"),
    vecgo.Create(128, vecgo.MetricL2),
    vecgo.WithQuantization(vecgo.QuantizationSQ8),
)
```

- **Compression**: 4× (float32 → uint8)
- **Training**: Per-dimension min/max normalization
- **Distance**: SIMD-optimized dequantized L2

#### Product Quantization (PQ)

Best for large-scale with high compression:

```go
db, err := vecgo.Open(ctx, vecgo.Local("./data"),
    vecgo.Create(128, vecgo.MetricL2),
    vecgo.WithQuantization(vecgo.QuantizationPQ),
    vecgo.WithPQSubvectors(8),     // M = 8 subvectors
    vecgo.WithPQCentroids(256),    // K = 256 centroids per subvector
)
```

- **Compression**: 64× with M=8, K=256 (512 bytes → 8 bytes)
- **Training**: K-means clustering per subvector
- **Distance**: ADC lookup tables (SIMD-optimized)

#### RaBitQ (Randomized Binary Quantization)

Best for ultra-fast pre-filtering:

```go
db, err := vecgo.Open(ctx, vecgo.Local("./data"),
    vecgo.Create(128, vecgo.MetricL2),
    vecgo.WithQuantization(vecgo.QuantizationRaBitQ),
)
```

- **Compression**: ~30× (stores norm for correction)
- **Distance**: Hamming + norm correction
- **Use case**: Coarse filtering before exact reranking

### Memory vs. Disk Trade-offs

Quantization reduces **RAM usage** for in-memory search. Full-precision vectors are still stored on disk for reranking.

**RAM Budget** (10M vectors, 1536 dimensions, compressed index in memory):

| Method | Per-Vector (RAM) | Total RAM | RAM Savings |
|--------|------------------|-----------|-------------|
| None (float32) | 6,144 B | 61.4 GB | - |
| SQ8 | 1,536 B | 15.4 GB | 75% |
| INT4 | 768 B | 7.7 GB | 87% |
| PQ (M=16) | 16 B | 160 MB | 99.7% |
| BQ | 192 B | 1.9 GB | 97% |

> **Important**: Disk usage is NOT reduced. Full vectors (61.4 GB) are always stored on disk for reranking. Add ~50% compaction headroom (~90 GB total disk).

**How it works**:
1. **Coarse search**: Use compressed vectors in RAM (fast Hamming/ADC distance)
2. **Reranking**: Fetch full-precision vectors from disk for top candidates
3. **Final result**: Exact distances from full vectors

This architecture trades disk I/O for RAM, enabling datasets larger than memory.

### Two-Stage Search (Reranking)

For best recall with compression, use two-stage search:

```go
// Stage 1: Coarse search with quantized vectors (fast, ~90% recall)
// Stage 2: Rerank top candidates with full vectors (100% recall on reranked set)

results, err := db.Search(ctx, query, 10,
    vecgo.WithOversample(3.0),  // Fetch 3× candidates for reranking
)
```

The oversample factor controls the recall/latency trade-off:
- `1.0`: No oversampling (quantized recall only)
- `2.0`: 2× candidates → ~95% recall
- `3.0`: 3× candidates → ~98% recall
- `5.0`: 5× candidates → ~99% recall

---

## Benchmarking

Run benchmarks with allocation tracking:

```bash
go test -bench=. -benchmem ./benchmark_test/
```

Search benchmarks report `recall@k` against an exact baseline.

---

## Query Debugging & Statistics

### QueryStats API

Use `WithStats()` to collect detailed execution metrics:

```go
var stats vecgo.QueryStats
results, err := db.Search(ctx, query, 10, vecgo.WithStats(&stats))

// Human-readable explanation
fmt.Println(stats.Explain())
// "searched 3 segments (1 pruned by stats, 2 by bloom), scanned 1200 vectors in 2.1ms"

// Programmatic access
fmt.Printf("Segments: searched=%d, pruned=%d\n", 
    stats.SegmentsSearched, 
    stats.SegmentsPrunedByStats + stats.SegmentsPrunedByBloom)
fmt.Printf("Vectors: scanned=%d, candidates=%d\n", 
    stats.VectorsScanned, stats.CandidatesRecalled)
fmt.Printf("Graph hops: %d, distance ops: %d (short-circuited: %d)\n",
    stats.GraphHops, stats.DistanceComputations, stats.DistanceShortCircuits)
```

### Performance Indicators

| Metric | Good | Warning | Action |
|--------|------|---------|--------|
| `SegmentsPrunedByStats` | >50% | <20% | Add categorical filters, use WithTimestamp for temporal data |
| `SegmentsPrunedByBloom` | >30% | 0% | Add equality filters on categorical fields |
| `DistanceShortCircuits` | >10% | 0% | Normal for filtered queries; indicates efficient pruning |
| `CandidatesRecalled / VectorsScanned` | >0.5 | <0.1 | Improve filter selectivity |

### Cost Estimation

```go
cost := stats.CostEstimate()
if cost > 1000 {
    log.Warn("Expensive query detected", "cost", cost, "latency", stats.Latency)
}
```

---

## Segment Pruning

Vecgo automatically prunes segments using manifest statistics. Understanding these can help optimize query performance.

### Triangle Inequality Pruning

For distance-based queries, segments where `dist(query, centroid) - Radius95 > maxDistance` are skipped entirely:

```
Query → Centroid distance: 0.8
Segment Radius95: 0.3
Current maxDistance: 0.4

Minimum possible distance to any vector: 0.8 - 0.3 = 0.5 > 0.4
→ Segment pruned (cannot improve results)
```

**Optimization**: Use `vecgo.WithMaxDistance(d)` to provide a tight upper bound early.

### Bloom Filter Pruning

Categorical equality filters (`category = "electronics"`) use Bloom filters for O(1) negative lookup:

```go
// Filter on high-cardinality field with equality
filter := metadata.NewFilterSet(
    metadata.Filter{Key: "product_id", Operator: metadata.OpEqual, Value: metadata.String("SKU-12345")},
)
// Bloom filter eliminates ~99% of segments that definitely don't contain this value
```

**Best Practices:**
- Use equality filters on categorical fields when possible
- High-cardinality fields (100+ distinct values) benefit most from Bloom filters
- Bloom filters have ~1% false positive rate (may search some segments unnecessarily)

### Numeric Range Stats

Each segment stores min/max per numeric field:

```go
filter := metadata.NewFilterSet(
    metadata.Filter{Key: "timestamp", Operator: metadata.OpGreaterThan, Value: metadata.Int(1704067200)},
)
// Segments with max(timestamp) < 1704067200 are skipped entirely
```

**Optimization**: Use temporal filters with `WithTimestamp()` for time-series data.

### Segment Ordering

Segments are searched in priority order, not ID order:

| Factor | Effect |
|--------|--------|
| **Centroid distance** | Closer segments searched first |
| **Cluster tightness** | Tight clusters have higher priority (more results per hop) |
| **Filter entropy** | Low-entropy segments searched first (better early termination) |
| **Tombstone ratio** | High-deletion segments deprioritized |

---

## Time-Travel Queries

Query historical database states without loading the latest version:

```go
// Open at a specific point in time
yesterday := time.Now().Add(-24 * time.Hour)
db, _ := vecgo.Open(ctx, vecgo.Local("./data"), vecgo.WithTimestamp(yesterday))

// Open at a specific version
db, _ := vecgo.Open(ctx, vecgo.Local("./data"), vecgo.WithVersion(42))

// Check current version
stats := db.Stats()
fmt.Printf("Opened at version %d\n", stats.ManifestID)
```

### Retention Policy

Control how many historical versions are preserved:

```go
policy := vecgo.RetentionPolicy{
    KeepVersions: 10,           // Keep last 10 versions
    KeepDuration: 7 * 24 * time.Hour, // Or keep versions from last 7 days
}
db, _ := vecgo.Open(ctx, vecgo.Local("./data"), vecgo.WithRetentionPolicy(policy))

// Clean up old versions
db.Vacuum(ctx)  // Removes versions beyond retention policy
```

### Storage Overhead

Time-travel preserves old segments until `Vacuum()` removes them:

```
┌─────────────────────────────────────────────────────────────┐
│ Storage = CurrentData × (1 + RetainedVersions × ChurnRate)  │
└─────────────────────────────────────────────────────────────┘

Example: 10GB data, 10 versions retained, 20% daily churn
  → Storage ≈ 10GB × (1 + 10 × 0.2) = 30GB
```

**Compaction still works!** It creates new optimized segments. Old segments are preserved for time-travel but deleted when `Vacuum()` removes the manifests referencing them.

| Retention Setting | Disk Overhead | Time-Travel Window |
|-------------------|---------------|--------------------|
| `KeepVersions: 1` | ~1.5× | Current only |
| `KeepVersions: 10` | ~2-3× | ~10 commits |
| `KeepDuration: 7d` | ~2-4× | 7 days |
| `KeepVersions: 0` | ~1× | None (disable time-travel) |

**Best Practice**: Run `Vacuum()` periodically (e.g., daily cron) to reclaim space from expired versions.

### Use Cases

| Scenario | API |
|----------|-----|
| Debug yesterday's recall issue | `WithTimestamp(yesterday)` |
| A/B test against version 42 | `WithVersion(42)` |
| Audit data at compliance checkpoint | `WithTimestamp(auditDate)` |
| Roll back after bad deployment | `WithVersion(knownGoodVersion)` |
