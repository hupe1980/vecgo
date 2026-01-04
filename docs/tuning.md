# Vecgo Performance Tuning Guide

This guide describes tuning for the **current public API** (Jan 2026) in this repo:

- Public entry point: `vecgo.Open(dir, dim, metric, ...opts)`.
- Storage engine: a tiered engine with an in-memory L0 MemTable and immutable on-disk segments.
- Segment types currently integrated: Flat (exact, mmap) and DiskANN (built by compaction above a size threshold).

If you’re looking for builder-style APIs (e.g. `vecgo.HNSW[T]().Build()`), those examples refer to the legacy surface and are not part of the current facade.

---

## Quick Start (Current API)

```go
db, err := vecgo.Open("./data", 128, vecgo.MetricL2)
if err != nil {
	panic(err)
}
defer db.Close()

// Ingest
_ = db.Insert(1, make([]float32, 128))

// Persist the active MemTable into an immutable Flat segment.
if err := db.Flush(); err != nil {
	panic(err)
}

// Query
results, _ := db.Search(ctx, make([]float32, 128), 10)
_ = results
```

Operational note: the engine supports **automatic flush triggers** via `engine.FlushConfig` (MemTable size / WAL size), and applies backpressure via `resource.Controller` when configured budgets are reached. You can still call `Flush()` explicitly (e.g. at batch boundaries).

See `REFACTORING.md` for the current technical concept and roadmap.

Benchmarking note: the benchmark suite is treated as a correctness+performance harness — benchmarks report allocation metrics by default, and search benchmarks additionally report `recall@k` against an exact baseline.

---

## SIMD and CPU Features

Vecgo’s hot-path distance math is implemented in `internal/simd` and dispatched at runtime based on CPU features (AVX/AVX-512 on amd64, NEON on arm64) with a safe generic fallback.

Practical guidance:

- Prefer AVX2/AVX-512 (amd64) or NEON (arm64) for best throughput.
- Build with `-tags noasm` when debugging portability issues (forces generic math).
- If you control embedding size, common dims like 128/256/384/512 tend to perform well.

---

## Compaction (When it Happens, What it Produces)

Current behavior:

- `Flush()` writes a new Flat segment to disk.
- Compaction is triggered after flush and runs in the background.
- Compaction chooses the output segment format based on total row count:
  - Below a threshold: compaction outputs a Flat segment.
  - Above a threshold: compaction outputs a DiskANN segment.

### Configurable knobs (current)

The public facade exposes compaction policy configuration via engine options:

```go
db, err := vecgo.Open(
	"./data",
	128,
	vecgo.MetricL2,
	vecgo.WithCompactionThreshold(8),
)
```

Notes:

- The default policy is size-tiered by segment *count* (default threshold is 4).
- DiskANN/quantization knobs are exposed via engine options (see `engine.CompactionConfig` and `engine.WithCompactionConfig`).

---

## Caching (Current Reality)

For local disk + mmap segments, the OS page cache is the primary “block cache”.

This repo includes a `cache` package (e.g. `cache.LRUBlockCache`) intended for non-mmap/BlobStore readers and small immutable blocks. The current `engine` constructs a default bounded block cache and threads it into segment readers.

Practical guidance:

- On local SSD: focus on access patterns (contiguous columns, fewer random seeks) and rely on the OS page cache.
- If/when adding a non-mmap reader: add an explicit bounded cache governed by ResourceController budgets (bytes) and ensure cache keys include SegmentID and (if needed) ManifestID.

---

## Segment Integration Status

- Flat segments: fully integrated (flush + open + search).
- DiskANN segments: integrated as a compaction output and openable on restart.
- HNSW: used for the L0 MemTable via `internal/hnsw`. There is currently no engine-integrated on-disk HNSW segment type.
- Quantization: flat segments support SQ8/PQ via header flags, but the engine compacts to *unquantized* flat segments by default unless configured via `engine.CompactionConfig.FlatQuantizationType`.

---

## Durability (WAL)

The engine exposes WAL durability via `vecgo.WithWALOptions(...)` (or `engine.WithWALOptions(...)`):

- `vecgo.DurabilityAsync`: highest throughput, weakest durability.
- `vecgo.DurabilitySync`: strongest durability; uses GroupCommit to batch fsync.
