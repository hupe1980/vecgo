# Benchmarks (Vecgo)

This folder contains the production-grade benchmark suite for performance testing and regression detection.

## Quick Start (Fast Benchmarks with Fixtures)

The benchmark suite uses **pre-built fixtures** for instant startup. This eliminates the massive overhead of rebuilding indexes (10-60 seconds) for each benchmark.

```bash
# One-time: Generate fixtures (~2 minutes)
go test -tags=genfixtures -run=TestGenerateFixtures -v ./benchmark_test/...

# Run fast benchmarks using fixtures
go test -bench=BenchmarkFast -benchtime=1s -run=^$ ./benchmark_test/...

# Quick CI benchmarks (small fixtures only)
go test -tags=genfixtures -run=TestGenerateFixtures -short -v ./benchmark_test/...
go test -bench=BenchmarkFast -benchtime=1s -short -run=^$ ./benchmark_test/...
```

**Performance comparison:**
| Benchmark Style | Time | CPU Usage | Notes |
|-----------------|------|-----------|-------|
| Old (rebuild each time) | 21s | 197s | Wastes 90% on index building |
| New (fixtures) | 18s | 19s | **10x less CPU, instant startup** |

## Fixture System

### The 5-Distribution Framework (Best-in-Class Methodology)

Based on industry analysis (Snowflake, BigQuery, DuckDB), we benchmark against **5 targeted distributions**:

| Distribution | Fixture | What It Tests | Failure Mode |
|--------------|---------|---------------|--------------|
| **Uniform** | `uniform_*` | Baseline, algorithmic efficiency | Happy path, SIMD |
| **Zipfian** | `zipfian_*` | Hot keys, cache behavior | Planner lies |
| **Segment-local skew** | `seglocal_*` | Planner correctness | Global 1% vs per-segment 90% |
| **Correlated vectors** | `correlated_*` | ANN + filter interaction | Graph pruning impact |
| **Boolean adversarial** | `booladv_*` | Bitmap operations | Allocation cliffs |

### Available Fixtures

| Fixture | Dim | Vectors | Distribution | Use Case |
|---------|-----|---------|--------------|----------|
| `uniform_128d_10k` | 128 | 10K | Uniform | CI, quick iteration |
| `zipfian_128d_10k` | 128 | 10K | Zipfian | CI, realistic |
| `uniform_128d_50k` | 128 | 50K | Uniform | Standard benchmarks |
| `zipfian_128d_50k` | 128 | 50K | Zipfian | Optimization validation |
| `seglocal_128d_50k` | 128 | 50K | Segment-local skew | Planner testing |
| `correlated_128d_50k` | 128 | 50K | Correlated | ANN+filter interaction |
| `booladv_128d_50k` | 128 | 50K | Boolean adversarial | Bitmap stress |
| `nofilter_128d_50k` | 128 | 50K | No filter | Unfiltered baseline |
| `uniform_768d_50k` | 768 | 50K | Uniform | Production dimensions |
| `zipfian_768d_50k` | 768 | 50K | Zipfian | Production + realistic |
| `uniform_768d_100k` | 768 | 100K | Uniform | Large-scale testing |
| `zipfian_768d_100k` | 768 | 100K | Zipfian | Production-scale |

### Fixture Commands

```bash
# List fixture status
go test -tags=genfixtures -run=TestListFixtures -v ./benchmark_test/...

# Generate specific fixture
go test -tags=genfixtures -run=TestGenerateFixture/uniform_768d_100k -v ./benchmark_test/...

# Clean all fixtures
go test -tags=genfixtures -run=TestCleanFixtures -v ./benchmark_test/...
```

## Benchmark Categories

### Fast Benchmarks (Recommended - uses fixtures)
- `BenchmarkFastSearch` - Search across all 5 distributions with varying selectivity
- `BenchmarkFastBatchSearch` - Batched search throughput
- `BenchmarkFastConcurrentSearch` - Concurrent search scaling
- `BenchmarkFastFiltered` - Filtered search at all selectivities
- `BenchmarkFastDimensions` - Search across embedding dimensions

### Production Metrics Benchmarks (`metrics_bench_test.go`) ⭐ NEW
Essential for SLA compliance and capacity planning:
- `BenchmarkLatencyPercentiles` - P50/P95/P99/P99.9 search latency
- `BenchmarkLatencyUnderLoad` - Latency percentiles under concurrent load
- `BenchmarkMemoryFootprint` - Memory usage per vector (bytes/vec, overhead ratio)
- `BenchmarkIndexBuildTime` - Time to build index from scratch
- `BenchmarkColdStart` - First-query latency after Open() (serverless critical)
- `BenchmarkRecallQPSTradeoff` - Recall vs throughput Pareto frontier
- `BenchmarkDeletePerformance` - Delete throughput and post-delete search
- `BenchmarkGCPressure` - Allocation patterns under sustained load

### Algorithm Tuning Benchmarks (`tuning_bench_test.go`) ⭐ NEW
For optimizing HNSW/DiskANN parameters:
- `BenchmarkEfSearchTuning` - Effect of ef_search on recall/latency
- `BenchmarkBuildQuality` - Build time vs search quality trade-off
- `BenchmarkDimensionScaling` - Performance across embedding dimensions
- `BenchmarkBatchSizeOptimal` - Find optimal batch size for insert throughput

### Insert Benchmarks (`insert_bench_test.go`)
- `BenchmarkInsert` - Single insert across dimensions (128, 768, 1536)
- `BenchmarkBatchInsert` - Batch insert with varying batch sizes
- `BenchmarkDeferredInsert` - Bulk load path (deferred indexing)
- `BenchmarkConcurrentInsert` - Concurrent insert scaling (1-8 goroutines)

### MemTable Search Benchmarks (`search_only_bench_test.go`)
- `BenchmarkSearchOnly` - MemTable (L0) search path with proper methodology
- `BenchmarkSearchOnlySelectivity` - Selectivity sweep on MemTable path

### Workload Benchmarks (`workload_bench_test.go`)
- `BenchmarkMixedWorkload` - Concurrent read/write at various ratios (50-99% reads)
- `BenchmarkBurstWorkload` - Burst traffic simulation
- `BenchmarkReadAfterWrite` - Memtable search path (real-time apps)
- `BenchmarkThroughputUnderLoad` - Sustained load throughput

### Hybrid Search (`hybrid_bench_test.go`)
- `BenchmarkHybridSearch` - Vector/Hybrid/Lexical comparison

### Other Benchmarks
- `BenchmarkCompaction_Pressure` - Compaction under write load
- `BenchmarkBinaryQuantizer_Distance` - Binary quantizer performance
- `BenchmarkStorage_ReadRandom` - Storage layer random reads

## Data Distributions

| Distribution | Description | Activates |
|--------------|-------------|-----------|
| **Uniform** | Random vectors, modulo bucket | Adversarial baseline |
| **Zipfian** | Clustered vectors, power-law buckets | Segment purity, HNSW locality, pruning |
| **NoFilter** | No metadata | Pure vector search baseline |

## Metrics Reported

| Metric | Description |
|--------|-------------|
| `ns/op` | Latency per operation (lower is better) |
| `allocs/op` | Allocations per operation (lower is better) |
| `vectors/sec` | Insert throughput |
| `qps` | Query throughput |
| `ops/sec` | Combined operations throughput |
| `recall@10` | Search quality (higher is better) |
| `P50_μs` | Median latency in microseconds |
| `P95_μs` | 95th percentile latency |
| `P99_μs` | 99th percentile latency (tail latency) |
| `P99.9_μs` | 99.9th percentile latency |
| `bytes/vec` | Memory usage per vector |
| `overhead_ratio` | Memory overhead vs raw vector size |
| `open_ms` | Time to open database |
| `cold_start_ms` | Open + first query latency |

## Running Benchmarks

```bash
# RECOMMENDED: Fast benchmarks with fixtures
go test -bench=BenchmarkFast -benchtime=1s -run=^$ ./benchmark_test/...

# Production metrics (latency percentiles, memory, cold start)
go test -bench='BenchmarkLatency|BenchmarkMemory|BenchmarkCold' -short -run=^$ ./benchmark_test/...

# Algorithm tuning (dimension scaling, batch size optimization)
go test -bench=BenchmarkDimension -short -run=^$ ./benchmark_test/...

# Legacy: All benchmarks (slower, rebuilds indexes)
just bench-current

# Filtered search comparison
go test -bench=BenchmarkFastFiltered -benchtime=3s -run=^$ ./benchmark_test/...

# Run with memory profiling
go test -bench=BenchmarkFastSearch -benchmem -memprofile=mem.out -run=^$ ./benchmark_test/...

# CPU profiling
go test -bench=BenchmarkFastSearch -cpuprofile=cpu.out -run=^$ ./benchmark_test/...
```

## Baseline Comparison

```bash
# Install benchstat (once)
go install golang.org/x/perf/cmd/benchstat@latest

# Record baseline
just bench-baseline

# Compare against baseline
benchstat benchmark_test/baseline.txt benchmark_test/current.txt
```

## Profiling

```bash
# CPU profile
go test -bench=BenchmarkConcurrentSearch -cpuprofile=cpu.out ./benchmark_test/...
go tool pprof -http=:8080 cpu.out

# Memory profile
go test -bench=BenchmarkMixedWorkload -memprofile=mem.out ./benchmark_test/...
go tool pprof -http=:8080 mem.out

# Block profile (for concurrency analysis)
go test -bench=BenchmarkConcurrentInsert -blockprofile=block.out ./benchmark_test/...
```

## Configuration

Standard dimensions (matching real-world embeddings):
- `128` - Fast CI benchmarks
- `768` - OpenAI text-embedding-3-small, Cohere v3
- `1536` - OpenAI text-embedding-3-large

Standard dataset sizes:
- `10K` - Quick iteration
- `50K` - Default CI
- `100K` - Production-scale

## Methodology

- **Single query per iteration** — Each `b.N` iteration executes exactly one query (not batched)
- **Deterministic seed** (`benchSeed=42`) for reproducible comparisons
- **Warmup** before search benchmarks
- **Setup outside timed region** (`b.ResetTimer()` after data loading)
- **Custom metrics** via `b.ReportMetric()`
- **Parallel benchmarks** via `b.RunParallel()` for concurrency testing
- **Memory limit** — Default 64MB memtable to prevent OOM during test runs
