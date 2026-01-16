# Benchmarks (Vecgo)

This folder contains the production-grade benchmark suite for performance testing and regression detection.

## Benchmark Categories

### Insert Benchmarks (`insert_bench_test.go`)
- `BenchmarkInsert` - Single insert across dimensions (128, 768, 1536)
- `BenchmarkBatchInsertNew` - Batch insert with varying batch sizes
- `BenchmarkDeferredInsert` - Bulk load path (deferred indexing)
- `BenchmarkConcurrentInsert` - Concurrent insert scaling (1-8 goroutines)

### Search Benchmarks (`search_bench_test.go`)
- `BenchmarkSearchDim` - Search latency across dimensions with recall@10
- `BenchmarkSearchScaling` - Search scaling with dataset size (1K-50K)
- `BenchmarkConcurrentSearch` - Concurrent search throughput
- `BenchmarkBatchSearch` - Batched search throughput
- `BenchmarkSearchRefineFactor` - Quality/latency tradeoff

### Workload Benchmarks (`workload_bench_test.go`)
- `BenchmarkMixedWorkload` - Concurrent read/write at various ratios (50-99% reads)
- `BenchmarkBurstWorkload` - Burst traffic simulation
- `BenchmarkReadAfterWrite` - Memtable search path (real-time apps)
- `BenchmarkThroughputUnderLoad` - Sustained throughput with background writes

### Filtered/Hybrid Benchmarks
- `BenchmarkFilteredSearchSelectivity` - Filtered search at selectivity levels
- `BenchmarkHybridSearch` - Vector/Hybrid/Lexical comparison

### Storage Benchmarks (`storage_bench_test.go`)
- Local NVMe simulation
- S3 latency simulation (with cache tiers)

## Metrics Reported

| Metric | Description |
|--------|-------------|
| `ns/op` | Latency per operation (lower is better) |
| `allocs/op` | Allocations per operation (lower is better) |
| `vectors/sec` | Insert throughput |
| `qps` | Query throughput |
| `ops/sec` | Combined operations throughput |
| `recall@10` | Search quality (higher is better) |

## Quick Start

```bash
# Run all benchmarks
just bench-current

# Run specific benchmark
go test -bench=BenchmarkInsert -benchtime=1s ./benchmark_test/...

# Run with memory profiling
go test -bench=BenchmarkSearch -benchmem -memprofile=mem.out ./benchmark_test/...
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

- **Deterministic seed** (`benchSeed=42`) for reproducible comparisons
- **Warmup** before search benchmarks
- **Setup outside timed region** (`b.ResetTimer()` after data loading)
- **Custom metrics** via `b.ReportMetric()`
- **Parallel benchmarks** via `b.RunParallel()` for concurrency testing
