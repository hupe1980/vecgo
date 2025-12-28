# Vecgo Benchmark Suite

Comprehensive benchmark suite for establishing performance baselines and tracking optimizations.

## Overview

This benchmark suite covers all major Vecgo features:

- **Index Types**: Flat, HNSW, DiskANN
- **Operations**: Insert, Batch Insert, Search, Update, Delete
- **Features**: Metadata filtering, Quantization, Persistence, WAL
- **Configurations**: Different dimensions, dataset sizes, and parameters

## Running Benchmarks

### Run All Benchmarks

```bash
go test -bench=. -benchmem ./benchmark_test
```

### Run Specific Categories

```bash
# Flat index benchmarks
go test -bench=BenchmarkFlat -benchmem ./benchmark_test

# HNSW index benchmarks
go test -bench=BenchmarkHNSW -benchmem ./benchmark_test

# DiskANN index benchmarks
go test -bench=BenchmarkDiskANN -benchmem ./benchmark_test

# Quantization benchmarks
go test -bench=BenchmarkBinary -benchmem ./benchmark_test
go test -bench=BenchmarkProduct -benchmem ./benchmark_test

# Persistence benchmarks
go test -bench=BenchmarkSnapshot -benchmem ./benchmark_test
go test -bench=BenchmarkWAL -benchmem ./benchmark_test

# Metadata benchmarks
go test -bench=BenchmarkMetadata -benchmem ./benchmark_test
```

### Generate Baseline Results

```bash
# Save baseline for future comparisons
go test -bench=. -benchmem ./benchmark_test > baseline_$(date +%Y%m%d).txt

# Compare with previous baseline
go test -bench=. -benchmem ./benchmark_test > current.txt
benchstat baseline_YYYYMMDD.txt current.txt
```

## Benchmark Categories

### 1. Flat Index (`flat_bench_test.go`)

- **Insert**: Single vector insertion at different dimensions (128, 384, 768, 1536)
- **BatchInsert**: Batch insertion with varying batch sizes (10, 100, 1000)
- **Search**: KNN search at different dataset sizes (1K, 10K, 100K)
- **SearchK**: Search with different K values (1, 10, 50, 100)
- **DistanceMetrics**: Comparison of SquaredL2, Cosine, DotProduct
- **Update**: Vector update performance
- **Delete**: Vector deletion performance
- **HybridSearch**: Search with metadata filters
- **StreamingSearch**: Streaming search with early termination

### 2. HNSW Index (`hnsw_bench_test.go`)

- **Insert**: Single vector insertion at different dimensions
- **BatchInsert**: Batch insertion performance
- **Search**: KNN search at scale (10K, 100K, 1M vectors)
- **SearchEF**: Search quality vs speed tradeoff (EF: 50, 100, 200, 400)
- **BuildM**: Index construction with different M values (8, 16, 32, 64)
- **ConcurrentSearch**: Parallel search throughput
- **ShardedInsert**: Sharded write performance (1, 2, 4, 8 shards)
- **ShardedSearch**: Sharded search latency
- **HybridSearch**: HNSW with metadata filters
- **Update**: Vector updates in HNSW

### 3. DiskANN Index (`diskann_bench_test.go`)

- **Insert**: Single vector insertion at different dimensions
- **BatchInsert**: Batch insertion performance
- **Search**: KNN search at scale (1K, 10K, 100K vectors)
- **BeamWidth**: Search with different beam widths (2, 4, 8, 16)
- **RTuning**: Index construction with different R values (32, 64, 100, 128)
- **DistanceMetrics**: Comparison of SquaredL2, Cosine, DotProduct
- **Update**: Vector updates in DiskANN
- **Delete**: Vector deletion performance
- **HybridSearch**: DiskANN with metadata filters
- **StreamingSearch**: Streaming search with early termination

### 4. Quantization (`quantization_bench_test.go`)

- **BinaryQuantization**: Encoding and Hamming distance (128-1536 dims)
- **ProductQuantization**: PQ encoding, decoding, asymmetric distance
- **OptimizedPQ**: OPQ training and operations
- **FlatWithPQ**: End-to-end search with PQ compression
- **ScalarQuantization**: 8-bit quantization performance

### 5. Persistence (`persistence_bench_test.go`)

- **SnapshotSave**: Snapshot creation time (1K, 10K, 100K vectors)
- **SnapshotLoad**: Regular loading time
- **SnapshotLoadMmap**: Zero-copy mmap loading
- **WALWrite**: WAL write performance (Async, GroupCommit, Sync)
- **WALReplay**: Recovery time from WAL
- **WALCompression**: Compressed vs uncompressed WAL

### 6. Metadata (`metadata_bench_test.go`)

- **FilterCompilation**: Filter compilation with multiple filters (1, 5, 10, 20)
- **MetadataInsert**: Insertion with varying metadata fields (0, 5, 10, 20)
- **SearchSelectivity**: Filter selectivity impact (1%, 10%, 50%, 90%)
- **MetadataOperators**: Different operators (Equal, GreaterThan, LessThan, Combined)

## Interpreting Results

### Key Metrics

- **ns/op**: Nanoseconds per operation (lower is better)
- **B/op**: Bytes allocated per operation (lower is better)
- **allocs/op**: Allocations per operation (lower is better)
- **vectors**: Custom metric for dataset size

### Example Output

```
BenchmarkFlatInsert/dim384-8         50000    30245 ns/op    2048 B/op    3 allocs/op
BenchmarkHNSWSearch/100K-8          10000   125432 ns/op       0 B/op    0 allocs/op
```

### Performance Targets

**Flat Index**:
- Insert: < 50µs (384-dim)
- Search (10K): < 5ms
- Zero allocations for search

**HNSW Index**:
- Insert: < 200µs (384-dim, M=16)
- Search (100K): < 150µs (EF=100)
- Zero allocations for search

**DiskANN Index**:
- Insert: < 300µs (384-dim)
- Search (100K): < 200µs (BeamWidth=4)
- Disk-optimized for large datasets

**Quantization**:
- Binary encoding: < 100ns (128-dim)
- Hamming distance: < 1ns
- PQ asymmetric: < 500ns

**WAL**:
- Async: < 10µs/op
- GroupCommit: < 50µs/op
- Sync: < 5ms/op

## Continuous Benchmarking

### Track Performance Over Time

```bash
# Run and save results with git commit
git rev-parse HEAD > commit.txt
go test -bench=. -benchmem ./benchmark_test > bench_$(git rev-parse --short HEAD).txt
```

### Compare Commits

```bash
# Compare two commits
benchstat bench_abc123.txt bench_def456.txt
```

### CI Integration

Add to `.github/workflows/benchmark.yml`:

```yaml
name: Benchmark
on: [push, pull_request]
jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-go@v4
      - run: go test -bench=. -benchmem ./benchmark_test
```

## Notes

- Benchmarks use random data for reproducibility
- Some benchmarks may take several minutes
- Use `-benchtime=10s` for more stable results
- Memory benchmarks require `-benchmem` flag
- Parallel benchmarks use `RunParallel` for concurrency testing

## Future Enhancements

- [ ] Memory profiling (`-memprofile`)
- [ ] CPU profiling (`-cpuprofile`)
- [ ] Trace generation (`-trace`)
- [ ] Benchmark regression detection
- [ ] Automated performance tracking
