# Vecgo Contributor Guide

Join us in building the best-in-class embeddable vector database for Go.

## Development Environment

### Prerequisites
*   Go 1.24+ (1.25+ recommended)
*   [Just](https://github.com/casey/just) task runner
*   Git

### Quick Start
```bash
git clone https://github.com/hupe1980/vecgo.git
cd vecgo
just test       # Fast tests
just test-race  # With race detector (CI requirement)
```

## CI/CD Pipeline

Our GitHub Actions CI runs on every push and PR:

| Job | Description |
|-----|-------------|
| `lint` | golangci-lint code quality |
| `test-platforms` | Linux/macOS/Windows × Go 1.24/1.25 |
| `test-simd` | SIMD kernel verification (AVX2, NEON) |
| `test-mmap` | Memory-mapped I/O cross-platform |
| `test-integration` | End-to-end tests |
| `benchmark` | Performance regression (PRs only) |

## Workflows

### 1. Running Tests
We use strict testing gates.
```bash
just test       # Fast: go test ./...
just test-race  # Race detection (required for CI)
just simd-test  # SIMD kernel tests only
```

### 2. Performance Tracking
Before submitting a PR, check if you regressed performance.
```bash
# Record baseline
just bench-baseline

# Make changes...

# Run current and compare
just bench-current
just bench-compare
```

CI automatically runs `benchstat` on PRs and warns about >10% regressions.

### 3. SIMD Development
```bash
just simd-bench         # Run SIMD benchmarks (with assembly)
just simd-bench-noasm   # Run with pure-Go fallbacks
```

### 4. Code Style
*   Standard Go formatting (`gofmt`).
*   Idiomatic Go 1.24+ (use `iter.Seq2`, `min/max`).
*   No `panic` in library code. Use `fmt.Errorf`.

## Project Structure

*   `internal/engine/`: Core logic (Commit, Compaction, Snapshots, Orchestration).
*   `internal/segment/`: Immutable segment implementations (Flat, DiskANN, HNSW).
*   `internal/simd/`: SIMD kernels (AVX2/AVX-512/NEON/SVE2).
*   `internal/mmap/`: Memory-mapped file I/O.
*   `internal/manifest/`: Metadata management.
*   `blobstore/`: Storage abstraction (Local, S3).
*   `internal/cache/`: Block cache (LRU, Disk).
*   `lexical/`: BM25/Hybrid search components.
*   `examples/`: User-facing examples.
*   `benchmark_test/`: Performance benchmarks.

## Pull Request Checklist

1.  [ ] Tests pass (`just test-race`).
2.  [ ] Benchmarks run (no regression via `just bench-compare`).
3.  [ ] SIMD equivalence tests pass (if modifying kernels).
4.  [ ] Documentation updated (godoc + `docs/`).
5.  [ ] No new panics introduced.
