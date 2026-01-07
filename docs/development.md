# Vecgo Contributor Guide

Join us in building the best-in-class embeddable vector database for Go.

## Development Environment

### Prerequisites
*   Go 1.23+
*   Make (optional)
*   Git

### Quick Start
```bash
git clone https://github.com/hupe1980/vecgo.git
cd vecgo
go test ./...
```

## Workflows

### 1. Running Tests
We use strict testing gates.
*   **Fast**: `go test ./...`
*   **Race Detection** (Required): `go test -race ./...`
*   **Benchmarks**: `go test -bench=. ./benchmark_test/...`

### 2. Performance Tracking
Before submitting a PR, check if you regressed performance.
```bash
# Save baseline
go test -bench=. -count=5 ./benchmark_test > old.txt

# Apply changes...

# Compare
go test -bench=. -count=5 ./benchmark_test > new.txt
benchstat old.txt new.txt
```
**Do NOT** submit PRs with significant regression unless justified.

### 3. Code Style
*   Standard Go formatting (`gofmt`).
*   Idiomatic Go 1.23+ (use `iter.Seq2`, `min/max`).
*   No `panic` in library code. Use `fmt.Errorf`.

## Project Structure

*   `engine/`: Core logic (WAL, Flush, Compaction, Orchestration).
*   `internal/segment/`: Immutable segment implementations (Flat, DiskANN, HNSW).
*   `internal/manifest/`: Metadata management.
*   `blobstore/`: Storage abstraction (Local, S3).
*   `cache/`: Block cache.
*   `lexical/`: BM25/Hybrid search components.
*   `examples/`: User-facing examples.

## Pull Request Checklist

1.  [ ] Tests pass (`-race`).
2.  [ ] Benchmarks run (no regression).
3.  [ ] Documentation updated (godoc + `docs/`).
4.  [ ] `TODO.md` updated (if completing a task).
5.  [ ] No new panics introduced.
