# Vecgo Development Guide

This guide outlines the workflows for building, testing, and benchmarking Vecgo.

## Prerequisites

- **Go 1.23+** (Check `go.mod` for exact version)
- **Make** (optional, for convenience scripts)
- **benchstat** (for benchmark comparison): `go install golang.org/x/perf/cmd/benchstat@latest`

## Build & Test

### Unit Tests
Run all unit tests with race detection (required for CI):
```bash
go test -race ./...
```

### Fuzz Testing
Run fuzz tests to ensure robustness against corrupted inputs:
```bash
go test -fuzz=. -fuzztime=30s ./...
```

### Linting
We use `golangci-lint`. Ensure it passes before submitting PRs.

## Benchmarking

Performance is a P0 feature. Regressions must be justified.

### Running Benchmarks
Run the standard benchmark suite:
```bash
go test -bench=. -benchmem ./benchmark_test
```

### Detecting Regressions
We use `benchstat` to compare against the baseline.

1. **Run baseline** (if not already present):
   ```bash
   git checkout main
   go test -bench=. -benchmem -count=5 ./benchmark_test > baseline.txt
   ```

2. **Run your changes**:
   ```bash
   git checkout my-feature
   go test -bench=. -benchmem -count=5 ./benchmark_test > new.txt
   ```

3. **Compare**:
   ```bash
   benchstat baseline.txt new.txt
   ```

**CI Gate**:
- Allocations > 10% increase -> **FAIL**
- Recall > 5% drop -> **FAIL**
- Latency > 20% increase -> **FAIL**

## Code Style

- **Zero Panics**: Never panic on user input. Return errors.
- **Zero Allocations (Hot Path)**: Search and Insert paths must minimize heap allocations. Use `iter.Seq2` for iterators.
- **Concurrency**: Use `go test -race` to verify thread safety.

## Release Process

(To be defined - currently in pre-release phase)
