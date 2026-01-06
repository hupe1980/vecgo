# Setup installs dependencies
setup:
    go mod tidy

# Runs the linter
lint:
    golangci-lint run --config .golangci.yml --timeout=2m ./...

# Runs go test with default values
test:
    go test -v -timeout 120s ./...

# Runs go test with the race detector
test-race:
    go test -race -v -timeout 300s ./...

# Runs benchmarks and saves output to benchmark_test/current.txt
bench-current:
    @echo "Running benchmarks..."
    go test -bench=. -benchmem ./benchmark_test | tee benchmark_test/current.txt
    @echo "Done. Results saved to benchmark_test/current.txt"


# Records a new baseline benchmark run.
# Intended for use on your own machine when you explicitly want to update
# the reference point for `benchstat` comparisons.
bench-baseline:
    @echo "Recording baseline benchmarks..."
    go test -bench=. -benchmem ./benchmark_test | tee benchmark_test/baseline.txt
    @echo "Done. Results saved to benchmark_test/baseline.txt"

# Compare benchmark_test/baseline.txt vs benchmark_test/current.txt using benchstat.
# Requires: go install golang.org/x/perf/cmd/benchstat@latest
bench-compare:
    benchstat benchmark_test/baseline.txt benchmark_test/current.txt

# Runs SIMD microbenchmarks (asm enabled via runtime dispatch).
simd-bench:
    go test ./internal/simd -run '^$' -bench . -benchmem

# Runs SIMD microbenchmarks with `-tags noasm` (pure-Go fallbacks).
simd-bench-noasm:
    go test ./internal/simd -run '^$' -bench . -benchmem -tags noasm

# Runs internal/simd unit tests.
simd-test:
    go test -v -timeout 60s ./internal/simd

# CPU profile a specific benchmark (edit BENCH to target).
profile-cpu BENCH='BenchmarkSearch_Mixed':
    go test -run '^$' -bench '{{BENCH}}' -benchmem -cpuprofile benchmark_test/cpu.out ./benchmark_test
    go tool pprof -http=:0 benchmark_test/cpu.out

# Heap profile a specific benchmark (edit BENCH to target).
profile-heap BENCH='BenchmarkSearch_Mixed':
    go test -run '^$' -bench '{{BENCH}}' -benchmem -memprofile benchmark_test/heap.out ./benchmark_test
    go tool pprof -http=:0 benchmark_test/heap.out
