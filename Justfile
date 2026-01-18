# Setup installs dependencies
setup:
    go mod tidy

# Runs the linter
lint:
    golangci-lint run --config .golangci.yml --timeout=2m ./...

# Runs go test with default values
test:
    go test -p 1 -v -timeout 300s ./...

# Runs go test with the race detector
test-race:
    go test -p 1 -race -v -timeout 300s ./...

# Runs benchmarks and saves output to benchmark_test/current.txt
# Uses -short to skip heavy benchmarks (500K vectors, etc.)
bench-current:
    @echo "Running benchmarks (use 'bench-full' for comprehensive suite)..."
    go test -bench=. -short -benchmem -benchtime=200ms -timeout=15m ./benchmark_test | tee benchmark_test/current.txt
    @echo "Done. Results saved to benchmark_test/current.txt"

# Runs full benchmark suite including heavy tests (500K+ vectors)
# WARNING: Takes 30+ minutes and requires significant RAM
bench-full:
    @echo "Running FULL benchmark suite (this will take a while)..."
    go test -bench=. -benchmem -benchtime=200ms -timeout=60m ./benchmark_test | tee benchmark_test/current.txt
    @echo "Done. Results saved to benchmark_test/current.txt"


# Records a new baseline benchmark run.
# Intended for use on your own machine when you explicitly want to update
# the reference point for `benchstat` comparisons.
bench-baseline:
    @echo "Recording baseline benchmarks..."
    go test -bench=. -short -benchmem -benchtime=200ms -timeout=15m ./benchmark_test | tee benchmark_test/baseline.txt
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

# Starts the local Jekyll documentation site (http://localhost:4000)
docs-serve:
    cd docs && bundle install && bundle exec jekyll serve --livereload --config _config.yml,_config.dev.yml
