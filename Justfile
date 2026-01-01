# Setup installs dependencies
setup:
    go mod tidy

# Runs the linter
lint:
    golangci-lint run --config .golangci.yml --timeout=2m ./...

# Runs go test with default values
test:
    go test -v -timeout 30s ./...

# Runs go test with the race detector
test-race:
    go test -race -v -timeout 300s ./...

# Runs benchmarks and saves output to benchmark_test/current.txt
bench-current:
    @echo "Running benchmarks..."
    go test -bench=. -benchmem ./benchmark_test ./benchmark_test/isolated | tee benchmark_test/current.txt
    @echo "Done. Results saved to benchmark_test/current.txt"

# Runs isolated benchmarks
bench-isolated:
    go test -bench=. -benchmem ./benchmark_test/isolated
