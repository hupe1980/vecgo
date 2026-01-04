# Benchmarks (Vecgo)

This folder contains the benchmark suite used for data-driven performance work.

## What we measure

- `ns/op`: latency (lower is better)
- `allocs/op` and `B/op`: allocation pressure (lower is better; benchmarks call `b.ReportAllocs()` so this is reported even without `-benchmem`)
- `recall@k`: result quality for search benchmarks (higher is better; computed against an exact baseline outside the timed region)
- scenario coverage: ingest, search (L0), search (mixed segments), durability modes

## Run

From repo root:

- `just bench-current`

To record/update a baseline run (intentionally, on your machine):

- `just bench-baseline`

Or directly:

- `go test -bench=. -benchmem ./benchmark_test`

Note: `-benchmem` is still useful (and recommended), but allocation metrics are reported even without it.

## Compare (benchstat)

Install once:

- `go install golang.org/x/perf/cmd/benchstat@latest`

Then compare two runs:

- `benchstat benchmark_test/baseline.txt benchmark_test/current.txt`

## Profiling (pprof)

CPU profile example:

- `go test -bench=BenchmarkSearch_Mixed -benchmem -cpuprofile cpu.out ./benchmark_test`
- `go tool pprof -http=:0 cpu.out`

Heap profile example:

- `go test -bench=BenchmarkSearch_Mixed -benchmem -memprofile mem.out ./benchmark_test`
- `go tool pprof -http=:0 mem.out`

## Methodology notes

- Prefer deterministic datasets: pre-generate vectors/queries outside the timed section.
- Avoid measuring random-number generation inside the benchmark loop.
- Keep benchmark parameters stable (dim, dataset size, k) so benchstat comparisons are meaningful.
