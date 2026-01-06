# internal/simd

This package provides SIMD-accelerated kernels (and pure-Go fallbacks) used by higher-level code such as `distance/` and `quantization/`.

## Benchmarking (asm vs noasm)

Benchmarks are designed to be run twice:

- **asm enabled** (default): uses runtime CPU feature detection to select the best available implementation (NEON on arm64; AVX2/AVX-512 on amd64).
- **generic** (`noasm`): forces pure-Go implementations by excluding all assembly.

Run from the repo root:

```bash
# asm enabled (SIMD dispatch when available)
go test ./internal/simd -run '^$' -bench . -benchmem

# noasm (pure-Go)
go test ./internal/simd -run '^$' -bench . -benchmem -tags noasm
```

Notes:
- On **arm64**, you can compare **NEON vs noasm** locally.
- On **amd64**, run the same commands on AVX2 / AVX-512 hardware to measure those paths.

## What’s covered

The benchmark suite in `kernels_bench_test.go` covers all exported APIs in this package, including:

- `Dot`, `SquaredL2`, `DotBatch`, `SquaredL2Batch`
- `ScaleInPlace`
- `PqAdcLookup`
- `F16ToF32`
- `Sq8L2Batch`, `Sq8uL2BatchPerDimension`
- `Popcount`, `Hamming`
- PQ int8 helpers: `SquaredL2Int8Dequantized`, `BuildDistanceTableInt8`, `FindNearestCentroidInt8`

## Generator (C -> Go asm)

Assembly is generated from C intrinsics in `internal/simd/src/` using `internal/simd/cmd/generator`.

Examples:

```bash
# Generate amd64 asm + stubs

go run ./internal/simd/cmd/generator -arch amd64 -goos linux internal/simd/src/pq_int8_avx.c

# Generate arm64 asm + stubs (cross-target)

go run ./internal/simd/cmd/generator -arch arm64 -goos linux internal/simd/src/pq_int8_neon.c
```

The generator enforces a “no relocations” constraint by default to keep the emitted Go assembly safe (raw `WORD`/`BYTE` emission). If you hit a relocation error, the usual fix is to rewrite the C to avoid literal pools / constants.
