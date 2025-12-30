# Vecgo Examples

This directory contains runnable examples demonstrating various features of Vecgo.

## Basic Usage

- **[hnsw/](hnsw/main.go)**: Basic HNSW index usage (In-Memory). Shows fluent builder, inserting vectors, and searching.
- **[flat/](flat/main.go)**: Flat index usage (Brute Force). Best for small datasets or 100% recall requirements.
- **[diskann/](diskann/main.go)**: DiskANN index usage (SSD-Resident). Best for datasets larger than RAM.

## Advanced Features

- **[sharding/](sharding/main.go)**: Demonstrates multi-core write scaling using sharding.
- **[persistence/](persistence/main.go)**: Saving and loading indexes to/from disk.
- **[wal/](wal/main.go)**: Configuring Write-Ahead Logging (WAL) for durability.
- **[streaming/](streaming/main.go)**: Using the Streaming Search API for processing results as they arrive.
- **[metrics/](metrics/main.go)**: Exposing internal metrics (latency, throughput, memory usage).

## Quantization (Compression)

- **[quantization/](quantization/main.go)**: Scalar Quantization (8-bit) for 4x memory reduction.
- **[opq/](opq/main.go)**: Optimized Product Quantization (OPQ) for higher compression ratios (8x-32x).

## Internals

- **[columnar/](columnar/main.go)**: Direct usage of the columnar vector store.

## Running Examples

You can run any example using `go run`:

```bash
go run ./_examples/hnsw/main.go
go run ./_examples/sharding/main.go
```
