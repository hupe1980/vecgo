# Vecgo Examples

This directory contains examples demonstrating Vecgo usage.

## Best Practices

All examples follow these conventions:

1. **Use `testutil.RNG`** for reproducible random vector generation
2. **Deterministic seeds** (e.g., seed=42) for consistent results
3. **Context-first API** with proper timeout handling
4. **Structured logging** via `log/slog`

## [Basic](./basic)

A simple example showing:
1. Create/open an index
2. Insert vectors
3. Perform a search
4. Commit to disk

## [Modern](./modern)

Demonstrates the fluent API with:
1. Structured logging (`log/slog`)
2. Schema-enforced metadata
3. Typed metadata fields
4. Scan iterator

## [RAG](./rag)

A complete Retrieval-Augmented Generation workflow:
1. **Ingest**: Store text chunks as payload alongside vectors
2. **Retrieve**: Fetch context in a single hop
3. **Generate**: Construct a prompt for an LLM (simulated)

## [Cloud Tiered](./cloud_tiered)

Demonstrates **writer/reader separation** architecture:
1. Build index locally with `Local()` backend
2. Sync to simulated S3 bucket
3. Open from S3 with `Remote()` backend (automatically read-only)
4. Multi-tier caching: RAM → Disk → Remote

## [Observability](./observability)

Full Prometheus metrics integration:
1. `MetricsObserver` implementation
2. Operation latency histograms
3. MemTable usage gauges
4. Backpressure event counters

*Note: Has separate `go.mod` for Prometheus dependency isolation.*

## [Bulk Load](./bulk_load)

Demonstrates **high-throughput ingestion** with `BatchInsertDeferred`:
1. Load 10K vectors at ~2M vec/s (no HNSW indexing)
2. Commit to DiskANN segment
3. Search verification
4. Performance comparison with standard `BatchInsert`

Key insight: `BatchInsertDeferred` achieves **~1000x faster** throughput by deferring
HNSW graph construction. Vectors become searchable after `Commit()`.

## Running Examples

```bash
# Main module examples
cd examples/basic && go run main.go
cd examples/modern && go run main.go
cd examples/rag && go run main.go
cd examples/cloud_tiered && go run main.go
cd examples/bulk_load && go run main.go

# Observability (separate module)
cd examples/observability && go run main.go
# View metrics at http://localhost:2112/metrics
```
