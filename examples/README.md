# Vecgo Examples

This directory contains examples demonstrating Vecgo usage.

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
4. Scan iterator (Go 1.23+)

## [RAG](./rag)

A complete Retrieval-Augmented Generation workflow:
1. **Ingest**: Store text chunks as payload alongside vectors
2. **Retrieve**: Fetch context in a single hop
3. **Generate**: Construct a prompt for an LLM (simulated)

## [Cloud Tiered](./cloud_tiered)

Demonstrates cloud storage with caching:
1. Build index locally
2. Upload to simulated S3
3. Open in read-only mode with disk cache
4. Compare cold vs warm cache performance

## Running Examples

```bash
cd examples/basic && go run main.go
cd examples/modern && go run main.go
cd examples/rag && go run main.go
cd examples/cloud_tiered && go run main.go
```
