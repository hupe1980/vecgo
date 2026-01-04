# Vecgo Examples

This directory contains examples demonstrating how to use Vecgo.

## [Basic](./basic)

A simple example showing how to:
1. Open the engine.
2. Insert vectors.
3. Perform a search.

## [Advanced](./advanced)

Demonstrates advanced configuration options:
1. **Resource Governance**: Limiting memory usage.
2. **Flush Control**: Configuring automatic flush thresholds.
3. **Compaction**: Tuning compaction triggers and DiskANN promotion.
4. **Durability**: Using Async WAL for higher throughput.

## [RAG](./rag)

A complete Retrieval-Augmented Generation workflow:
1. **Ingest**: Store text chunks as payload alongside vectors.
2. **Retrieve**: Use `WithPayload()` to fetch context in a single hop.
3. **Generate**: Construct a prompt for an LLM (simulated).
