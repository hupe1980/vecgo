# RAG (Retrieval-Augmented Generation) Example

This example demonstrates how to build a simple RAG pipeline using Vecgo.

## Overview

Retrieval-Augmented Generation (RAG) enhances LLM responses by retrieving relevant data from a knowledge base and feeding it as context.

This example shows:
1.  **Ingestion**: Storing text documents (chunks) as `payload` alongside their vector embeddings.
2.  **Retrieval**: Performing a vector search and retrieving the text payload in a single operation using `engine.WithPayload()`.
3.  **Context Construction**: Formatting the retrieved text for an LLM prompt.

## Running the Example

```bash
go run main.go
```

## Key Concepts

- **Payload Storage**: Vecgo allows storing arbitrary binary data (like text chunks, JSON, or serialized objects) directly in the segment files. This eliminates the need for a separate "blob store" or database lookup for simple RAG use cases.
- **Zero-Copy Retrieval**: When using memory-mapped segments, retrieving the payload is efficient and does not require complex joins.
- **`engine.WithPayload()`**: This search option instructs the engine to materialize the payload field for the top-k results.
```