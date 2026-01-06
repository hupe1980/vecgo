# Vecgo Deployment Guide

Vecgo is an **embedded library**, meaning it runs inside your application process. This guide covers deployment patterns for local and cloud environments.

## Deployment Models

### 1. Local / Single-Node
Best for: CLI tools, desktop apps, small-scale servers.

- **Storage**: Local filesystem (SSD recommended).
- **Concurrency**: Single process (Vecgo uses file locks).
- **Scaling**: Vertical (larger instance).

### 2. Cloud / Stateless Service
Best for: Microservices, RAG APIs, scalable workers.

- **Architecture**:
  - **Writer**: Single writer instance (leader) handles ingestion.
  - **Readers**: Multiple reader instances scale search throughput.
- **Storage**:
  - **Object Store (S3/GCS)**: Store immutable segments.
  - **Local Cache**: Readers cache hot segments on local disk/RAM.
- **Replication**:
  - Writer pushes new segments to S3.
  - Readers poll manifest from S3 and download new segments.

## Cloud Object Storage (S3)

Vecgo supports reading segments directly from object storage via the `BlobStore` interface.

### Configuration
```go
store, _ := s3blob.New("my-bucket", "region")
engine, _ := vecgo.Open(..., vecgo.WithBlobStore(store))
```

### Performance Considerations
- **Latency**: S3 TTFB is ~20-50ms.
- **Caching**: Essential. Use `WithBlockCache(256MB)` to cache index nodes.
- **Cost**: Minimizes GET requests by reading large blocks (4KB-64KB).

## Resource Sizing

### CPU
- **Vector Search**: Heavily uses SIMD.
- **Recommendation**: ARM64 (Graviton/Apple Silicon) or AVX-512 capable x86.
- **Ratio**: 1 vCPU per 500-1000 QPS (approx).

### Memory
- **L0 (MemTable)**: Needs RAM.
- **Segments**: Mmapped. OS manages caching.
- **Recommendation**: RAM >= 2x Active Dataset Size for low latency.

### Disk
- **IOPS**: High write IOPS needed for WAL/Flush.
- **Throughput**: High read throughput needed for cold searches / compaction.
- **Type**: NVMe SSD (Local) or EBS gp3 (Cloud).
