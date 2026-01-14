# Vecgo Deployment Guide

Vecgo is an **embedded** library. Deployment means deploying **your application**.
However, Vecgo's presence introduces stateful requirements.

## Deployment Models

### 1. Local Disk (Single Node)
*   **Best for**: Low latency, simpler ops, smaller datasets (<1TB).
*   **Architecture**: App + NVMe SSD.
*   **Requirements**:
    *   **Persistent Volume**: If deploying in K8s/Docker, mount a PVC. Local ephemeral disk means data loss on pod restart!
    *   **One Process**: Only **one** process can hold the file lock on the Vecgo directory. DO NOT run replicas sharing the same NFS/EFS mount.

### 2. Cloud Blob Store (Stateless-ish)
*   **Best for**: "Bottomless" storage, separation of compute/storage, large datasets.
*   **Architecture**: App + S3/GCS + Local Cache (SSD/RAM).
*   **Configuration**:
    ```go
    store := s3.NewStore(client, "my-bucket", "prefix")
    eng, _ := vecgo.Open(ctx, vecgo.Remote(store), vecgo.WithBlockCacheSize(4*1024*1024*1024))
    ```
*   **Caveat**: Manifest is currently still **local**. Only immutable segments are offloaded.
    *   The PK index is held in memory and is rebuilt at startup by scanning segment PK columns.
    *   *Note: Truly stateless shared-storage requires specific coordination not yet fully GA in v1.0.*

## Resource Sizing

### CPU
*   **Indexing**: CPU intensive. 1 core per 2k vectors/sec ingest.
*   **Search**: Latency bound.
    *   **L0 (HNSW)**: Fast, lower CPU.
    *   **Quantization (SQ8/PQ)**: SIMD heavy. AVX-512/NEON recommended.

### Memory
*   **Mandatory Overhead**: 
    *   MemTable (configured limit, default 64MB).
    *   BlockCache (configured limit, default 256MB).
*   **Index Overhead**:
    *   HNSW: ~5-10% of vector size.
    *   PK Index: ~16 bytes per vector (can be large! 100M vectors = 1.6GB).

### Disk I/O
*   **IOPS**: Critical for:
    1.  Commit syncs (latency).
    2.  Compaction (throughput).
    3.  Segment reads during search (if not in block cache).
*   **Recommendation**: NVMe or high-IOPS EBS (gp3/io2). Avoid HDD.

## Scaling Patterns

### Vertical Scaling (Scale Up)
*   Vecgo scales well with cores (concurrent search/index).
*   Simplest approach.

### Horizontal Sharding
*   Vecgo is **not distributed**. 
*   To shard:
    1.  App Layer partitions data (hash of PK, or by tenant).
    2.  Each App Instance manages a separate Vecgo directory/DB.
    3.  Aggregator performs scatter-gather search.

## Backups
*   **Snapshot**: 
    1.  Pause writes (optional, but ensures consistent point-in-time).
    2.  Copy directory (snapshot-isolation means copying `.bin` files and Manifest is safe).
*   **Scan API**: Use `engine.Scan()` to stream all records to JSON/Parquet for logical backup.
