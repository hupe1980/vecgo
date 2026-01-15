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

### 2. Cloud Blob Store (Cloud-Native)
*   **Best for**: "Bottomless" storage, serverless/Kubernetes, large datasets.
*   **Architecture**: Writers/Readers → S3/GCS (direct cloud I/O)

#### Direct Cloud Writes
```go
// Write directly to S3 (no local sync needed)
store := s3.NewStore(client, "my-bucket", "vecgo")
eng, _ := vecgo.Open(ctx, vecgo.Remote(store), vecgo.Create(128, vecgo.MetricL2))
eng.Insert(ctx, vector, metadata, nil)
eng.Close()
```

#### Reader Nodes (Stateless, Horizontally Scalable)
```go
// Open read-only from S3
store := s3.NewStore(client, "my-bucket", "vecgo")
eng, _ := vecgo.Open(ctx, vecgo.Remote(store), 
    vecgo.ReadOnly(),                     // Explicitly read-only for safety
    vecgo.WithBlockCacheSize(4<<30),      // 4GB block cache
    vecgo.WithCacheDir("/fast/nvme"),     // Persistent disk cache
)

// Search works
results, _ := eng.Search(ctx, query, 10)

// Writes return ErrReadOnly
_, err := eng.Insert(ctx, vec, nil, nil)  // err == vecgo.ErrReadOnly
```

*   **Why This Architecture?**
    *   Direct cloud writes via atomic Put operations
    *   Stateless readers scale horizontally (serverless/Kubernetes)
    *   Clean separation of compute and storage
    *   Optional DynamoDB coordination for concurrent writers

### 3. MinIO (Self-Hosted S3)
*   **Best for**: On-prem deployments, air-gapped environments, dev/test
*   **Architecture**: App → MinIO (S3-compatible API)

```go
// MinIO with native client
import (
    "github.com/minio/minio-go/v7"
    "github.com/minio/minio-go/v7/pkg/credentials"
    minioblob "github.com/hupe1980/vecgo/blobstore/minio"
)

client, _ := minio.New("localhost:9000", &minio.Options{
    Creds:  credentials.NewStaticV4("minioadmin", "minioadmin", ""),
    Secure: false,  // Use true for HTTPS
})
store := minioblob.NewStore(client, "my-bucket", "vectors/")
eng, _ := vecgo.Open(ctx, vecgo.Remote(store), vecgo.Create(128, vecgo.MetricL2))
```

*   **Benefits**:
    *   Native MinIO client with optimal performance
    *   Works with any S3-compatible storage (Ceph, Garage, SeaweedFS)
    *   Air-gap friendly (no AWS dependencies)

### 4. S3 Express One Zone (Low Latency)
*   **Best for**: Lambda functions, Kubernetes, real-time inference
*   **Architecture**: App → S3 Express (single-digit millisecond latency)

```go
// S3 Express bucket (must end with --azid--x-s3)
expressStore := s3.NewExpressStore(client, "my-bucket--usw2-az1--x-s3", "vectors/")
eng, _ := vecgo.Open(ctx, vecgo.Remote(expressStore))
```

*   **Benefits**:
    *   Single-digit millisecond latency (vs ~35ms for standard S3)
    *   Supports conditional writes for atomic operations
    *   Ideal for serverless/ephemeral compute

### 5. DynamoDB Commit Store (Concurrent Writers)
*   **Best for**: Multiple writer pods, high-availability write paths
*   **Architecture**: Writers → S3 (data) + DynamoDB (coordination)

```go
// Base S3 store for data
s3Store := s3.NewStore(client, "my-bucket", "vectors/")

// Wrap with DynamoDB commit store for concurrent writes
commitStore := s3.NewDDBCommitStore(s3Store, ddbClient, "vecgo-commits", "s3://my-bucket/vectors/")
eng, _ := vecgo.Open(ctx, vecgo.Remote(commitStore), vecgo.Create(128, vecgo.MetricL2))
```

*   **How it works**:
    1.  Segments are written to S3 (immutable)
    2.  Manifest updates use DynamoDB conditional writes
    3.  Only one writer can commit a new version at a time
    4.  Failed writers retry with the new base version

*   **DynamoDB Table Setup**:
```bash
aws dynamodb create-table \
  --table-name vecgo-commits \
  --attribute-definitions \
    AttributeName=base_uri,AttributeType=S \
    AttributeName=version,AttributeType=N \
  --key-schema \
    AttributeName=base_uri,KeyType=HASH \
    AttributeName=version,KeyType=RANGE \
  --billing-mode PAY_PER_REQUEST
```

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
