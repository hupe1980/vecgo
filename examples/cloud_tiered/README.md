# Cloud-Tiered Storage Example

This example demonstrates Vecgo's **cloud-native** architecture with direct cloud writes.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Writer Node(s)                                                 │
│  ├─ vecgo.Open(ctx, vecgo.Remote(s3Store))                      │
│  ├─ Direct writes to S3 (no local sync needed)                  │
│  └─ Optional: DynamoDB commit store for concurrent writers      │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Object Storage                                                 │
│  ├─ S3 Standard          — cost-effective, ~35ms latency        │
│  ├─ S3 Express One Zone  — low-latency (~1ms), ideal for Lambda │
│  ├─ MANIFEST-*.bin       — version pointers                     │
│  ├─ segment_*.bin        — vector data                          │
│  └─ pkindex_*.bin        — primary key index checkpoint         │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Reader Nodes (Stateless, Horizontally Scalable)                │
│  ├─ vecgo.Open(ctx, vecgo.Remote(store), vecgo.ReadOnly())      │
│  ├─ L1: RAM Cache (4KB blocks, LRU) — ~1µs hit                  │
│  ├─ L2: Disk Cache (4MB blocks, async) — ~80µs hit              │
│  └─ L3: Remote Storage — ~1-35ms (depends on storage class)     │
└─────────────────────────────────────────────────────────────────┘
```

## Usage

### Writer Node (Direct to S3)

Writers can write directly to S3 without local filesystem:

```go
// Build index DIRECTLY to S3 (no local sync needed!)
s3Store, _ := s3.New(ctx, "my-bucket", s3.WithPrefix("vectors/"))
eng, _ := vecgo.Open(ctx, vecgo.Remote(s3Store), vecgo.Create(128, vecgo.MetricL2))
eng.Insert(ctx, vector, metadata, nil)
eng.Close()  // Segments are already in S3
```

For concurrent writers, use DynamoDB commit store (see below).

### Reader Nodes (Stateless, Horizontally Scalable)

```go
import (
    "github.com/hupe1980/vecgo"
    "github.com/hupe1980/vecgo/blobstore/s3"
)

// Open from S3 with explicit ReadOnly for stateless search nodes
s3Store, _ := s3.New(ctx, "my-bucket", s3.WithPrefix("vecgo/"))
eng, err := vecgo.Open(ctx, vecgo.Remote(s3Store),
    vecgo.ReadOnly(),                       // Stateless read-only for safety
    vecgo.WithCacheDir("/tmp/cache"),
    vecgo.WithBlockCacheSize(4 << 30),  // 4GB
)

// Search works
results, _ := eng.Search(ctx, query, 10)

// Writes return ErrReadOnly
_, err = eng.Insert(ctx, vec, nil, nil)  // err == vecgo.ErrReadOnly
```

### Why This Architecture?

1. **True cloud-native** — no local filesystem required for writers
2. **Stateless readers** — scale horizontally (Kubernetes, serverless)
3. **Direct S3 writes** — no sync step needed
4. **Optional local cache** — for hot data acceleration
5. **MinIO compatible** — works with any S3-compatible storage

## Running the Example

```bash
go run main.go
```

### Expected Output

```
🏗️  Building Index directly to 'S3'...
✅ Built index directly to cloud store!
🚀 Starting Stateless Search Node...
⏱️  Engine Open Time: ~Xms
✅ Write correctly rejected in read-only mode

🔎 Executing Query 1 (Cold Cache)...
   Cold Query Latency: ~Xms

🔎 Executing Query 2 (Warm Cache)...
   Warm Query Latency: ~Xµs
```

## Production Deployment

For real S3 deployment:

```go
import "github.com/hupe1980/vecgo/blobstore/s3"

// === Writer node (direct to S3) ===
s3Store, _ := s3.New(ctx, "my-bucket", s3.WithPrefix("vectors/prod/"))
db, _ := vecgo.Open(ctx, vecgo.Remote(s3Store), vecgo.Create(128, vecgo.MetricL2))
// ... insert vectors ...
db.Close()

// === Reader nodes (stateless from S3) ===
s3Store, _ := s3.New(ctx, "my-bucket", s3.WithPrefix("vectors/prod/"))
db, _ := vecgo.Open(ctx, vecgo.Remote(s3Store),
    vecgo.ReadOnly(),                     // Explicitly read-only for safety
    vecgo.WithCacheDir("/fast/nvme"),
)
```

## S3 Express One Zone (Low Latency)

For Lambda, Kubernetes, or latency-sensitive workloads, use S3 Express One Zone:

```go
import "github.com/hupe1980/vecgo/blobstore/s3"

// S3 Express bucket (must end with --azid--x-s3)
expressStore := s3.NewExpressStore(s3Client, "my-bucket--usw2-az1--x-s3", "vectors/")
db, _ := vecgo.Open(ctx, vecgo.Remote(expressStore))
```

**Benefits:**
- Single-digit millisecond latency (vs ~35ms for standard S3)
- Supports conditional writes for atomic operations
- Ideal for serverless/ephemeral compute

## DynamoDB Commit Store (Concurrent Writers)

S3 lacks atomic writes. For safe concurrent writers (e.g., multiple pods updating the same index), use DynamoDB as a commit store:

```go
import (
    "github.com/aws/aws-sdk-go-v2/config"
    "github.com/aws/aws-sdk-go-v2/service/dynamodb"
    "github.com/aws/aws-sdk-go-v2/service/s3"
    vecgos3 "github.com/hupe1980/vecgo/blobstore/s3"
)

cfg, _ := config.LoadDefaultConfig(ctx)
s3Client := s3.NewFromConfig(cfg)
ddbClient := dynamodb.NewFromConfig(cfg)

// Create base S3 store
s3Store := vecgos3.NewStore(s3Client, "my-bucket", "vectors/")

// Wrap with DynamoDB commit store for concurrent writes
commitStore := vecgos3.NewDDBCommitStore(
    s3Store, 
    ddbClient, 
    "vecgo-commits",           // DynamoDB table name
    "s3://my-bucket/vectors/", // Base URI as partition key
)

db, _ := vecgo.Open(ctx, vecgo.Remote(commitStore), vecgo.Create(128, vecgo.MetricL2))
```

**DynamoDB Table Setup:**

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

**How it works:**
1. Segments are written to S3 (immutable)
2. Manifest updates use DynamoDB conditional writes
3. Only one writer can commit a new version at a time
4. Failed writers retry with the new base version

## MinIO (Self-Hosted S3)

For on-premise or air-gapped deployments, use the native MinIO client:

```go
import (
    "github.com/minio/minio-go/v7"
    "github.com/minio/minio-go/v7/pkg/credentials"
    minioblob "github.com/hupe1980/vecgo/blobstore/minio"
)

client, _ := minio.New("localhost:9000", &minio.Options{
    Creds:  credentials.NewStaticV4("minioadmin", "minioadmin", ""),
    Secure: false,  // Use true for HTTPS in production
})

store := minioblob.NewStore(client, "my-bucket", "vectors/")
db, _ := vecgo.Open(ctx, vecgo.Remote(store), vecgo.Create(128, vecgo.MetricL2))
```

**Benefits:**
- Native MinIO client with optimal performance
- Works with any S3-compatible storage (Ceph, Garage, SeaweedFS)
- Air-gap friendly (no AWS dependencies required)

## Key Benefits

- **Zero-configuration**: Just pass the remote store
- **Self-describing indexes**: No need to remember dimension/metric
- **Automatic cache warming**: Hot data stays in RAM
- **Persistent disk cache**: Survives process restarts
- **Read coalescing**: Sequential reads are batched for S3
- **Concurrent-safe**: Optional DynamoDB commit store for multi-writer scenarios
- **Low-latency option**: S3 Express One Zone for serverless workloads

## See Also

- [docs/deployment.md](../../docs/deployment.md) - Production deployment guide
- [docs/tuning.md](../../docs/tuning.md) - Performance tuning
- [docs/architecture.md](../../docs/architecture.md) - Architecture deep-dive
