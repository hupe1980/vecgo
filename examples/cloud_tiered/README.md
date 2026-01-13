# Cloud-Tiered Storage Example

This example demonstrates Vecgo's **serverless-ready** architecture with multi-tier caching.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  L1: RAM Cache (4KB blocks, LRU)                                │
│  ├─ Hit: ~1.25µs latency                                        │
│  └─ Miss: Proceed to L2                                         │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  L2: Disk Cache (4MB blocks, Async Write, LRU)                  │
│  ├─ Hit: ~83µs latency                                          │
│  └─ Miss: Proceed to L3                                         │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  L3: Remote Storage (S3, GCS/Azure via BlobStore)               │
│  └─ Cold: ~35µs (simulated 20ms S3 latency)                     │
└─────────────────────────────────────────────────────────────────┘
```

## Usage

### Simple API

```go
import (
    "github.com/hupe1980/vecgo"
    "github.com/hupe1980/vecgo/blobstore/s3"
)

// Create S3-backed blob store
s3Store, _ := s3.New(ctx, "my-bucket", s3.WithPrefix("vectors/"))

// Open with remote backend (read-only for search nodes)
eng, err := vecgo.Open(vecgo.Remote(s3Store),
    vecgo.ReadOnly(),
    vecgo.WithCacheDir("/tmp/cache"),
    vecgo.WithBlockCacheSize(64 * 1024 * 1024),
)
```

### What This Does

1. **Remote backend** stores segment data in S3/cloud
2. **Local cache directory** for block caching
3. **Self-describing index** — dimension/metric loaded from manifest
4. **Read-only mode** — commit-oriented durability, no writes to remote

## Running the Example

```bash
go run main.go
```

### Expected Output

```
🏗️  Building Index locally...
☁️  Uploading blocks to 'S3'...
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

// Writer node (builds index)
s3Store, _ := s3.New(ctx, "my-bucket", s3.WithPrefix("vectors/prod/"))
db, _ := vecgo.Open(vecgo.Remote(s3Store), vecgo.Create(128, vecgo.MetricL2))
// ... insert vectors ...
db.Commit(ctx)
db.Close()

// Reader nodes (stateless search)
db, _ := vecgo.Open(vecgo.Remote(s3Store),
    vecgo.ReadOnly(),
    vecgo.WithCacheDir("/fast/nvme"),
)
```

## Key Benefits

- **Zero-configuration**: Just pass the remote store
- **Self-describing indexes**: No need to remember dimension/metric
- **Automatic cache warming**: Hot data stays in RAM
- **Persistent disk cache**: Survives process restarts
- **Read coalescing**: Sequential reads are batched for S3

## See Also

- [docs/deployment.md](../../docs/deployment.md) - Production deployment guide
- [docs/tuning.md](../../docs/tuning.md) - Performance tuning
- [docs/architecture.md](../../docs/architecture.md) - Architecture deep-dive
