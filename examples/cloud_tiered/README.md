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

### Simple API (LanceDB-style)

```go
// Create S3-backed blob store
s3Store, _ := s3.New(ctx, "my-bucket", s3.WithPrefix("vectors/"))

// Open with auto-configuration
eng, err := engine.OpenCloud("/tmp/cache",
    engine.WithRemoteStore(s3Store),
    engine.WithBlockCacheSize(64 * 1024 * 1024),
)
```

### What `OpenCloud` Does

1. **Sets BlobStore** for segment data automatically
2. **Sets ManifestStore** for metadata (via BlobStoreAdapter)
3. **Uses scratch directory** for local caching
4. **Loads dimension/metric** from persisted manifest (self-describing index)
5. **Enables read-optimized mode** (no WAL, no writes to remote)

## Running the Example

```bash
go run main.go
```

### Expected Output

```
=== Cloud-Tiered Caching Demo ===

Ingesting 1000 vectors...
✓ Ingested 1000 vectors in Xms

Flushing to remote storage...
✓ Flushed to remote storage

--- Search Latency Tests ---

Cold search (cache empty):
  Search latency: ~35µs

Warm search (RAM cache hit):
  Search latency: ~1.25µs

Disk cache search (after simulated restart):
  Search latency: ~83µs
```

## Production Deployment

For real S3 deployment, replace the simulated store:

```go
import "github.com/hupe1980/vecgo/blobstore/s3"

s3Store, err := s3.New(ctx, "my-bucket",
    s3.WithPrefix("vectors/prod/"),
    s3.WithRegion("us-east-1"),
)
```

## Key Benefits

- **Zero-configuration**: Just pass the remote store
- **Self-describing indexes**: No need to remember dimension/metric
- **Automatic cache warming**: Hot data stays in RAM
- **Persistent disk cache**: Survives process restarts
- **Read coalescing**: Sequential reads are batched for S3

## See Also

- [FINDINGS.md](../../FINDINGS.md) - Full architecture analysis
- [docs/deployment.md](../../docs/deployment.md) - Production deployment guide
- [docs/tuning.md](../../docs/tuning.md) - Performance tuning
