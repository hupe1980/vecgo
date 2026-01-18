---
layout: default
title: API Reference
nav_order: 4
has_children: true
permalink: /reference/
---

# API Reference

Complete reference for the Vecgo API.
{: .fs-6 .fw-300 }

---

## Package `vecgo`

The main package for interacting with Vecgo databases.

```go
import "github.com/hupe1980/vecgo"
```

---

## Opening & Closing

### `Open`

Opens or creates a database.

```go
func Open(ctx context.Context, backend Backend, opts ...Option) (*DB, error)
```

**Example:**

```go
db, err := vecgo.Open(ctx, vecgo.NewLocalBackend("./data"),
    vecgo.WithDimension(384),
    vecgo.WithDistance(vecgo.DistanceCosine),
)
if err != nil {
    log.Fatal(err)
}
defer db.Close(ctx)
```

### `Close`

Closes the database, flushing pending writes.

```go
func (db *DB) Close(ctx context.Context) error
```

{: .warning }
Always call `Close()` to ensure data is persisted. Use `defer db.Close(ctx)`.

---

## Write Operations

### `Insert`

Inserts a single vector.

```go
func (db *DB) Insert(ctx context.Context, id string, vector []float32, metadata map[string]any) error
```

**Parameters:**
- `id` - Unique identifier (overwrites if exists)
- `vector` - Float32 slice matching configured dimension
- `metadata` - Optional key-value metadata (can be `nil`)

**Example:**

```go
err := db.Insert(ctx, "doc-123", embedding, map[string]any{
    "title": "Hello World",
    "tags":  []string{"greeting", "demo"},
})
```

### `BatchInsert`

Inserts multiple vectors efficiently.

```go
func (db *DB) BatchInsert(ctx context.Context, vectors []Vector) error
```

**Example:**

```go
vectors := []vecgo.Vector{
    {ID: "doc-1", Values: emb1, Metadata: meta1},
    {ID: "doc-2", Values: emb2, Metadata: meta2},
}
err := db.BatchInsert(ctx, vectors)
```

### `Delete`

Deletes a vector by ID.

```go
func (db *DB) Delete(ctx context.Context, id string) error
```

### `Commit`

Persists pending writes to durable storage.

```go
func (db *DB) Commit(ctx context.Context) error
```

{: .note }
Data is buffered in memory until `Commit()` is called (or auto-commit triggers).

---

## Read Operations

### `Search`

Finds the k nearest neighbors to a query vector.

```go
func (db *DB) Search(ctx context.Context, query []float32, k int, opts ...SearchOption) ([]Result, error)
```

**Parameters:**
- `query` - Query vector (same dimension as database)
- `k` - Number of results to return
- `opts` - Optional search options

**Returns:** Slice of `Result` sorted by similarity (best first).

**Example:**

```go
results, err := db.Search(ctx, queryEmbedding, 10)
for _, r := range results {
    fmt.Printf("ID: %s, Score: %.4f\n", r.ID, r.Score)
}
```

### `Get`

Retrieves a vector by ID.

```go
func (db *DB) Get(ctx context.Context, id string) (*Vector, error)
```

**Returns:** `nil` if not found (no error).

### `Exists`

Checks if a vector exists.

```go
func (db *DB) Exists(ctx context.Context, id string) (bool, error)
```

### `Count`

Returns the total number of vectors.

```go
func (db *DB) Count(ctx context.Context) (int64, error)
```

---

## Search Options

### `WithFilter`

Filters results by metadata.

```go
vecgo.WithFilter(metadata.Eq("category", "tech"))
```

See [Metadata Filters](#metadata-filters) for filter syntax.

### `WithEfSearch`

Overrides HNSW search quality (higher = better recall, slower).

```go
vecgo.WithEfSearch(200)
```

### `WithNProbe`

Overrides IVF probe count (higher = better recall, slower).

```go
vecgo.WithNProbe(64)
```

### `WithMinScore`

Filters results below minimum similarity score.

```go
vecgo.WithMinScore(0.7)
```

### `WithIncludeVectors`

Include/exclude vector values in results.

```go
vecgo.WithIncludeVectors(false)  // Faster, less memory
```

### `WithIncludeMetadata`

Include/exclude metadata in results.

```go
vecgo.WithIncludeMetadata(true)  // Default
```

---

## Types

### `Vector`

```go
type Vector struct {
    ID       string
    Values   []float32
    Metadata map[string]any
}
```

### `Result`

```go
type Result struct {
    ID       string
    Score    float32           // Similarity score
    Vector   []float32         // If WithIncludeVectors(true)
    Metadata map[string]any    // If WithIncludeMetadata(true)
}
```

### `Distance`

```go
type Distance int

const (
    DistanceCosine Distance = iota  // Cosine similarity
    DistanceL2                       // Euclidean distance
    DistanceDot                      // Dot product
)
```

### `IndexType`

```go
type IndexType int

const (
    IndexHNSW    IndexType = iota  // Hierarchical NSW
    IndexIVFFlat                    // Inverted file + flat
    IndexFlat                       // Brute force
)
```

---

## Metadata Filters
{: #metadata-filters }

Import the metadata package:

```go
import "github.com/hupe1980/vecgo/metadata"
```

### Comparison Operators

```go
metadata.Eq("field", value)       // field == value
metadata.Ne("field", value)       // field != value
metadata.Gt("field", value)       // field > value
metadata.Gte("field", value)      // field >= value
metadata.Lt("field", value)       // field < value
metadata.Lte("field", value)      // field <= value
```

### String Operators

```go
metadata.Contains("field", "substr")     // field contains substr
metadata.StartsWith("field", "prefix")   // field starts with prefix
metadata.EndsWith("field", "suffix")     // field ends with suffix
```

### Array Operators

```go
metadata.In("field", value)      // value in field (array)
metadata.NotIn("field", value)   // value not in field
```

### Logical Operators

```go
metadata.And(filter1, filter2, ...)   // All must match
metadata.Or(filter1, filter2, ...)    // Any must match
metadata.Not(filter)                   // Negate filter
```

### Example

```go
filter := metadata.And(
    metadata.Eq("category", "tech"),
    metadata.Gte("year", 2023),
    metadata.Or(
        metadata.Contains("title", "Go"),
        metadata.Contains("title", "Rust"),
    ),
)

results, _ := db.Search(ctx, query, 10, vecgo.WithFilter(filter))
```

---

## Backends

### `LocalBackend`

File-based storage.

```go
backend := vecgo.NewLocalBackend("/path/to/data")
```

### `MemoryBackend`

In-memory storage (testing/ephemeral).

```go
backend := vecgo.NewMemoryBackend()
```

### `S3Backend`

AWS S3 storage.

```go
import "github.com/hupe1980/vecgo/blobstore/s3"

backend := s3.NewBackend(s3Client, "bucket-name", "prefix/")
```

### `MinIOBackend`

MinIO storage.

```go
import "github.com/hupe1980/vecgo/blobstore/minio"

backend := minio.NewBackend(minioClient, "bucket-name", "prefix/")
```

---

## Stats

```go
stats := db.Stats()

fmt.Printf("Vectors: %d\n", stats.VectorCount)
fmt.Printf("Segments: %d\n", stats.SegmentCount)
fmt.Printf("MemTable Usage: %.1f%%\n", stats.MemTableUsage * 100)
fmt.Printf("Cache Hit Rate: %.1f%%\n", stats.CacheHitRate * 100)
```
