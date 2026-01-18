---
layout: default
title: Getting Started
nav_order: 2
has_children: true
permalink: /getting-started/
---

# Getting Started

Learn how to install Vecgo and build your first vector search application.
{: .fs-6 .fw-300 }

---

## Installation

Add Vecgo to your Go project:

```bash
go get github.com/hupe1980/vecgo
```

### Requirements

- Go 1.21 or later
- Linux, macOS, or Windows
- For optimal performance: CPU with AVX2/AVX-512 (x86) or NEON (ARM)

---

## Your First Database

### 1. Create a Database

```go
package main

import (
    "context"
    "github.com/hupe1980/vecgo"
)

func main() {
    ctx := context.Background()

    // Open database with local file storage
    db, err := vecgo.Open(ctx, vecgo.NewLocalBackend("./my-vectors"),
        vecgo.WithDimension(384),              // Vector dimension
        vecgo.WithDistance(vecgo.DistanceCosine), // Similarity metric
    )
    if err != nil {
        panic(err)
    }
    defer db.Close(ctx)
}
```

### 2. Insert Vectors

```go
// Single insert
err := db.Insert(ctx, "doc-1", embedding, map[string]any{
    "title": "Introduction to Vector Databases",
    "category": "tutorial",
})

// Batch insert for better performance
vectors := []vecgo.Vector{
    {ID: "doc-2", Values: embedding2, Metadata: meta2},
    {ID: "doc-3", Values: embedding3, Metadata: meta3},
}
err := db.BatchInsert(ctx, vectors)

// Commit to persist to disk
err := db.Commit(ctx)
```

### 3. Search

```go
// Basic search - find 10 nearest neighbors
results, err := db.Search(ctx, queryEmbedding, 10)

for _, result := range results {
    fmt.Printf("ID: %s, Score: %.4f\n", result.ID, result.Score)
}
```

### 4. Filtered Search

```go
// Search with metadata filter
results, err := db.Search(ctx, queryEmbedding, 10,
    vecgo.WithFilter(
        metadata.And(
            metadata.Eq("category", "tutorial"),
            metadata.Gte("year", 2024),
        ),
    ),
)
```

---

## Complete Example

Here's a full working example:

```go
package main

import (
    "context"
    "fmt"
    "math/rand"

    "github.com/hupe1980/vecgo"
    "github.com/hupe1980/vecgo/metadata"
)

func main() {
    ctx := context.Background()

    // Create database
    db, _ := vecgo.Open(ctx, vecgo.NewLocalBackend("./demo"),
        vecgo.WithDimension(128),
        vecgo.WithDistance(vecgo.DistanceCosine),
    )
    defer db.Close(ctx)

    // Insert sample vectors
    categories := []string{"tech", "science", "art"}
    for i := 0; i < 1000; i++ {
        vec := randomVector(128)
        meta := map[string]any{
            "category": categories[i%3],
            "index":    i,
        }
        db.Insert(ctx, fmt.Sprintf("doc-%d", i), vec, meta)
    }
    db.Commit(ctx)

    // Search
    query := randomVector(128)
    results, _ := db.Search(ctx, query, 5,
        vecgo.WithFilter(metadata.Eq("category", "tech")),
    )

    fmt.Println("Top 5 results in 'tech' category:")
    for _, r := range results {
        fmt.Printf("  %s (score: %.4f)\n", r.ID, r.Score)
    }
}

func randomVector(dim int) []float32 {
    vec := make([]float32, dim)
    for i := range vec {
        vec[i] = rand.Float32()
    }
    return vec
}
```

---

## Next Steps

- [Core Concepts](concepts/) - Understand how Vecgo works
- [Configuration](../guides/configuration/) - All available options
- [Tuning Guide](../guides/tuning/) - Optimize for your workload
