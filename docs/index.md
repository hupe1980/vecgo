---
layout: default
title: Home
nav_order: 1
permalink: /
---

<div class="hero-section">
  <div class="hero-content">
    <div class="hero-badge">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/>
      </svg>
      Pure Go • Zero CGO • Production Ready
    </div>
    
    <h1 class="hero-title">Vector Search.<br/>Embedded.</h1>
    
    <p class="hero-subtitle">
      High-performance embeddable vector database for Go.<br/>
      No external services. Just import and go.
    </p>
    
    <div class="hero-buttons">
      <a href="{{ '/getting-started/' | relative_url }}" class="btn-hero primary">
        Get Started →
      </a>
      <a href="https://github.com/hupe1980/vecgo" class="btn-hero secondary">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
          <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
        </svg>
        GitHub
      </a>
    </div>
    
    <div class="hero-visual">
      <div class="hero-code">
        <div class="hero-code-header">
          <span class="dot red"></span>
          <span class="dot yellow"></span>
          <span class="dot green"></span>
          <span class="title">main.go</span>
        </div>
        <div class="hero-code-content">
<span class="comment">// Open or create a vector database</span>
db, _ := vecgo.<span class="func">Open</span>(ctx, vecgo.<span class="func">Local</span>(<span class="string">"./data"</span>),
    vecgo.<span class="func">Create</span>(<span class="number">1536</span>, vecgo.MetricCosine),
)

<span class="comment">// Insert and search — it's that simple</span>
db.<span class="func">Insert</span>(ctx, embedding, metadata, payload)
results, _ := db.<span class="func">Search</span>(ctx, query, <span class="number">10</span>)
        </div>
      </div>
    </div>
  </div>
</div>

<div class="stats-section">
  <div class="stat-item">
    <div class="stat-value">2M+</div>
    <div class="stat-label">vectors/sec</div>
  </div>
  <div class="stat-item">
    <div class="stat-value">&lt;1ms</div>
    <div class="stat-label">p99 latency</div>
  </div>
  <div class="stat-item">
    <div class="stat-value">15MB</div>
    <div class="stat-label">binary</div>
  </div>
  <div class="stat-item">
    <div class="stat-value">0</div>
    <div class="stat-label">CGO deps</div>
  </div>
</div>

## Why Vecgo?

<div class="features-grid">
  <div class="feature-card">
    <div class="feature-icon">⚡</div>
    <div class="feature-title">Embedded & Fast</div>
    <div class="feature-desc">No network overhead. Direct memory access with zero-copy vectors and SIMD acceleration.</div>
  </div>
  
  <div class="feature-card">
    <div class="feature-icon">🔧</div>
    <div class="feature-title">Production Ready</div>
    <div class="feature-desc">MVCC, commit-oriented durability, cloud storage, and time-travel queries.</div>
  </div>
  
  <div class="feature-card">
    <div class="feature-icon">🎯</div>
    <div class="feature-title">Pure Go</div>
    <div class="feature-desc">Static binaries, easy cross-compilation. Works everywhere Go works.</div>
  </div>
  
  <div class="feature-card">
    <div class="feature-icon">☁️</div>
    <div class="feature-title">Cloud Native</div>
    <div class="feature-desc">S3/GCS/Azure storage. Separate writer/reader nodes for scaling.</div>
  </div>
  
  <div class="feature-card">
    <div class="feature-icon">🏗️</div>
    <div class="feature-title">Modern Architecture</div>
    <div class="feature-desc">HNSW + DiskANN hybrid indexing. Append-only commits, no WAL.</div>
  </div>
  
  <div class="feature-card">
    <div class="feature-icon">🔀</div>
    <div class="feature-title">Hybrid Search</div>
    <div class="feature-desc">Vector + BM25 keyword search with Reciprocal Rank Fusion.</div>
  </div>
</div>

---

## Quick Start

<div class="install-section">
  <div class="install-title">Installation</div>
  <div class="install-command">
    <code>go get github.com/hupe1980/vecgo</code>
  </div>
</div>

```go
package main

import (
    "context"
    "github.com/hupe1980/vecgo"
    "github.com/hupe1980/vecgo/metadata"
)

func main() {
    ctx := context.Background()

    // Open or create
    db, _ := vecgo.Open(ctx, vecgo.Local("./vectors"),
        vecgo.Create(1536, vecgo.MetricCosine),
    )
    defer db.Close()

    // Insert
    id, _ := db.Insert(ctx, embedding, metadata.Document{
        "category": metadata.String("tech"),
    }, nil)

    // Commit (durable after this)
    db.Commit(ctx)

    // Search with filters
    results, _ := db.Search(ctx, query, 10,
        vecgo.WithFilter(metadata.NewFilterSet(
            metadata.Filter{Key: "category", Operator: metadata.OpEqual, 
                Value: metadata.String("tech")},
        )),
    )
}
```

---

## Index Types

| Index | Description | Use Case |
|-------|-------------|----------|
| **HNSW** | Hierarchical Navigable Small World | In-memory, lock-free search |
| **DiskANN** | Disk-resident with quantization | Large-scale, on-disk |
| **FreshDiskANN** | Streaming updates | Real-time, soft deletion |
| **Flat** | Exact search with SIMD | Small datasets |

---

## Quantization

| Method | RAM Reduction | Recall | Best For |
|--------|---------------|--------|----------|
| **PQ** | 8-64× | 90-95% | High compression |
| **OPQ** | 8-64× | 93-97% | Best recall |
| **SQ8** | 4× | 95-99% | General purpose |
| **BQ** | 32× | 70-85% | Pre-filtering |
| **RaBitQ** | ~30× | 80-90% | Modern BQ |
| **INT4** | 8× | 90-95% | Memory-constrained |

---

<div class="cta-section">
  <h2>Ready to get started?</h2>
  <p>Build your first vector search in under 5 minutes.</p>
  <a href="{{ '/getting-started/' | relative_url }}" class="btn-hero primary">
    Read the Docs →
  </a>
</div>
