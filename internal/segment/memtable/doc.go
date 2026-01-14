// Package memtable implements the in-memory L0 segment with HNSW indexing.
//
// # Architecture
//
// MemTable is the write buffer for the LSM-tree. It provides mutable storage
// for vectors before they are flushed to immutable disk segments.
//
//	┌─────────────────────────────────────────────────────────────────────────┐
//	│                              MemTable                                   │
//	│  ┌─────────────────────────────────────────────────────────────────┐    │
//	│  │                     16-way Sharding                             │    │
//	│  │   ID % 16 → Shard Selection (avoid lock contention)             │    │
//	│  └─────────────────────────────────────────────────────────────────┘    │
//	│                                                                         │
//	│  ┌──────────┐ ┌──────────┐ ┌──────────┐         ┌──────────┐           │
//	│  │ Shard 0  │ │ Shard 1  │ │ Shard 2  │   ...   │ Shard 15 │           │
//	│  │┌────────┐│ │┌────────┐│ │┌────────┐│         │┌────────┐│           │
//	│  ││ HNSW   ││ ││ HNSW   ││ ││ HNSW   ││         ││ HNSW   ││           │
//	│  ││M=32    ││ ││M=32    ││ ││M=32    ││         ││M=32    ││           │
//	│  ││EF=300  ││ ││EF=300  ││ ││EF=300  ││         ││EF=300  ││           │
//	│  │└────────┘│ │└────────┘│ │└────────┘│         │└────────┘│           │
//	│  │┌────────┐│ │┌────────┐│ │┌────────┐│         │┌────────┐│           │
//	│  ││Vectors ││ ││Vectors ││ ││Vectors ││         ││Vectors ││           │
//	│  ││ Store  ││ ││ Store  ││ ││ Store  ││         ││ Store  ││           │
//	│  │└────────┘│ │└────────┘│ │└────────┘│         │└────────┘│           │
//	│  │┌────────┐│ │┌────────┐│ │┌────────┐│         │┌────────┐│           │
//	│  ││Paged   ││ ││Paged   ││ ││Paged   ││         ││Paged   ││           │
//	│  ││ Stores ││ ││ Stores ││ ││ Stores ││         ││ Stores ││           │
//	│  │└────────┘│ │└────────┘│ │└────────┘│         │└────────┘│           │
//	│  │┌────────┐│ │┌────────┐│ │┌────────┐│         │┌────────┐│           │
//	│  ││Columnar││ ││Columnar││ ││Columnar││         ││Columnar││           │
//	│  ││Metadata││ ││Metadata││ ││Metadata││         ││Metadata││           │
//	│  │└────────┘│ │└────────┘│ │└────────┘│         │└────────┘│           │
//	│  └──────────┘ └──────────┘ └──────────┘         └──────────┘           │
//	└─────────────────────────────────────────────────────────────────────────┘
//
// # Global RowID Encoding
//
// RowIDs encode both shard index and local offset:
//
//	┌───────────────────────────────────────┐
//	│           32-bit Global RowID         │
//	├────────────┬──────────────────────────┤
//	│  4 bits    │       28 bits            │
//	│ Shard Idx  │    Local RowID           │
//	│  (0-15)    │   (0-268M per shard)     │
//	└────────────┴──────────────────────────┘
//
// # Storage Components
//
// Each shard maintains:
//
//	HNSW Index       - Navigable small-world graph (M=32, EF=300)
//	VectorStore      - Arena-allocated float32 vectors
//	PagedIDStore     - External IDs in 64K-entry pages
//	PagedMetaStore   - Interned metadata documents in pages
//	PagedPayloadStore - Binary payloads in pages
//	Columns          - Columnar storage for fast metadata filtering
//
// # Columnar Metadata
//
// Metadata is stored both row-oriented (PagedMetaStore for retrieval) and
// columnar (columns map for filtering). Supported column types:
//
//	intColumn    - int64 values with validity bitmap
//	floatColumn  - float64 values with int promotion
//	stringColumn - interned strings via unique.Handle
//	boolColumn   - boolean values
//
// # Search Flow
//
//  1. Search iterates all shards sequentially (goroutine overhead avoided)
//  2. Each shard runs HNSW KNN search with columnar filter wrapper
//  3. Results pushed to shared searcher heap with global RowID encoding
//  4. Top-K maintained across all shards
//
// # Lifecycle
//
// MemTables use reference counting for lifecycle management:
//
//  1. Create  - New(ctx, id, dim, metric, acquirer) with refs=1
//  2. Use     - IncRef() before use, DecRef() after
//  3. Flush   - Iterate() exports all data to disk segment
//  4. Close   - DecRef() to 0 triggers resource cleanup
//
// # Concurrency
//
//	Writes  - Per-shard mutex (16 independent write paths)
//	Reads   - RWMutex allows concurrent readers per shard
//	Search  - Lock-free HNSW search with shared heap
//	Rerank  - Sync.Pool for dispatch slice allocation
//
// # Memory Budget
//
// Initial memory allocation per MemTable:
//
//	16 shards × 256KB arena = 4MB base allocation
//	+ Paged stores grow on demand (64K entries per page)
//	+ Columnar storage grows with unique keys
package memtable
