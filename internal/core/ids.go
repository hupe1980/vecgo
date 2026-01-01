package core

// LocalID is a dense, internal identifier for a vector.
// It is used for all hot-path structures (graph adjacency, bitsets, heaps).
// Scope: Local to a single Segment (Shard or DiskANN file).
type LocalID = uint32

// GlobalID is a stable, durable, user-facing identifier.
// It is used for routing, API results, and WAL entries.
// Invariant: Never reused. Tombstones are permanent until compaction.
type GlobalID = uint64
