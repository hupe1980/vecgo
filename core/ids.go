package core

// LocalID is a dense, internal identifier for a vector within a single shard.
// It is strictly 32-bit, allowing for max 4 Billion vectors per shard.
// Used for all hot-path structures (graph adjacency, bitsets, heaps).
type LocalID uint32

// MaxLocalID is the maximum possible value for a LocalID.
const MaxLocalID = ^LocalID(0)
