package engine

import "github.com/hupe1980/vecgo/core"

// GlobalID encodes shard routing in high bits for O(1) shard lookup.
//
// Format: [ShardID:32 bits][LocalID:32 bits]
//
// → 4 Billion shards max
// → 4 Billion vectors per shard
// → Infinite total capacity (2^64)
//
// This encoding allows Update/Delete operations to route to the correct shard
// without maintaining external mapping tables.
type GlobalID uint64

const (
	ShardBits = 32
	LocalBits = 32
	MaxShards = 1 << ShardBits
)

// NewGlobalID creates a global ID from shard index and local ID.
func NewGlobalID(shardIdx uint32, localID core.LocalID) GlobalID {
	return GlobalID((uint64(shardIdx) << LocalBits) | uint64(localID))
}

// ShardID extracts the shard index (high 32 bits).
func (g GlobalID) ShardID() uint32 {
	return uint32(g >> LocalBits)
}

// LocalID extracts the local ID within the shard (low 32 bits).
func (g GlobalID) LocalID() core.LocalID {
	return core.LocalID(g) // Implicit masking by casting to uint32
}
