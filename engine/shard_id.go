package engine

// GlobalID encodes shard routing in high bits for O(1) shard lookup.
//
// Format: [ShardID:8 bits][LocalID:56 bits]
//
//	→ 256 shards max
//	→ ~72 Quadrillion vectors per shard (effectively infinite)
//	→ Infinite total capacity
//
// This encoding allows Update/Delete operations to route to the correct shard
// without maintaining external mapping tables.
type GlobalID uint64

const (
	shardBits  = 8
	localBits  = 56
	shardMask  = (1 << shardBits) - 1
	localMask  = (1 << localBits) - 1
	MaxShards  = 1 << shardBits
	MaxLocalID = 1 << localBits
)

// NewGlobalID creates a global ID from shard index and local ID.
//
// Example:
//
//	gid := NewGlobalID(1, 42) // Shard 1, local ID 42
func NewGlobalID(shardIdx int, localID uint64) GlobalID {
	return GlobalID((uint64(shardIdx) << localBits) | (localID & localMask))
}

// ShardIndex extracts the shard index (high 8 bits).
func (g GlobalID) ShardIndex() int {
	return int(g >> localBits)
}

// LocalID extracts the local ID within the shard (low 56 bits).
func (g GlobalID) LocalID() uint64 {
	return uint64(g) & localMask
}

// IsValid returns true if the shard index is within bounds.
func (g GlobalID) IsValid(numShards int) bool {
	return g.ShardIndex() < numShards
}
