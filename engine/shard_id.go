package engine

// GlobalID encodes shard routing in high bits for O(1) shard lookup.
//
// Format: [ShardID:8 bits][LocalID:24 bits]
//
//	→ 256 shards max
//	→ 16M vectors per shard
//	→ 4 billion total vectors
//
// This encoding allows Update/Delete operations to route to the correct shard
// without maintaining external mapping tables.
type GlobalID uint32

const (
	shardBits  = 8
	localBits  = 24
	shardMask  = (1 << shardBits) - 1 // 0xFF
	localMask  = (1 << localBits) - 1 // 0xFFFFFF
	MaxShards  = 1 << shardBits       // 256
	MaxLocalID = 1 << localBits       // 16,777,216
)

// NewGlobalID creates a global ID from shard index and local ID.
//
// Example:
//
//	gid := NewGlobalID(1, 42) // Shard 1, local ID 42 → 0x01_00002A
func NewGlobalID(shardIdx int, localID uint32) GlobalID {
	return GlobalID((uint32(shardIdx) << localBits) | (localID & localMask))
}

// ShardIndex extracts the shard index (high 8 bits).
func (g GlobalID) ShardIndex() int {
	return int(g >> localBits)
}

// LocalID extracts the local ID within the shard (low 24 bits).
func (g GlobalID) LocalID() uint32 {
	return uint32(g) & localMask
}

// IsValid returns true if the shard index is within bounds.
func (g GlobalID) IsValid(numShards int) bool {
	return g.ShardIndex() < numShards
}
