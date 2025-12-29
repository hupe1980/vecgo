package engine

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestGlobalID_RoundTrip(t *testing.T) {
	tests := []struct {
		shardIdx int
		localID  uint32
	}{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 42},
		{255, 0},              // Max shard
		{0, MaxLocalID - 1},   // Max local ID
		{255, MaxLocalID - 1}, // Max both
		{127, 1000000},        // Middle range
	}

	for _, tt := range tests {
		gid := NewGlobalID(tt.shardIdx, tt.localID)
		assert.Equal(t, tt.shardIdx, gid.ShardIndex(), "shard mismatch for gid=%d", gid)
		assert.Equal(t, tt.localID, gid.LocalID(), "local ID mismatch for gid=%d", gid)
	}
}

func TestGlobalID_Overflow(t *testing.T) {
	// Local ID should be masked to 24 bits
	gid := NewGlobalID(0, 0xFFFFFFFF)
	assert.Equal(t, uint32(0xFFFFFF), gid.LocalID())
	assert.Equal(t, 0, gid.ShardIndex())
}

func TestGlobalID_IsValid(t *testing.T) {
	gid := NewGlobalID(5, 100)
	assert.True(t, gid.IsValid(10))
	assert.True(t, gid.IsValid(6))
	assert.False(t, gid.IsValid(5))
	assert.False(t, gid.IsValid(1))
}

func TestGlobalID_BitLayout(t *testing.T) {
	// Verify exact bit layout: [ShardID:8][LocalID:24]
	gid := NewGlobalID(1, 0x42)
	assert.Equal(t, GlobalID(0x01000042), gid)

	gid2 := NewGlobalID(0xFF, 0xABCDEF)
	assert.Equal(t, GlobalID(0xFFABCDEF), gid2)
}

func BenchmarkGlobalID_Encode(b *testing.B) {
	var i int
	for b.Loop() {
		_ = NewGlobalID(i%256, uint32(i))
		i++
	}
}

func BenchmarkGlobalID_Decode(b *testing.B) {
	gid := NewGlobalID(127, 1000000)
	for b.Loop() {
		_ = gid.ShardIndex()
		_ = gid.LocalID()
	}
}
