package engine

import (
	"testing"

	"github.com/hupe1980/vecgo/core"
	"github.com/stretchr/testify/assert"
)

func TestGlobalID_RoundTrip(t *testing.T) {
	tests := []struct {
		shardIdx uint32
		localID  core.LocalID
	}{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 42},
		{255, 0},                   // Max shard
		{0, core.MaxLocalID - 1},   // Max local ID
		{255, core.MaxLocalID - 1}, // Max both
		{127, 1000000},             // Middle range
	}

	for _, tt := range tests {
		gid := NewGlobalID(tt.shardIdx, tt.localID)
		assert.Equal(t, tt.shardIdx, gid.ShardID(), "shard mismatch for gid=%d", gid)
		assert.Equal(t, tt.localID, gid.LocalID(), "local ID mismatch for gid=%d", gid)
	}
}

func TestGlobalID_BitLayout(t *testing.T) {
	// Verify exact bit layout: [ShardID:32][LocalID:32]
	gid := NewGlobalID(1, 0x42)
	assert.Equal(t, GlobalID(0x0000000100000042), gid)

	gid2 := NewGlobalID(0xFF, 0xABCDEF)
	assert.Equal(t, GlobalID(0x000000FF00ABCDEF), gid2)
}

func BenchmarkGlobalID_Encode(b *testing.B) {
	var i int
	for b.Loop() {
		_ = NewGlobalID(uint32(i%256), core.LocalID(i))
		i++
	}
}

func BenchmarkGlobalID_Decode(b *testing.B) {
	gid := NewGlobalID(127, 1000000)
	for b.Loop() {
		_ = gid.ShardID()
		_ = gid.LocalID()
	}
}
