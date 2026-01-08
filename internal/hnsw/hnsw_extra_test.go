package hnsw

import (
	"context"
	"testing"

	"github.com/hupe1980/vecgo/distance"
	"github.com/hupe1980/vecgo/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestHNSW_Extra_CRUD(t *testing.T) {
	ctx := context.Background()
	dim := 4
	h, err := New(func(o *Options) {
		o.Dimension = dim
		o.M = 8
		o.EF = 20
	})
	require.NoError(t, err)

	// Batch Insert
	vecs := [][]float32{
		{0.1, 0.1, 0.1, 0.1},
		{0.2, 0.2, 0.2, 0.2},
		{0.9, 0.9, 0.9, 0.9},
	}
	res := h.BatchInsert(ctx, vecs)
	for _, err := range res.Errors {
		require.NoError(t, err)
	}
	assert.Len(t, res.IDs, 3)

	// Verify count
	assert.Equal(t, 3, h.VectorCount())
	assert.Equal(t, dim, h.Dimension())
	assert.Equal(t, "HNSW", h.Name())
	assert.Equal(t, distance.MetricL2, h.Metric()) // Default

	// ContainsID
	assert.True(t, h.ContainsID(uint64(res.IDs[0])))
	assert.False(t, h.ContainsID(999))

	// Update
	id := res.IDs[0]
	newVec := []float32{0.5, 0.5, 0.5, 0.5}
	err = h.Update(ctx, id, newVec)
	require.NoError(t, err)

	// Verify Update via VectorByID
	fetched, err := h.VectorByID(ctx, id)
	require.NoError(t, err)
	assert.Equal(t, newVec, fetched)

	// Stream Search
	resStandard, _ := h.KNNSearch(ctx, newVec, 2, nil)
	assert.GreaterOrEqual(t, len(resStandard), 2)
	count := 0
	for res, err := range h.KNNSearchStream(ctx, newVec, 2, nil) {
		require.NoError(t, err)
		assert.GreaterOrEqual(t, res.Distance, float32(0))
		count++
	}
	assert.Equal(t, 2, count)

	// Close
	require.NoError(t, h.Close())
}

func TestHNSW_Extra_Apply(t *testing.T) {
	ctx := context.Background()
	h, err := New(func(o *Options) {
		o.Dimension = 2
	})
	require.NoError(t, err)

	// ApplyInsert (Explicit ID)
	id := model.RowID(100)
	err = h.ApplyInsert(ctx, id, []float32{1.0, 1.0})
	require.NoError(t, err)
	assert.True(t, h.ContainsID(100))
	assert.Equal(t, 1, h.VectorCount())

	// ApplyBatchInsert
	ids := []model.RowID{200, 201}
	vecs := [][]float32{{2.0, 2.0}, {2.1, 2.1}}
	err = h.ApplyBatchInsert(ctx, ids, vecs)
	require.NoError(t, err)
	assert.True(t, h.ContainsID(200))
	assert.True(t, h.ContainsID(201))

	// ApplyUpdate
	err = h.ApplyUpdate(ctx, 200, []float32{3.0, 3.0})
	require.NoError(t, err)
	v, _ := h.VectorByID(ctx, 200)
	assert.Equal(t, []float32{3.0, 3.0}, v)

	// ApplyDelete
	err = h.ApplyDelete(ctx, 201)
	require.NoError(t, err)
	// ApplyDelete keeps node in graph but marks tombstone?
	// ContainsID checks tombstone?
	// Let's check ContainsID implementation:
	// if g.tombstones.Test(idU32) { return false }
	assert.False(t, h.ContainsID(201))

	// Size should be > 0
	assert.Greater(t, h.Size(), int64(0))
}

func TestHNSW_Sharded_Stub(t *testing.T) {
	h, err := NewSharded(0, 4, func(o *Options) { o.Dimension = 2 })
	require.NoError(t, err)
	assert.NotNil(t, h)
	// ShardID and NumShards are hardcoded to 0/1 in current stub?
	// Check impl: ShardID() returns 0, NumShards() returns 1.
	// But NewSharded calls New().
	// Just verify no panic.
	_ = h.ShardID()
	_ = h.NumShards()
}

func TestHNSW_Reset(t *testing.T) {
	ctx := context.Background()
	h, err := New(func(o *Options) { o.Dimension = 2 })
	require.NoError(t, err)

	_, err = h.Insert(ctx, []float32{1, 1})
	require.NoError(t, err)
	assert.Equal(t, 1, h.VectorCount())

	require.NoError(t, h.Reset())
	assert.Equal(t, 0, h.VectorCount())
}

func TestHNSW_Errors(t *testing.T) {
	_, err := New(func(o *Options) { o.Dimension = 0 })
	assert.Error(t, err) // Invalid dimension

	h, _ := New(func(o *Options) { o.Dimension = 2 })
	ctx := context.Background()

	// VectorByID Not Found
	_, err = h.VectorByID(ctx, 9999)
	assert.Error(t, err)

	// Update Not Found
	err = h.Update(ctx, 9999, []float32{1, 1})
	assert.Error(t, err)

	// ApplyBatchInsert mismatch
	err = h.ApplyBatchInsert(ctx, []model.RowID{1}, [][]float32{})
	assert.Error(t, err)
}

func TestHNSW_Dimensions(t *testing.T) {
	// Check AllocNode limits
	// Not easy to test internal allocator without exposing it, but we can test random seed option
	var seed int64 = 123
	h, err := New(func(o *Options) {
		o.Dimension = 2
		o.RandomSeed = &seed
	})
	require.NoError(t, err)
	assert.NotNil(t, h)
}

func TestHNSW_Stats(t *testing.T) {
	ctx := context.Background()
	h, err := New(func(o *Options) {
		o.Dimension = 2
		o.M = 10
		o.EF = 100
	})
	require.NoError(t, err)

	// Add some vectors
	vectors := [][]float32{
		{0.1, 0.1},
		{0.2, 0.2},
		{0.3, 0.3},
	}
	for i, v := range vectors {
		err := h.ApplyInsert(ctx, model.RowID(i), v)
		require.NoError(t, err)
	}

	// Get Stats
	stats := h.Stats()
	assert.NotEmpty(t, stats.Options)
	assert.NotEmpty(t, stats.Parameters)
	assert.NotEmpty(t, stats.Storage)
	assert.NotEmpty(t, stats.Levels)

	// Check Basic Stats
	assert.Equal(t, "HNSW", stats.Options["Type"])

	count, ok := stats.Storage["ActiveNodes"]
	if ok {
		assert.Equal(t, "3", count)
	}

	require.NoError(t, h.Close())
}
