package integration_test

import (
	"context"
	"strings"
	"testing"

	"github.com/hupe1980/vecgo"
	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestFullLifecycle(t *testing.T) {
	dir := t.TempDir()

	// Use a schema to enforce types (Phase 5 requirement)
	schema := metadata.Schema{
		"tag":     metadata.FieldTypeString,
		"version": metadata.FieldTypeInt,
	}

	opts := []vecgo.Option{
		vecgo.WithSchema(schema),
	}

	// 1. Open
	e, err := vecgo.Open(dir, 2, vecgo.MetricL2, opts...)
	require.NoError(t, err)

	// Context for operations
	ctx := context.Background()

	// Data preparation
	pk1 := vecgo.PKUint64(1)
	vec1 := []float32{1.0, 0.0}
	meta1 := metadata.Document{
		"tag":     metadata.String("v1"),
		"version": metadata.Int(1),
	}
	payload1 := []byte("payload-v1")

	// 2. Insert
	err = e.Insert(pk1, vec1, meta1, payload1)
	require.NoError(t, err)

	// 3. Get (Verify Insert)
	rec, err := e.Get(pk1)
	require.NoError(t, err)
	assert.Equal(t, pk1, rec.PK)
	assert.InDeltaSlice(t, vec1, rec.Vector, 1e-6)
	assert.Equal(t, "v1", rec.Metadata["tag"].StringValue())
	assert.Equal(t, int64(1), rec.Metadata["version"].I64)
	assert.Equal(t, payload1, rec.Payload)

	// 4. Search (Visible)
	// We need to wait for eventual consistency if async WAL?
	// But R1 says "after Insert returns, subsequent reads must observe changes" (Monotonic Visibility).
	// So we expect immediate visibility.
	res, err := e.Search(ctx, []float32{1.0, 0.0}, 1, func(so *model.SearchOptions) {
		so.IncludeVector = true
		so.IncludeMetadata = true
		so.IncludePayload = true
	})
	require.NoError(t, err)
	require.Len(t, res, 1)
	assert.Equal(t, pk1, res[0].PK)
	assert.InDeltaSlice(t, vec1, res[0].Vector, 1e-6)

	// 5. Update (Upsert) - Change Vector, Metadata, Payload
	vec2 := []float32{0.0, 1.0}
	meta2 := metadata.Document{
		"tag":     metadata.String("v2"),
		"version": metadata.Int(2),
	}
	payload2 := []byte("payload-v2")

	err = e.Insert(pk1, vec2, meta2, payload2)
	require.NoError(t, err)

	// 6. Get (Verify Update)
	rec, err = e.Get(pk1)
	require.NoError(t, err)
	assert.InDeltaSlice(t, vec2, rec.Vector, 1e-6) // Use InDeltaSlice for floats
	assert.Equal(t, "v2", rec.Metadata["tag"].StringValue())
	assert.Equal(t, int64(2), rec.Metadata["version"].I64)
	assert.Equal(t, payload2, rec.Payload)

	// 7. Search (Verify Index Update)
	// Searching for old vector should be distant, new vector should be close
	res, err = e.Search(ctx, vec2, 1, func(so *model.SearchOptions) {
		so.IncludeVector = true
	})
	require.NoError(t, err)
	require.Len(t, res, 1)
	assert.Equal(t, pk1, res[0].PK)
	assert.InDeltaSlice(t, vec2, res[0].Vector, 1e-6)

	// 8. Filtered Search (Metadata)
	// Should match tag="v2"
	filterV2 := metadata.Filter{
		Key:      "tag",
		Operator: metadata.OpEqual,
		Value:    metadata.String("v2"),
	}
	res, err = e.Search(ctx, vec2, 1, func(so *model.SearchOptions) {
		so.Filter = metadata.NewFilterSet(filterV2)
	})
	require.NoError(t, err)
	require.Len(t, res, 1)

	// Should NOT match tag="v1" (old value)
	filterV1 := metadata.Filter{
		Key:      "tag",
		Operator: metadata.OpEqual,
		Value:    metadata.String("v1"),
	}
	res, err = e.Search(ctx, vec2, 1, func(so *model.SearchOptions) {
		so.Filter = metadata.NewFilterSet(filterV1)
	})
	require.NoError(t, err)
	assert.Len(t, res, 0)

	// 9. Delete
	err = e.Delete(pk1)
	require.NoError(t, err)

	// 10. Get (Verify Delete)
	_, err = e.Get(pk1)
	assert.Error(t, err)
	// Currently engine returns ErrInvalidArgument for missing PK
	assert.True(t, strings.Contains(err.Error(), "invalid argument") || strings.Contains(err.Error(), "not found"),
		"Expected NotFound or InvalidArgument error, got %v", err)

	// 11. Search (Verify Delete)
	res, err = e.Search(ctx, vec2, 1)
	require.NoError(t, err)
	assert.Len(t, res, 0)

	// 12. Persistence (Restart)
	err = e.Close()
	require.NoError(t, err)

	e, err = vecgo.Open(dir, 2, vecgo.MetricL2, opts...)
	require.NoError(t, err)
	defer e.Close()

	// Verify still deleted
	_, err = e.Get(pk1)
	assert.Error(t, err)

	res, err = e.Search(ctx, vec2, 1)
	require.NoError(t, err)
	assert.Len(t, res, 0)
}

func TestBatchCRUD(t *testing.T) {
	dir := t.TempDir()
	e, err := vecgo.Open(dir, 4, vecgo.MetricL2)
	require.NoError(t, err)
	defer e.Close()

	ctx := context.Background()

	// 1. Batch Insert
	count := 10
	records := make([]vecgo.Record, count)
	for i := 0; i < count; i++ {
		records[i] = vecgo.Record{
			PK:     vecgo.PKUint64(uint64(i)),
			Vector: []float32{float32(i), 0, 0, 0},
			// Metadata/Payload optional
		}
	}

	err = e.BatchInsert(records)
	require.NoError(t, err)

	// Verify all present
	for i := 0; i < count; i++ {
		rec, err := e.Get(vecgo.PKUint64(uint64(i)))
		require.NoError(t, err)
		assert.Equal(t, float32(i), rec.Vector[0])
	}

	// 2. Batch Delete (Evens)
	pksToDelete := []vecgo.PrimaryKey{}
	for i := 0; i < count; i += 2 {
		pksToDelete = append(pksToDelete, vecgo.PKUint64(uint64(i)))
	}

	err = e.BatchDelete(pksToDelete)
	require.NoError(t, err)

	// Verify
	for i := 0; i < count; i++ {
		pk := vecgo.PKUint64(uint64(i))
		rec, err := e.Get(pk)
		if i%2 == 0 {
			// Deleted
			assert.Error(t, err, "PK %d should be deleted", i)
		} else {
			// Present
			require.NoError(t, err, "PK %d should exist", i)
			assert.Equal(t, float32(i), rec.Vector[0])
		}
	}

	// 3. Search (Verify only odds return)
	// Searching near 0 (which was deleted) should find 1 or others
	res, err := e.Search(ctx, []float32{0, 0, 0, 0}, 5)
	require.NoError(t, err)

	for _, cand := range res {
		u, ok := cand.PK.Uint64()
		require.True(t, ok)
		assert.NotEqual(t, uint64(0), u, "Deleted PK 0 should not be returned")
		assert.True(t, u%2 != 0, "Only odd PKs should be returned, got %d", u)
	}
}
