package engine

import (
	"context"
	"testing"

	"github.com/hupe1980/vecgo/codec"
	"github.com/hupe1980/vecgo/index"
	"github.com/hupe1980/vecgo/index/flat"
	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/wal"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// createTestCoordinator creates a coordinator for testing metadata safety.
func createTestCoordinator[T any](t *testing.T) Coordinator[T] {
	t.Helper()

	idx, err := flat.New(func(o *flat.Options) {
		o.Dimension = 2
		o.DistanceType = index.DistanceTypeSquaredL2
	})
	require.NoError(t, err)

	dataStore := NewMapStore[T]()
	metaStore := metadata.NewUnifiedIndex()

	walLog, err := wal.New(func(o *wal.Options) {
		o.Path = t.TempDir()
		o.Sync = false
	})
	require.NoError(t, err)
	t.Cleanup(func() { walLog.Close() })

	coord, err := New(idx, dataStore, metaStore, walLog, codec.Default)
	require.NoError(t, err)

	return coord
}

// TestMetadataMutationProtection verifies that external mutation of metadata
// after Insert/Update does not affect the stored metadata.
func TestMetadataMutationProtection(t *testing.T) {
	t.Run("Insert metadata mutation", func(t *testing.T) {
		ctx := context.Background()
		coord := createTestCoordinator[string](t)

		// Create metadata with mutable content
		originalMeta := metadata.Metadata{
			"tag":   metadata.String("production"),
			"count": metadata.Int(42),
			"values": metadata.Array([]metadata.Value{
				metadata.Int(1),
				metadata.Int(2),
				metadata.Int(3),
			}),
		}

		// Insert with metadata
		id, err := coord.Insert(ctx, []float32{0.1, 0.2}, "data1", originalMeta)
		require.NoError(t, err)

		// Mutate the original metadata AFTER insert
		originalMeta["tag"] = metadata.String("development")
		originalMeta["count"] = metadata.Int(999)
		originalMeta["extra"] = metadata.String("should not appear")
		
		// Mutate array values
		if arr, ok := originalMeta["values"]; ok && arr.Kind == metadata.KindArray {
			arr.A[0] = metadata.Int(999)
		}

		// Retrieve metadata and verify it's unchanged
		meta, ok := coord.GetMetadata(id)
		require.True(t, ok, "metadata should exist")

		// Verify stored metadata is unchanged
		assert.Equal(t, "production", meta["tag"].S, "tag should not be mutated")
		assert.Equal(t, int64(42), meta["count"].I64, "count should not be mutated")
		assert.NotContains(t, meta, "extra", "extra field should not appear")

		// Verify array is unchanged
		arr := meta["values"]
		require.Equal(t, metadata.KindArray, arr.Kind)
		require.Len(t, arr.A, 3)
		assert.Equal(t, int64(1), arr.A[0].I64, "array should not be mutated")
		assert.Equal(t, int64(2), arr.A[1].I64)
		assert.Equal(t, int64(3), arr.A[2].I64)
	})

	t.Run("BatchInsert metadata mutation", func(t *testing.T) {
		ctx := context.Background()
		coord := createTestCoordinator[string](t)

		// Create metadata slice with mutable content
		metadataSlice := []metadata.Metadata{
			{
				"batch": metadata.Int(1),
				"nested": metadata.Array([]metadata.Value{
					metadata.String("a"),
					metadata.String("b"),
				}),
			},
			{
				"batch": metadata.Int(2),
				"nested": metadata.Array([]metadata.Value{
					metadata.String("c"),
					metadata.String("d"),
				}),
			},
		}

		// Batch insert
		ids, err := coord.BatchInsert(
			ctx,
			[][]float32{{0.1, 0.2}, {0.3, 0.4}},
			[]string{"data1", "data2"},
			metadataSlice,
		)
		require.NoError(t, err)
		require.Len(t, ids, 2)

		// Mutate original metadata slice
		metadataSlice[0]["batch"] = metadata.Int(999)
		metadataSlice[1]["batch"] = metadata.Int(888)
		
		// Mutate nested arrays
		metadataSlice[0]["nested"].A[0] = metadata.String("MUTATED")
		metadataSlice[1]["nested"].A[1] = metadata.String("MUTATED")

		// Verify first document
		meta1, ok := coord.GetMetadata(ids[0])
		require.True(t, ok)
		assert.Equal(t, int64(1), meta1["batch"].I64, "batch 1 should not be mutated")
		assert.Equal(t, "a", meta1["nested"].A[0].S, "nested array should not be mutated")
		assert.Equal(t, "b", meta1["nested"].A[1].S)

		// Verify second document
		meta2, ok := coord.GetMetadata(ids[1])
		require.True(t, ok)
		assert.Equal(t, int64(2), meta2["batch"].I64, "batch 2 should not be mutated")
		assert.Equal(t, "c", meta2["nested"].A[0].S)
		assert.Equal(t, "d", meta2["nested"].A[1].S, "nested array should not be mutated")
	})

	t.Run("Update metadata mutation", func(t *testing.T) {
		ctx := context.Background()
		coord := createTestCoordinator[string](t)

		// Insert initial document
		id, err := coord.Insert(ctx, []float32{0.1, 0.2}, "data1", metadata.Metadata{
			"version": metadata.Int(1),
		})
		require.NoError(t, err)

		// Update with new metadata
		updateMeta := metadata.Metadata{
			"version": metadata.Int(2),
			"status":  metadata.String("active"),
			"tags": metadata.Array([]metadata.Value{
				metadata.String("tag1"),
				metadata.String("tag2"),
			}),
		}

		err = coord.Update(ctx, id, []float32{0.3, 0.4}, "data2", updateMeta)
		require.NoError(t, err)

		// Mutate the update metadata
		updateMeta["version"] = metadata.Int(999)
		updateMeta["status"] = metadata.String("MUTATED")
		updateMeta["tags"].A[0] = metadata.String("MUTATED")

		// Verify stored metadata is unchanged
		meta, ok := coord.GetMetadata(id)
		require.True(t, ok)
		assert.Equal(t, int64(2), meta["version"].I64, "version should not be mutated")
		assert.Equal(t, "active", meta["status"].S, "status should not be mutated")
		assert.Equal(t, "tag1", meta["tags"].A[0].S, "tags array should not be mutated")
		assert.Equal(t, "tag2", meta["tags"].A[1].S)
	})
}

// TestMetadataCloneNil verifies that nil and empty metadata are handled efficiently.
func TestMetadataCloneNil(t *testing.T) {
	ctx := context.Background()
	coord := createTestCoordinator[string](t)

	t.Run("nil metadata", func(t *testing.T) {
		// Insert with nil metadata
		id, err := coord.Insert(ctx, []float32{0.1, 0.2}, "data1", nil)
		require.NoError(t, err)

		// Verify no metadata stored
		_, ok := coord.GetMetadata(id)
		assert.False(t, ok, "nil metadata should not be stored")
	})

	t.Run("empty metadata", func(t *testing.T) {
		// Insert with empty metadata map
		id, err := coord.Insert(ctx, []float32{0.3, 0.4}, "data2", metadata.Metadata{})
		require.NoError(t, err)

		// Verify no metadata stored (CloneIfNeeded optimization)
		_, ok := coord.GetMetadata(id)
		assert.False(t, ok, "empty metadata should not be stored due to CloneIfNeeded optimization")
	})
}

// TestMetadataCloneDeep verifies deep copying of nested structures.
func TestMetadataCloneDeep(t *testing.T) {
	ctx := context.Background()
	coord := createTestCoordinator[string](t)

	// Create deeply nested metadata
	deepMeta := metadata.Metadata{
		"level1": metadata.Array([]metadata.Value{
			metadata.Array([]metadata.Value{
				metadata.Array([]metadata.Value{
					metadata.String("deep"),
					metadata.Int(123),
				}),
			}),
		}),
	}

	// Insert
	id, err := coord.Insert(ctx, []float32{0.1, 0.2}, "data", deepMeta)
	require.NoError(t, err)

	// Mutate deeply nested value
	deepMeta["level1"].A[0].A[0].A[0] = metadata.String("MUTATED")
	deepMeta["level1"].A[0].A[0].A[1] = metadata.Int(999)

	// Verify stored metadata is unchanged
	meta, ok := coord.GetMetadata(id)
	require.True(t, ok)
	
	level1 := meta["level1"]
	require.Equal(t, metadata.KindArray, level1.Kind)
	
	level2 := level1.A[0]
	require.Equal(t, metadata.KindArray, level2.Kind)
	
	level3 := level2.A[0]
	require.Equal(t, metadata.KindArray, level3.Kind)
	
	assert.Equal(t, "deep", level3.A[0].S, "deeply nested string should not be mutated")
	assert.Equal(t, int64(123), level3.A[1].I64, "deeply nested int should not be mutated")
}

// TestMetadataCloneAllTypes verifies cloning works for all metadata value types.
func TestMetadataCloneAllTypes(t *testing.T) {
	ctx := context.Background()
	coord := createTestCoordinator[string](t)

	// Create metadata with all value types
	allTypesMeta := metadata.Metadata{
		"string": metadata.String("test"),
		"int64":  metadata.Int(42),
		"float64": metadata.Float(3.14),
		"bool":   metadata.Bool(true),
		"array": metadata.Array([]metadata.Value{
			metadata.String("a"),
			metadata.Int(1),
			metadata.Float(2.5),
			metadata.Bool(false),
		}),
	}

	// Insert
	id, err := coord.Insert(ctx, []float32{0.1, 0.2}, "data", allTypesMeta)
	require.NoError(t, err)

	// Mutate all fields
	allTypesMeta["string"] = metadata.String("MUTATED")
	allTypesMeta["int64"] = metadata.Int(999)
	allTypesMeta["float64"] = metadata.Float(9.99)
	allTypesMeta["bool"] = metadata.Bool(false)
	allTypesMeta["array"].A[0] = metadata.String("MUTATED")

	// Verify all types are unchanged
	meta, ok := coord.GetMetadata(id)
	require.True(t, ok)
	
	assert.Equal(t, "test", meta["string"].S)
	assert.Equal(t, int64(42), meta["int64"].I64)
	assert.Equal(t, 3.14, meta["float64"].F64)
	assert.Equal(t, true, meta["bool"].B)
	assert.Equal(t, "a", meta["array"].A[0].S)
	assert.Equal(t, int64(1), meta["array"].A[1].I64)
	assert.Equal(t, 2.5, meta["array"].A[2].F64)
	assert.Equal(t, false, meta["array"].A[3].B)
}
