package flat

import (
	"context"
	"testing"

	"github.com/hupe1980/vecgo/core"
	"github.com/hupe1980/vecgo/distance"
	"github.com/hupe1980/vecgo/index"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestFlat(t *testing.T) {
	t.Run("Insert", func(t *testing.T) {
		// Initialize the flat index
		f, err := New(func(o *Options) {
			o.Dimension = 3
			o.DistanceType = index.DistanceTypeSquaredL2
		})
		require.NoError(t, err)

		// Insert a vector
		id, err := f.Insert(context.Background(), []float32{1.0, 2.0, 3.0})
		require.NoError(t, err)
		assert.Equal(t, core.LocalID(0), id)

		// Test dimension mismatch error
		_, err = f.Insert(context.Background(), []float32{1.0, 2.0})
		assert.Error(t, err)
		assert.IsType(t, &index.ErrDimensionMismatch{}, err)
	})

	t.Run("KNNSearch", func(t *testing.T) {
		// Initialize the flat index
		f, err := New(func(o *Options) {
			o.Dimension = 3
			o.DistanceType = index.DistanceTypeSquaredL2
		})
		require.NoError(t, err)

		// Insert vectors
		_, _ = f.Insert(context.Background(), []float32{1.0, 2.0, 3.0})
		_, _ = f.Insert(context.Background(), []float32{4.0, 5.0, 6.0})
		_, _ = f.Insert(context.Background(), []float32{7.0, 8.0, 9.0})

		// Perform brute-force search
		results, err := f.KNNSearch(context.Background(), []float32{0.0, 0.0, 0.0}, 2, &index.SearchOptions{
			Filter: func(id core.LocalID) bool { return true },
		})
		require.NoError(t, err)
		assert.Equal(t, 2, len(results))
		assert.Equal(t, uint32(0), results[0].ID)
		assert.Equal(t, uint32(1), results[1].ID)
	})

	t.Run("KNNSearch_DotProduct", func(t *testing.T) {
		ctx := context.Background()
		f, err := New(func(o *Options) {
			o.Dimension = 3
			o.DistanceType = index.DistanceTypeDotProduct
		})
		require.NoError(t, err)

		id0, err := f.Insert(ctx, []float32{1, 0, 0})
		require.NoError(t, err)
		id1, err := f.Insert(ctx, []float32{2, 0, 0})
		require.NoError(t, err)
		id2, err := f.Insert(ctx, []float32{-1, 0, 0})
		require.NoError(t, err)

		query := []float32{1, 0, 0}
		results, err := f.KNNSearch(ctx, query, 3, &index.SearchOptions{Filter: func(id core.LocalID) bool { return true }})
		require.NoError(t, err)
		require.Len(t, results, 3)

		assert.Equal(t, uint32(id1), results[0].ID)
		assert.Equal(t, uint32(id0), results[1].ID)
		assert.Equal(t, uint32(id2), results[2].ID)

		expected0 := -distance.Dot(query, []float32{2, 0, 0})
		expected1 := -distance.Dot(query, []float32{1, 0, 0})
		expected2 := -distance.Dot(query, []float32{-1, 0, 0})
		assert.InDelta(t, expected0, results[0].Distance, 1e-6)
		assert.InDelta(t, expected1, results[1].Distance, 1e-6)
		assert.InDelta(t, expected2, results[2].Distance, 1e-6)
	})

	t.Run("BruteSearch", func(t *testing.T) {
		// Initialize the flat index
		f, err := New(func(o *Options) {
			o.Dimension = 3
			o.DistanceType = index.DistanceTypeSquaredL2
		})
		require.NoError(t, err)

		// Insert vectors
		_, _ = f.Insert(context.Background(), []float32{1.0, 2.0, 3.0})
		_, _ = f.Insert(context.Background(), []float32{4.0, 5.0, 6.0})
		_, _ = f.Insert(context.Background(), []float32{7.0, 8.0, 9.0})

		// Perform brute-force search
		results, err := f.BruteSearch(context.Background(), []float32{0.0, 0.0, 0.0}, 2, func(id core.LocalID) bool { return true })
		require.NoError(t, err)
		assert.Equal(t, 2, len(results))
		assert.Equal(t, uint32(0), results[0].ID)
		assert.Equal(t, uint32(1), results[1].ID)
	})
}
