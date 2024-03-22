package flat

import (
	"testing"

	"github.com/hupe1980/vecgo/index"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestFlat(t *testing.T) {
	t.Run("Insert", func(t *testing.T) {
		// Initialize the flat index
		f := New()

		// Insert a vector
		id, err := f.Insert([]float32{1.0, 2.0, 3.0})
		require.NoError(t, err)
		assert.Equal(t, uint32(0), id)

		// Test dimension mismatch error
		_, err = f.Insert([]float32{1.0, 2.0})
		assert.Error(t, err)
		assert.IsType(t, &index.ErrDimensionMismatch{}, err)
	})

	t.Run("KNNSearch", func(t *testing.T) {
		// Initialize the flat index
		f := New()

		// Insert vectors
		_, _ = f.Insert([]float32{1.0, 2.0, 3.0})
		_, _ = f.Insert([]float32{4.0, 5.0, 6.0})
		_, _ = f.Insert([]float32{7.0, 8.0, 9.0})

		// Perform brute-force search
		results, err := f.KNNSearch([]float32{0.0, 0.0, 0.0}, 2, 0, func(id uint32) bool { return true })
		require.NoError(t, err)
		assert.Equal(t, 2, len(results))
		assert.Equal(t, uint32(0), results[0].ID)
		assert.Equal(t, uint32(1), results[1].ID)
	})

	t.Run("BruteSearch", func(t *testing.T) {
		// Initialize the flat index
		f := New()

		// Insert vectors
		_, _ = f.Insert([]float32{1.0, 2.0, 3.0})
		_, _ = f.Insert([]float32{4.0, 5.0, 6.0})
		_, _ = f.Insert([]float32{7.0, 8.0, 9.0})

		// Perform brute-force search
		results, err := f.BruteSearch([]float32{0.0, 0.0, 0.0}, 2, func(id uint32) bool { return true })
		require.NoError(t, err)
		assert.Equal(t, 2, len(results))
		assert.Equal(t, uint32(0), results[0].ID)
		assert.Equal(t, uint32(1), results[1].ID)
	})
}
