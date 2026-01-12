package quantization

import (
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestInt4Quantizer(t *testing.T) {
	dim := 128
	vectors := make([][]float32, 100)
	rng := rand.New(rand.NewSource(1))

	for i := 0; i < 100; i++ {
		vectors[i] = make([]float32, dim)
		for j := 0; j < dim; j++ {
			vectors[i][j] = rng.Float32()
		}
	}

	q := NewInt4Quantizer(dim)
	err := q.Train(vectors)
	require.NoError(t, err)

	t.Run("EncodeDecode", func(t *testing.T) {
		vec := vectors[0]
		encoded, err := q.Encode(vec)
		require.NoError(t, err)
		assert.Equal(t, (dim+1)/2, len(encoded))

		decoded, err := q.Decode(encoded)
		require.NoError(t, err)
		assert.Equal(t, dim, len(decoded))

		// Check MSE
		var mse float32
		for i := 0; i < dim; i++ {
			diff := vec[i] - decoded[i]
			mse += diff * diff
		}
		mse /= float32(dim)

		// Int4 quantization error is expected to be roughly (1/15)^2 / 12 ~= 0.00037 for unit range
		// Our range is roughly [0, 1].
		// Expected MSE should be low.
		assert.Less(t, mse, float32(0.01))
	})

	t.Run("OddDimension", func(t *testing.T) {
		oddDim := 3
		qOdd := NewInt4Quantizer(oddDim)
		vecs := [][]float32{{0.1, 0.5, 0.9}}
		qOdd.Train(vecs)

		encoded, err := qOdd.Encode(vecs[0])
		require.NoError(t, err)
		assert.Equal(t, 2, len(encoded)) // ceil(3/2) = 2

		decoded, err := qOdd.Decode(encoded)
		require.NoError(t, err)
		assert.Equal(t, oddDim, len(decoded))
		assert.InDelta(t, 0.1, decoded[0], 0.1)
		assert.InDelta(t, 0.5, decoded[1], 0.1)
		assert.InDelta(t, 0.9, decoded[2], 0.1)
	})
}
