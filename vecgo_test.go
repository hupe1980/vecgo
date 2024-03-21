package vecgo

import (
	"bytes"
	"testing"

	"github.com/hupe1980/vecgo/util"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestVecgo(t *testing.T) {
	t.Run("InsertAndRetrieve", func(t *testing.T) {
		vg := NewHNSW[float32]()

		vec := VectorWithData[float32]{
			Vector: []float32{1.0, 2.0, 3.0},
			Data:   42.0,
		}

		id, err := vg.Insert(&vec)
		require.NoError(t, err)

		data, err := vg.Get(id)
		require.NoError(t, err)
		assert.Equal(t, vec.Data, data)
	})

	t.Run("SaveAndLoad", func(t *testing.T) {
		vg := NewHNSW[float32]()

		vec := VectorWithData[float32]{
			Vector: []float32{1.0, 2.0, 3.0},
			Data:   42.0,
		}

		id, err := vg.Insert(&vec)
		require.NoError(t, err)

		r, err := vg.KNNSearch([]float32{1.0, 2.0, 3.0}, 1)
		require.NoError(t, err)
		require.Len(t, r, 1)

		var buf bytes.Buffer
		err = vg.SaveToWriter(&buf)
		require.NoError(t, err)

		vgLoaded, err := NewFromReader[float32](&buf)
		require.NoError(t, err)

		data, err := vgLoaded.Get(id)
		require.NoError(t, err)
		assert.Equal(t, vec.Data, data)

		r, err = vgLoaded.KNNSearch([]float32{1.0, 2.0, 3.0}, 1)
		require.NoError(t, err)
		require.Len(t, r, 1)
	})

	t.Run("KNN", func(t *testing.T) {
		vg := NewHNSW[float32]()

		vec1 := VectorWithData[float32]{
			Vector: []float32{05, 1.0, 0.5},
			Data:   42.0,
		}

		vec2 := VectorWithData[float32]{
			Vector: []float32{0.5, 1.0, 1.0},
			Data:   24.0,
		}

		vec3 := VectorWithData[float32]{
			Vector: []float32{0.5, 0.5, 1.0},
			Data:   12.0,
		}

		_, err := vg.Insert(&vec1)
		require.NoError(t, err)

		_, err = vg.Insert(&vec2)
		require.NoError(t, err)

		_, err = vg.Insert(&vec3)
		require.NoError(t, err)

		query := []float32{0.5, 0.5, 0.5}

		results, err := vg.KNNSearch(query, 2)
		require.NoError(t, err)
		require.Len(t, results, 2)
		assert.Equal(t, float32(24.0), results[0].Data)
		assert.Equal(t, float32(12.0), results[1].Data)
	})

	t.Run("Brute", func(t *testing.T) {
		vg := NewHNSW[float32]()

		vec1 := VectorWithData[float32]{
			Vector: []float32{05, 1.0, 0.5},
			Data:   42.0,
		}

		vec2 := VectorWithData[float32]{
			Vector: []float32{0.5, 1.0, 1.0},
			Data:   24.0,
		}

		vec3 := VectorWithData[float32]{
			Vector: []float32{0.5, 0.5, 1.0},
			Data:   12.0,
		}

		_, err := vg.Insert(&vec1)
		require.NoError(t, err)

		_, err = vg.Insert(&vec2)
		require.NoError(t, err)

		_, err = vg.Insert(&vec3)
		require.NoError(t, err)

		query := []float32{0.5, 0.5, 0.5}

		results, err := vg.BruteSearch(query, 2)
		require.NoError(t, err)
		require.Len(t, results, 2)
		assert.Equal(t, float32(24.0), results[0].Data)
		assert.Equal(t, float32(12.0), results[1].Data)
	})
}

func BenchmarkInsertAndBatchInsert(b *testing.B) {
	dim := 1024

	b.Run("InsertOneByOne", func(b *testing.B) {
		vg := NewHNSW[int]()

		rng := util.NewRNG(4711)

		vectors := rng.GenerateRandomVectors(b.N, dim)
		vectorWithData := make([]*VectorWithData[int], b.N)

		for i := 0; i < b.N; i++ {
			vectorWithData[i] = &VectorWithData[int]{
				Vector: vectors[i],
				Data:   i,
			}
		}

		b.ResetTimer()

		for i := 0; i < b.N; i++ {
			_, err := vg.Insert(vectorWithData[i])
			if err != nil {
				b.Fatalf("Insert failed: %v", err)
			}
		}
	})
}
