package vecgo

import (
	"bytes"
	"testing"

	"github.com/hupe1980/vecgo/hnsw"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestVecgo(t *testing.T) {
	t.Run("InsertAndRetrieve", func(t *testing.T) {
		vg := New[float32](3)

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
		vg := New[float32](3)

		vec := VectorWithData[float32]{
			Vector: []float32{1.0, 2.0, 3.0},
			Data:   42.0,
		}

		id, err := vg.Insert(&vec)
		require.NoError(t, err)

		var buf bytes.Buffer
		err = vg.SaveToWriter(&buf)
		require.NoError(t, err)

		vgLoaded, err := NewFromReader[float32](&buf)
		require.NoError(t, err)

		data, err := vgLoaded.Get(id)
		require.NoError(t, err)
		assert.Equal(t, vec.Data, data)
	})

	t.Run("KNN", func(t *testing.T) {
		vg := New[float32](3)

		vec1 := VectorWithData[float32]{
			Vector: []float32{1.0, 0.0, 0.0},
			Data:   42.0,
		}

		vec2 := VectorWithData[float32]{
			Vector: []float32{0.0, 1.0, 0.0},
			Data:   24.0,
		}

		vec3 := VectorWithData[float32]{
			Vector: []float32{0.0, 0.0, 1.0},
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
		require.Len(t, results, 1)
		assert.Equal(t, float32(24.0), results[0].Data)
	})

	t.Run("Brute", func(t *testing.T) {
		vg := New[float32](3)

		vec1 := VectorWithData[float32]{
			Vector: []float32{1.0, 0.0, 0.1},
			Data:   42.0,
		}

		vec2 := VectorWithData[float32]{
			Vector: []float32{0.0, 0.5, 0.0},
			Data:   24.0,
		}

		vec3 := VectorWithData[float32]{
			Vector: []float32{0.3, 0.0, 1.0},
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
		require.Len(t, results, 1)
		assert.Equal(t, float32(12.0), results[0].Data)
	})
}

func BenchmarkInsertAndBatchInsert(b *testing.B) {
	dim := 1024

	// BenchmarkInsertAndBatchInsert/InsertOneByOne-10         	    2508	   2217397 ns/op	   42861 B/op	    1107 allocs/op
	b.Run("InsertOneByOne", func(b *testing.B) {
		vg := New[int](dim)

		vectors := hnsw.GenerateRandomVectors(b.N, dim, 4711)
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

	// BenchmarkInsertAndBatchInsert/BatchInsert-10            	    2643	   2269005 ns/op	   43646 B/op	    1140 allocs/op
	b.Run("BatchInsert", func(b *testing.B) {
		vg := New[int](dim)

		vectors := hnsw.GenerateRandomVectors(b.N, dim, 4711)
		vectorWithData := make([]*VectorWithData[int], b.N)

		for i := 0; i < b.N; i++ {
			vectorWithData[i] = &VectorWithData[int]{
				Vector: vectors[i],
				Data:   i,
			}
		}

		b.ResetTimer()

		_, err := vg.BatchInsert(vectorWithData)
		if err != nil {
			b.Fatalf("BatchInsert failed: %v", err)
		}
	})
}
