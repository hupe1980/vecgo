package vecgo

import (
	"bytes"
	"context"
	"sync"
	"testing"

	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestVecgo(t *testing.T) {
	t.Run("InsertAndRetrieve", func(t *testing.T) {
		vg, err := HNSW[float32](3).SquaredL2().Build()
		require.NoError(t, err)

		vec := VectorWithData[float32]{
			Vector: []float32{1.0, 2.0, 3.0},
			Data:   42.0,
		}

		id, err := vg.Insert(context.Background(), vec)
		require.NoError(t, err)

		data, err := vg.Get(id)
		require.NoError(t, err)
		assert.Equal(t, vec.Data, data)
	})

	t.Run("SaveAndLoad", func(t *testing.T) {
		vg, err := HNSW[float32](3).SquaredL2().Build()
		require.NoError(t, err)

		vec := VectorWithData[float32]{
			Vector: []float32{1.0, 2.0, 3.0},
			Data:   42.0,
		}

		id, err := vg.Insert(context.Background(), vec)
		require.NoError(t, err)

		r, err := vg.KNNSearch(context.Background(), []float32{1.0, 2.0, 3.0}, 1)
		require.NoError(t, err)
		require.Len(t, r, 1)

		var buf bytes.Buffer
		err = vg.SaveToWriter(&buf)
		require.NoError(t, err)

		// Save to file for mmap loading (NewFromReader was removed due to 153GB allocation)
		path := t.TempDir() + "/snap.bin"
		err = vg.SaveToFile(path)
		require.NoError(t, err)

		vgLoaded, err := NewFromFile[float32](path)
		require.NoError(t, err)
		defer vgLoaded.Close()

		data, err := vgLoaded.Get(id)
		require.NoError(t, err)
		assert.Equal(t, vec.Data, data)

		r, err = vgLoaded.KNNSearch(context.Background(), []float32{1.0, 2.0, 3.0}, 1)
		require.NoError(t, err)
		require.Len(t, r, 1)
	})

	t.Run("KNN", func(t *testing.T) {
		vg, err := HNSW[float32](3).SquaredL2().Build()
		require.NoError(t, err)

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

		_, err = vg.Insert(context.Background(), vec1)
		require.NoError(t, err)

		_, err = vg.Insert(context.Background(), vec2)
		require.NoError(t, err)

		_, err = vg.Insert(context.Background(), vec3)
		require.NoError(t, err)

		query := []float32{0.5, 0.5, 0.5}

		results, err := vg.KNNSearch(context.Background(), query, 2)
		require.NoError(t, err)
		require.Len(t, results, 2)
		assert.Equal(t, float32(12.0), results[0].Data)
		assert.Equal(t, float32(24.0), results[1].Data)
	})

	t.Run("KNNStream", func(t *testing.T) {
		vg, err := HNSW[float32](3).SquaredL2().Build()
		require.NoError(t, err)

		vec1 := VectorWithData[float32]{
			Vector: []float32{05, 1.0, 0.5}, // Same as KNN test: 05 = 5
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

		_, err = vg.Insert(context.Background(), vec1)
		require.NoError(t, err)

		_, err = vg.Insert(context.Background(), vec2)
		require.NoError(t, err)

		_, err = vg.Insert(context.Background(), vec3)
		require.NoError(t, err)

		query := []float32{0.5, 0.5, 0.5}

		// Test streaming search returns same results as regular search
		var streamResults []SearchResult[float32]
		for result, err := range vg.KNNSearchStream(context.Background(), query, 2) {
			require.NoError(t, err)
			streamResults = append(streamResults, result)
		}
		require.Len(t, streamResults, 2)
		assert.Equal(t, float32(12.0), streamResults[0].Data)
		assert.Equal(t, float32(24.0), streamResults[1].Data)

		// Test early termination
		count := 0
		for result, err := range vg.KNNSearchStream(context.Background(), query, 3) {
			require.NoError(t, err)
			count++
			_ = result
			if count == 1 {
				break // Early termination after first result
			}
		}
		assert.Equal(t, 1, count)
	})

	t.Run("Brute", func(t *testing.T) {
		vg, err := HNSW[float32](3).SquaredL2().Build()
		require.NoError(t, err)

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

		_, err = vg.Insert(context.Background(), vec1)
		require.NoError(t, err)

		_, err = vg.Insert(context.Background(), vec2)
		require.NoError(t, err)

		_, err = vg.Insert(context.Background(), vec3)
		require.NoError(t, err)

		query := []float32{0.5, 0.5, 0.5}

		results, err := vg.BruteSearch(context.Background(), query, 2)
		require.NoError(t, err)
		require.Len(t, results, 2)
		assert.Equal(t, float32(12.0), results[0].Data)
		assert.Equal(t, float32(24.0), results[1].Data)
	})
}

func BenchmarkInsertAndBatchInsert(b *testing.B) {
	dim := 1024

	b.Run("InsertOneByOne", func(b *testing.B) {
		vg, err := HNSW[int](dim).SquaredL2().Build()
		if err != nil {
			b.Fatal(err)
		}

		rng := testutil.NewRNG(4711)

		vectors := rng.UniformVectors(b.N, dim)
		vectorWithData := make([]VectorWithData[int], b.N)

		for i := range vectorWithData {
			vectorWithData[i] = VectorWithData[int]{
				Vector: vectors[i],
				Data:   i,
			}
		}

		b.ResetTimer()
		var i int
		for b.Loop() {
			_, err := vg.Insert(context.Background(), vectorWithData[i])
			if err != nil {
				b.Fatalf("Insert failed: %v", err)
			}
			i++
		}
	})

	b.Run("InsertParallel", func(b *testing.B) {
		vg, err := HNSW[int](dim).SquaredL2().Build()
		if err != nil {
			b.Fatal(err)
		}

		rng := testutil.NewRNG(4711)

		vectors := rng.UniformVectors(b.N, dim)
		vectorWithData := make([]VectorWithData[int], b.N)

		for i := range vectorWithData {
			vectorWithData[i] = VectorWithData[int]{
				Vector: vectors[i],
				Data:   i,
			}
		}

		b.ResetTimer()

		errCh := make(chan error, len(vectorWithData)) // Error channel to receive errors from goroutines

		var wg sync.WaitGroup

		wg.Add(len(vectorWithData))

		for i := range vectorWithData {
			go func(i int) {
				defer wg.Done()

				_, err := vg.Insert(context.Background(), vectorWithData[i])
				if err != nil {
					errCh <- err
				}
			}(i)
		}

		go func() {
			wg.Wait()
			close(errCh) // Close the error channel after all goroutines are done
		}()

		// Collect errors from the error channel
		for err := range errCh {
			if err != nil {
				b.Fatalf("Insert failed: %v", err)
			}
		}
	})
}

func TestNewFromFile_Smoke(t *testing.T) {
	ctx := context.Background()

	vg, err := Flat[string](3).SquaredL2().Build()
	require.NoError(t, err)
	defer vg.Close()

	_, err = vg.Insert(ctx, VectorWithData[string]{
		Vector:   []float32{1, 2, 3},
		Data:     "a",
		Metadata: metadata.Metadata{"k": metadata.String("v")},
	})
	require.NoError(t, err)

	path := t.TempDir() + "/snap.bin"
	require.NoError(t, vg.SaveToFile(path))

	vg2, err := NewFromFile[string](path)
	require.NoError(t, err)
	defer vg2.Close()

	res, err := vg2.KNNSearch(ctx, []float32{1, 2, 3}, 1)
	require.NoError(t, err)
	require.Len(t, res, 1)
}
