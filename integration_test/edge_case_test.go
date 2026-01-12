package integration_test

import (
	"context"
	"math"
	"testing"

	"github.com/hupe1980/vecgo"
	"github.com/hupe1980/vecgo/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestEdgeCases_Vectors(t *testing.T) {
	dir := t.TempDir()
	eng, err := vecgo.Open(dir, vecgo.Create(128, vecgo.MetricL2))
	require.NoError(t, err)
	defer eng.Close()

	ctx := context.Background()

	t.Run("Insert NaN", func(t *testing.T) {
		vec := make([]float32, 128)
		vec[0] = float32(math.NaN())
		_, err := eng.Insert(vec, nil, nil)
		assert.Error(t, err)
	})

	t.Run("Insert Inf", func(t *testing.T) {
		vec := make([]float32, 128)
		vec[0] = float32(math.Inf(1))
		_, err := eng.Insert(vec, nil, nil)
		assert.Error(t, err)
	})

	t.Run("Insert Zero Length", func(t *testing.T) {
		vec := []float32{}
		_, err := eng.Insert(vec, nil, nil)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "dimension")
	})

	t.Run("Insert Wrong Dimension", func(t *testing.T) {
		vec := make([]float32, 127)
		_, err := eng.Insert(vec, nil, nil)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "dimension")
	})

	t.Run("Search NaN", func(t *testing.T) {
		vec := make([]float32, 128)
		vec[0] = float32(math.NaN())
		_, err := eng.Search(ctx, vec, 10)
		assert.Error(t, err)
	})

	t.Run("Search Wrong Dimension", func(t *testing.T) {
		vec := make([]float32, 127)
		_, err := eng.Search(ctx, vec, 10)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "dimension")
	})
}

func TestEdgeCases_EmptyEngine(t *testing.T) {
	dir := t.TempDir()
	eng, err := vecgo.Open(dir, vecgo.Create(128, vecgo.MetricL2))
	require.NoError(t, err)
	defer eng.Close()

	ctx := context.Background()

	t.Run("Search Empty", func(t *testing.T) {
		// Should return empty results, no error
		vec := make([]float32, 128)
		res, err := eng.Search(ctx, vec, 10)
		assert.NoError(t, err)
		assert.Empty(t, res)
	})

	t.Run("Get Non Existent", func(t *testing.T) {
		_, err := eng.Get(model.ID(99999))
		assert.Error(t, err)
		// Should be not found or invalid argument
	})

	t.Run("Delete Non Existent", func(t *testing.T) {
		// Delete is idempotent
		err := eng.Delete(model.ID(99999))
		assert.NoError(t, err)
	})
}

func TestEdgeCases_MaxDimension(t *testing.T) {
	// vecgo typically supports up to some reasonable limit, or uint16/uint32 limit.
	// We test a reasonably large dimension.
	dim := 4096
	dir := t.TempDir()
	eng, err := vecgo.Open(dir, vecgo.Create(dim, vecgo.MetricL2))
	require.NoError(t, err)
	defer eng.Close()

	vec := make([]float32, dim)
	vec[0] = 1.0

	id, err := eng.Insert(vec, nil, nil)
	require.NoError(t, err)

	res, err := eng.Search(context.Background(), vec, 10)
	require.NoError(t, err)
	require.NotEmpty(t, res)
	assert.Equal(t, id, res[0].ID)
}
