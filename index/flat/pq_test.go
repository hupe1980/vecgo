package flat

import (
	"context"
	"testing"

	"github.com/hupe1980/vecgo/index"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestFlat_ProductQuantization_EnableDisable(t *testing.T) {
	f, err := New(func(o *Options) {
		o.Dimension = 4
		o.DistanceType = index.DistanceTypeSquaredL2
	})
	require.NoError(t, err)

	id0, err := f.Insert(context.Background(), []float32{0, 0, 0, 0})
	require.NoError(t, err)
	_, err = f.Insert(context.Background(), []float32{10, 10, 10, 10})
	require.NoError(t, err)
	_, err = f.Insert(context.Background(), []float32{1, 1, 1, 1})
	require.NoError(t, err)

	require.False(t, f.ProductQuantizationEnabled())
	require.Nil(t, f.pq.Load())
	require.Nil(t, f.getState().pqCodes)

	err = f.EnableProductQuantization(index.ProductQuantizationConfig{NumSubvectors: 2, NumCentroids: 16})
	require.NoError(t, err)
	require.True(t, f.ProductQuantizationEnabled())

	st := f.getState()
	require.NotNil(t, st.pqCodes)
	require.Len(t, st.pqCodes, len(st.nodes))
	require.NotNil(t, st.pqCodes[id0])

	query := []float32{0.25, 0.25, 0.25, 0.25}
	pq := f.pq.Load()
	require.NotNil(t, pq)
	expected := pq.ComputeAsymmetricDistance(query, st.pqCodes[id0])

	res, err := f.BruteSearch(context.Background(), query, 1, func(id uint64) bool { return id == id0 })
	require.NoError(t, err)
	require.Len(t, res, 1)
	assert.Equal(t, id0, res[0].ID)
	assert.InDelta(t, expected, res[0].Distance, 1e-6)

	idNew, err := f.Insert(context.Background(), []float32{2, 2, 2, 2})
	require.NoError(t, err)
	require.NotNil(t, f.getState().pqCodes[idNew])

	f.DisableProductQuantization()
	require.False(t, f.ProductQuantizationEnabled())
	require.Nil(t, f.pq.Load())
	require.Nil(t, f.getState().pqCodes)
}
