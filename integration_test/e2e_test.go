package integration_test

import (
	"context"
	"testing"

	"github.com/hupe1980/vecgo"
	"github.com/stretchr/testify/require"
)

func TestE2E_Restart(t *testing.T) {
	dir := t.TempDir()

	// 1. Open and Insert
	e, err := vecgo.Open(dir, vecgo.Create(2, vecgo.MetricL2))
	require.NoError(t, err)

	id, err := e.Insert([]float32{1.0, 0.0}, nil, nil)
	require.NoError(t, err)

	err = e.Flush()
	require.NoError(t, err)

	err = e.Close()
	require.NoError(t, err)

	// 2. Reopen and Verify
	e, err = vecgo.Open(dir)
	require.NoError(t, err)
	defer e.Close()

	res, err := e.Search(context.Background(), []float32{1.0, 0.0}, 1)
	require.NoError(t, err)
	require.Len(t, res, 1)
	require.Equal(t, id, res[0].ID)
}
