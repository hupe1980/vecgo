package engine

import (
	"context"
	"testing"

	"github.com/hupe1980/vecgo/distance"
	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestFiltering(t *testing.T) {
	dir := t.TempDir()
	e, err := Open(dir, 2, distance.MetricL2)
	require.NoError(t, err)
	defer e.Close()

	// Insert with metadata
	err = e.Insert(model.PKUint64(1), []float32{1.0, 0.0}, map[string]interface{}{
		"category": "A",
		"price":    10.0,
	}, nil)
	require.NoError(t, err)

	err = e.Insert(model.PKUint64(2), []float32{0.0, 1.0}, map[string]interface{}{
		"category": "B",
		"price":    20.0,
	}, nil)
	require.NoError(t, err)

	err = e.Insert(model.PKUint64(3), []float32{1.0, 1.0}, map[string]interface{}{
		"category": "A",
		"price":    30.0,
	}, nil)
	require.NoError(t, err)

	ctx := context.Background()

	// 1. Filter by category = "A"
	filterA := metadata.NewFilterSet(metadata.Filter{
		Key:      "category",
		Operator: metadata.OpEqual,
		Value:    metadata.String("A"),
	})

	res, err := e.Search(ctx, []float32{0.0, 0.0}, 10, WithFilter(filterA))
	require.NoError(t, err)
	assert.Len(t, res, 2)
	// Order depends on distance, but both match
	pks := make([]uint64, len(res))
	for i, r := range res {
		u64, _ := r.PK.Uint64()
		pks[i] = u64
	}
	assert.ElementsMatch(t, []uint64{1, 3}, pks)

	// 2. Filter by price > 15
	filterPrice := metadata.NewFilterSet(metadata.Filter{
		Key:      "price",
		Operator: metadata.OpGreaterThan,
		Value:    metadata.Float(15.0),
	})

	res, err = e.Search(ctx, []float32{0.0, 0.0}, 10, WithFilter(filterPrice))
	require.NoError(t, err)
	assert.Len(t, res, 2) // 2 and 3
	pks = make([]uint64, len(res))
	for i, r := range res {
		u64, _ := r.PK.Uint64()
		pks[i] = u64
	}
	assert.ElementsMatch(t, []uint64{2, 3}, pks)

	// 3. Filter by category = "B" AND price > 15
	filterCombined := metadata.NewFilterSet(
		metadata.Filter{
			Key:      "category",
			Operator: metadata.OpEqual,
			Value:    metadata.String("B"),
		},
		metadata.Filter{
			Key:      "price",
			Operator: metadata.OpGreaterThan,
			Value:    metadata.Float(15.0),
		},
	)

	res, err = e.Search(ctx, []float32{0.0, 0.0}, 10, WithFilter(filterCombined))
	require.NoError(t, err)
	assert.Len(t, res, 1)
	u64, _ := res[0].PK.Uint64()
	assert.Equal(t, uint64(2), u64)
}
