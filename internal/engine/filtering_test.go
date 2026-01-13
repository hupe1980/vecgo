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

	ctx := context.Background()

	// Insert with metadata
	id1, err := e.Insert(ctx, []float32{1.0, 0.0}, metadata.Document{
		"category": metadata.String("A"),
		"price":    metadata.Float(10.0),
	}, nil)
	require.NoError(t, err)

	id2, err := e.Insert(ctx, []float32{0.0, 1.0}, metadata.Document{
		"category": metadata.String("B"),
		"price":    metadata.Float(20.0),
	}, nil)
	require.NoError(t, err)

	id3, err := e.Insert(ctx, []float32{1.0, 1.0}, metadata.Document{
		"category": metadata.String("A"),
		"price":    metadata.Float(30.0),
	}, nil)
	require.NoError(t, err)

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
	ids := make([]model.ID, len(res))
	for i, r := range res {
		ids[i] = r.ID
	}
	assert.ElementsMatch(t, []model.ID{id1, id3}, ids)

	// 2. Filter by price > 15
	filterPrice := metadata.NewFilterSet(metadata.Filter{
		Key:      "price",
		Operator: metadata.OpGreaterThan,
		Value:    metadata.Float(15.0),
	})

	res, err = e.Search(ctx, []float32{0.0, 0.0}, 10, WithFilter(filterPrice))
	require.NoError(t, err)
	assert.Len(t, res, 2) // 2 and 3
	ids = make([]model.ID, len(res))
	for i, r := range res {
		ids[i] = r.ID
	}
	assert.ElementsMatch(t, []model.ID{id2, id3}, ids)

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
	assert.Equal(t, id2, res[0].ID)
}
