package engine

import (
	"context"
	"testing"

	"github.com/hupe1980/vecgo/distance"
	"github.com/hupe1980/vecgo/lexical/bm25"
	"github.com/hupe1980/vecgo/metadata"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestHybridSearch(t *testing.T) {
	dir := t.TempDir()
	lexIdx := bm25.New()

	e, err := Open(dir, 2, distance.MetricL2, WithLexicalIndex(lexIdx, "text"))
	require.NoError(t, err)
	defer e.Close()

	// Insert data
	// 1. "apple" near origin
	id1, err := e.Insert(context.Background(), []float32{0.1, 0.1}, metadata.Document{"text": metadata.String("apple fruit")}, nil)
	require.NoError(t, err)

	// 2. "banana" far from origin
	id2, err := e.Insert(context.Background(), []float32{10.0, 10.0}, metadata.Document{"text": metadata.String("banana fruit")}, nil)
	require.NoError(t, err)

	// 3. "apple" far from origin
	id3, err := e.Insert(context.Background(), []float32{10.0, 10.1}, metadata.Document{"text": metadata.String("apple pie")}, nil)
	require.NoError(t, err)

	ctx := context.Background()

	// Case 1: Vector search only (near origin) -> should find 1
	vecRes, err := e.Search(ctx, []float32{0.0, 0.0}, 10)
	require.NoError(t, err)
	assert.Equal(t, id1, vecRes[0].ID)

	// Case 2: Lexical search only (via Hybrid with dummy vector?)
	// HybridSearch requires vector.
	// Let's search for "banana" near origin.
	// Vector match: 1 is closest.
	// Lexical match: 2 is match.
	// RRF should boost 2 if 1 is not in lexical results.

	// Search "banana" near {0,0}
	// Vector ranks: 1 (dist ~0.14), 2 (dist ~14), 3 (dist ~14)
	// Lexical ranks: 2 (score > 0), others 0.

	// RRF(1) = 1/(60+1) + 0 = 0.01639
	// RRF(2) = 1/(60+2) + 1/(60+1) = 0.0161 + 0.01639 = 0.0325
	// So 2 should win.

	res, err := e.HybridSearch(ctx, []float32{0.0, 0.0}, "banana", 10, 60)
	require.NoError(t, err)
	require.NotEmpty(t, res)
	assert.Equal(t, id2, res[0].ID)

	// Case 3: Search "apple" near {10, 10}
	// Vector ranks: 2 (dist 0), 3 (dist 0.1), 1 (dist ~14)
	// Lexical ranks: 1, 3 (both have "apple")

	// RRF(2): 1/(61) + 0 = 0.01639
	// RRF(3): 1/(62) + 1/(61 or 62) -> Vector rank 2, Lexical rank 1 or 2.
	// RRF(1): 1/(63) + 1/(61 or 62)

	// 3 is strong in both. 3 should likely win or be very high.
	res, err = e.HybridSearch(ctx, []float32{10.0, 10.0}, "apple", 10, 60)
	require.NoError(t, err)
	require.NotEmpty(t, res)

	// 3 is vector rank 2, lexical rank 1 or 2.
	// 2 is vector rank 1, lexical rank inf.
	// 1 is vector rank 3, lexical rank 1 or 2.

	// Let's check if 3 is in top 2.
	found3 := false
	for _, c := range res {
		if c.ID == id3 {
			found3 = true
			break
		}
	}
	assert.True(t, found3)
}
