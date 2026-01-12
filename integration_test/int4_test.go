package integration_test

import (
	"context"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/hupe1980/vecgo"
	"github.com/stretchr/testify/require"
)

func TestINT4Quantization(t *testing.T) {
	dir := t.TempDir()
	defer os.RemoveAll(dir)

	// Create DB with INT4 quantization
	db, err := vecgo.Open(dir,
		vecgo.Create(128, vecgo.MetricL2),
		vecgo.WithQuantization(vecgo.QuantizationTypeINT4),
		vecgo.WithCompactionThreshold(2), // Compact when 2 segments exist
		vecgo.WithDiskANNThreshold(0),    // Force DiskANN
	)
	require.NoError(t, err)
	defer db.Close()

	// Vector 1: All 0.2
	vec1 := make([]float32, 128)
	for i := 0; i < 128; i++ {
		vec1[i] = 0.2
	}
	id1, _ := db.Insert(vec1, nil, nil)
	db.Flush() // Segment 1

	// Vector 2: All 0.8
	vec2 := make([]float32, 128)
	for i := 0; i < 128; i++ {
		vec2[i] = 0.8
	}
	id2, _ := db.Insert(vec2, nil, nil)
	db.Flush() // Segment 2

	// Wait for compaction (2 segments -> 1)
	time.Sleep(1 * time.Second)

	// Search for Vec1
	// Should find ID 1. Score 0 (Exact match as 0.2 is 2/10 -> representable or min-aligned)
	res1, err := db.Search(context.Background(), vec1, 10)
	require.NoError(t, err)
	fmt.Printf("Search Vec1 Results: %+v\n", res1)

	require.Equal(t, id1, res1[0].ID)
	// 0.2 and 0.8 are min/max, so they are exact boundaries
	require.InDelta(t, 0.0, res1[0].Score, 0.01)

	// Search for Vec2
	res2, err := db.Search(context.Background(), vec2, 10)
	require.NoError(t, err)
	fmt.Printf("Search Vec2 Results: %+v\n", res2)
	require.Equal(t, id2, res2[0].ID)
	require.InDelta(t, 0.0, res2[0].Score, 0.01)
}
