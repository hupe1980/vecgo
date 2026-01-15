package integration_test

import (
	"context"
	"testing"
	"time"

	"github.com/hupe1980/vecgo"
	"github.com/hupe1980/vecgo/distance"
	"github.com/hupe1980/vecgo/model"
	"github.com/hupe1980/vecgo/testutil"
	"github.com/stretchr/testify/require"
)

// TestQuantizationRecall validates that quantized search maintains acceptable recall.
// This tests the two-stage search: quantized coarse search + full-precision reranking.
func TestQuantizationRecall(t *testing.T) {
	testCases := []struct {
		name         string
		quantization vecgo.QuantizationType
		minRecall    float64 // Minimum acceptable recall@10
		vectorCount  int
		dim          int
	}{
		{
			name:         "SQ8",
			quantization: vecgo.QuantizationTypeSQ8,
			minRecall:    0.90, // SQ8 should achieve >90% recall
			vectorCount:  500,
			dim:          128,
		},
		{
			name:         "INT4",
			quantization: vecgo.QuantizationTypeINT4,
			minRecall:    0.85, // INT4 has higher quantization error
			vectorCount:  500,
			dim:          128,
		},
		{
			name:         "PQ",
			quantization: vecgo.QuantizationTypePQ,
			minRecall:    0.80, // PQ with reranking should achieve >80%
			vectorCount:  500,
			dim:          128,
		},
		{
			name:         "RaBitQ",
			quantization: vecgo.QuantizationTypeRaBitQ,
			minRecall:    0.75, // Binary quantization has lower base recall
			vectorCount:  500,
			dim:          128,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			dir := t.TempDir()

			// Create DB with quantization
			db, err := vecgo.Open(ctx, vecgo.Local(dir),
				vecgo.Create(tc.dim, vecgo.MetricL2),
				vecgo.WithQuantization(tc.quantization),
				vecgo.WithCompactionThreshold(2),
				vecgo.WithDiskANNThreshold(0), // Force DiskANN to use quantization
			)
			require.NoError(t, err)
			defer db.Close()

			// Generate random vectors
			rng := testutil.NewRNG(42)
			vectors := rng.UnitVectors(tc.vectorCount, tc.dim)
			ids := make([]model.ID, tc.vectorCount)

			// Insert vectors in batches
			batchSize := 100
			for i := 0; i < tc.vectorCount; i += batchSize {
				end := i + batchSize
				if end > tc.vectorCount {
					end = tc.vectorCount
				}
				for j := i; j < end; j++ {
					id, err := db.Insert(ctx, vectors[j], nil, nil)
					require.NoError(t, err)
					ids[j] = id
				}
				require.NoError(t, db.Commit(ctx))
			}

			// Wait for compaction to create quantized segment
			time.Sleep(2 * time.Second)

			// Run recall test with multiple query vectors
			numQueries := 20
			k := 10
			queryVectors := rng.UnitVectors(numQueries, tc.dim)

			totalRecall := 0.0
			for _, query := range queryVectors {
				// Get results from vecgo
				results, err := db.Search(ctx, query, k)
				require.NoError(t, err)

				// Compute ground truth using brute force
				groundTruth := bruteForceTopK(vectors, ids, query, k)

				// Calculate recall
				recall := recallAtK(results, groundTruth)
				totalRecall += recall
			}

			avgRecall := totalRecall / float64(numQueries)
			t.Logf("%s: Average recall@%d = %.2f%% (min required: %.0f%%)",
				tc.name, k, avgRecall*100, tc.minRecall*100)

			require.GreaterOrEqual(t, avgRecall, tc.minRecall,
				"Recall %.2f%% is below minimum threshold %.0f%%",
				avgRecall*100, tc.minRecall*100)
		})
	}
}

// bruteForceTopK returns the ground truth top-k IDs using exact L2 distance.
func bruteForceTopK(vectors [][]float32, ids []model.ID, query []float32, k int) []model.ID {
	type distID struct {
		dist float32
		id   model.ID
	}

	results := make([]distID, len(vectors))
	for i, v := range vectors {
		results[i] = distID{
			dist: distance.SquaredL2(query, v),
			id:   ids[i],
		}
	}

	// Simple selection sort for top-k (sufficient for test sizes)
	for i := 0; i < k && i < len(results); i++ {
		minIdx := i
		for j := i + 1; j < len(results); j++ {
			if results[j].dist < results[minIdx].dist {
				minIdx = j
			}
		}
		results[i], results[minIdx] = results[minIdx], results[i]
	}

	topK := make([]model.ID, min(k, len(results)))
	for i := range topK {
		topK[i] = results[i].id
	}
	return topK
}

// recallAtK calculates the recall of search results against ground truth.
func recallAtK(results []model.Candidate, truth []model.ID) float64 {
	if len(truth) == 0 {
		return 0
	}

	truthSet := make(map[model.ID]struct{}, len(truth))
	for _, id := range truth {
		truthSet[id] = struct{}{}
	}

	hits := 0
	for _, r := range results {
		if _, ok := truthSet[r.ID]; ok {
			hits++
		}
	}

	return float64(hits) / float64(len(truth))
}
