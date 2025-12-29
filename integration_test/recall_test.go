package vecgo_test

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"sort"
	"testing"

	"github.com/hupe1980/vecgo/distance"
	"github.com/hupe1980/vecgo/index"
	"github.com/hupe1980/vecgo/index/diskann"
	"github.com/hupe1980/vecgo/index/flat"
	"github.com/hupe1980/vecgo/index/hnsw"
	"github.com/hupe1980/vecgo/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// cosineDistance computes 1 - cosine_similarity(a, b)
func cosineDistance(a, b []float32) float32 {
	dot := distance.Dot(a, b)
	normA := float32(math.Sqrt(float64(distance.Dot(a, a))))
	normB := float32(math.Sqrt(float64(distance.Dot(b, b))))
	if normA == 0 || normB == 0 {
		return 1.0
	}
	return 1.0 - dot/(normA*normB)
}

// RecallMetrics contains detailed recall statistics
type RecallMetrics struct {
	Recall1   float64 // Recall@1 (top result accuracy)
	Recall10  float64 // Recall@10
	Recall100 float64 // Recall@100 (if k >= 100)
	MRR       float64 // Mean Reciprocal Rank
	NDCG      float64 // Normalized Discounted Cumulative Gain
	AvgRecall float64 // Average recall across all queries
}

// GroundTruth stores exact nearest neighbors for validation
type GroundTruth struct {
	QueryIdx int
	TopK     []uint32  // Exact top-k neighbor IDs
	Dists    []float32 // Exact distances
}

// computeGroundTruth computes exact nearest neighbors using brute-force
func computeGroundTruth(vectors [][]float32, queries [][]float32, k int, distFunc func(a, b []float32) float32) []GroundTruth {
	groundTruth := make([]GroundTruth, len(queries))

	for qi, query := range queries {
		// Compute all distances
		type idDist struct {
			id   uint32
			dist float32
		}

		dists := make([]idDist, len(vectors))
		for vi, vec := range vectors {
			dists[vi] = idDist{id: uint32(vi), dist: distFunc(query, vec)}
		}

		// Sort by distance
		sort.Slice(dists, func(i, j int) bool {
			return dists[i].dist < dists[j].dist
		})

		// Take top-k
		topK := k
		if topK > len(dists) {
			topK = len(dists)
		}

		groundTruth[qi] = GroundTruth{
			QueryIdx: qi,
			TopK:     make([]uint32, topK),
			Dists:    make([]float32, topK),
		}
		for i := 0; i < topK; i++ {
			groundTruth[qi].TopK[i] = dists[i].id
			groundTruth[qi].Dists[i] = dists[i].dist
		}
	}

	return groundTruth
}

// computeRecall calculates recall@k for a single query
func computeRecall(predicted []uint32, groundTruth []uint32, k int) float64 {
	if len(groundTruth) == 0 || k == 0 {
		return 0
	}

	// Build ground truth set for top-k
	gtSet := make(map[uint32]struct{})
	topK := k
	if topK > len(groundTruth) {
		topK = len(groundTruth)
	}
	for i := 0; i < topK; i++ {
		gtSet[groundTruth[i]] = struct{}{}
	}

	// Count hits
	hits := 0
	checkK := k
	if checkK > len(predicted) {
		checkK = len(predicted)
	}
	for i := 0; i < checkK; i++ {
		if _, ok := gtSet[predicted[i]]; ok {
			hits++
		}
	}

	return float64(hits) / float64(topK)
}

// computeMRR calculates Mean Reciprocal Rank
func computeMRR(predicted []uint32, groundTruth []uint32) float64 {
	if len(groundTruth) == 0 || len(predicted) == 0 {
		return 0
	}

	// Find rank of first correct result
	for rank, pid := range predicted {
		if pid == groundTruth[0] {
			return 1.0 / float64(rank+1)
		}
	}

	return 0
}

// TestRecallConfig defines a recall test configuration
type TestRecallConfig struct {
	Name         string
	NumVectors   int
	Dimension    int
	NumQueries   int
	K            int
	DistanceType index.DistanceType
	MinRecall    float64 // Minimum acceptable recall@k
	MinRecall1   float64 // Minimum acceptable recall@1
	FilterRatio  float64 // 0 = no filter, 0.5 = filter 50% of vectors
}

var recallTestConfigs = []TestRecallConfig{
	// Small dataset - high recall expected
	{
		Name:         "Small_L2",
		NumVectors:   1000,
		Dimension:    64,
		NumQueries:   100,
		K:            10,
		DistanceType: index.DistanceTypeSquaredL2,
		MinRecall:    0.95,
		MinRecall1:   0.90,
	},
	// Medium dataset
	{
		Name:         "Medium_L2",
		NumVectors:   5000,
		Dimension:    128,
		NumQueries:   100,
		K:            10,
		DistanceType: index.DistanceTypeSquaredL2,
		MinRecall:    0.80,
		MinRecall1:   0.80,
	},
	// Cosine similarity
	{
		Name:         "Small_Cosine",
		NumVectors:   1000,
		Dimension:    64,
		NumQueries:   100,
		K:            10,
		DistanceType: index.DistanceTypeCosine,
		MinRecall:    0.95,
		MinRecall1:   0.90,
	},
	// Dot product
	{
		Name:         "Small_DotProduct",
		NumVectors:   1000,
		Dimension:    64,
		NumQueries:   100,
		K:            10,
		DistanceType: index.DistanceTypeDotProduct,
		MinRecall:    0.70,
		MinRecall1:   0.45,
	},
	// With filtering - NOW USES PRE-FILTERING (as of Dec 2024 refactoring)
	// Pre-filtering ensures correct recall by filtering during graph traversal
	{
		Name:         "Small_L2_Filtered",
		NumVectors:   1000,
		Dimension:    64,
		NumQueries:   100,
		K:            10,
		DistanceType: index.DistanceTypeSquaredL2,
		MinRecall:    0.85, // With pre-filtering, recall should be high (similar to unfiltered)
		MinRecall1:   0.90, // First result should be accurate
		FilterRatio:  0.5,
	},
	// Large K
	{
		Name:         "Small_L2_LargeK",
		NumVectors:   1000,
		Dimension:    64,
		NumQueries:   50,
		K:            100,
		DistanceType: index.DistanceTypeSquaredL2,
		MinRecall:    0.90,
		MinRecall1:   0.85,
	},
	// High dimension
	{
		Name:         "Small_L2_HighDim",
		NumVectors:   1000,
		Dimension:    768,
		NumQueries:   50,
		K:            10,
		DistanceType: index.DistanceTypeSquaredL2,
		MinRecall:    0.90,
		MinRecall1:   0.85,
	},
}

func getDistanceFunc(dt index.DistanceType) func(a, b []float32) float32 {
	switch dt {
	case index.DistanceTypeSquaredL2:
		return distance.SquaredL2
	case index.DistanceTypeCosine:
		return cosineDistance
	case index.DistanceTypeDotProduct:
		return func(a, b []float32) float32 {
			return -distance.Dot(a, b)
		}
	default:
		return distance.SquaredL2
	}
}

// ===========================================================================
// HNSW Recall Tests
// ===========================================================================

func TestHNSW_Recall(t *testing.T) {
	ctx := context.Background()

	for _, cfg := range recallTestConfigs {
		t.Run(cfg.Name, func(t *testing.T) {
			// Generate data
			rng := testutil.NewRNG(42)
			vectors := rng.GenerateRandomVectors(cfg.NumVectors, cfg.Dimension)
			queries := rng.GenerateRandomVectors(cfg.NumQueries, cfg.Dimension)

			// Compute ground truth
			distFunc := getDistanceFunc(cfg.DistanceType)
			groundTruth := computeGroundTruth(vectors, queries, cfg.K, distFunc)

			// Build HNSW index
			h, err := hnsw.New(func(o *hnsw.Options) {
				o.Dimension = cfg.Dimension
				o.DistanceType = cfg.DistanceType
				o.M = 16
				o.EF = 300
				o.Heuristic = true
			})
			require.NoError(t, err)

			for i, vec := range vectors {
				id, err := h.Insert(ctx, vec)
				require.NoError(t, err)
				assert.Equal(t, uint32(i), id)
			}

			// Create filter if needed
			var filter func(uint32) bool
			var filteredGT []GroundTruth

			if cfg.FilterRatio > 0 {
				// Filter out even IDs
				filter = func(id uint32) bool {
					return id%2 == 1 // Only odd IDs pass
				}

				// Recompute ground truth with filter
				filteredGT = computeFilteredGroundTruth(vectors, queries, cfg.K, distFunc, filter)
			} else {
				filter = func(id uint32) bool { return true }
				filteredGT = groundTruth
			}

			// Run searches and compute recall
			totalRecall := 0.0
			totalRecall1 := 0.0
			totalMRR := 0.0

			for qi, query := range queries {
				results, err := h.KNNSearch(ctx, query, cfg.K, &index.SearchOptions{
					EFSearch: 200,
					Filter:   filter,
				})
				require.NoError(t, err)

				// Extract IDs
				predicted := make([]uint32, len(results))
				for i, r := range results {
					predicted[i] = r.ID
				}

				// Compute metrics
				recall := computeRecall(predicted, filteredGT[qi].TopK, cfg.K)
				recall1 := computeRecall(predicted, filteredGT[qi].TopK, 1)
				mrr := computeMRR(predicted, filteredGT[qi].TopK)

				totalRecall += recall
				totalRecall1 += recall1
				totalMRR += mrr
			}

			avgRecall := totalRecall / float64(cfg.NumQueries)
			avgRecall1 := totalRecall1 / float64(cfg.NumQueries)
			avgMRR := totalMRR / float64(cfg.NumQueries)

			t.Logf("HNSW Recall@%d: %.4f, Recall@1: %.4f, MRR: %.4f",
				cfg.K, avgRecall, avgRecall1, avgMRR)

			assert.GreaterOrEqual(t, avgRecall, cfg.MinRecall,
				"Recall@%d too low: got %.4f, want >= %.4f", cfg.K, avgRecall, cfg.MinRecall)
			assert.GreaterOrEqual(t, avgRecall1, cfg.MinRecall1,
				"Recall@1 too low: got %.4f, want >= %.4f", avgRecall1, cfg.MinRecall1)
		})
	}
}

func TestHNSW_RecallVsEF(t *testing.T) {
	// Test how recall varies with EF parameter
	ctx := context.Background()
	rng := testutil.NewRNG(42)

	numVectors := 2000
	dimension := 64
	numQueries := 100
	k := 10

	vectors := rng.GenerateRandomVectors(numVectors, dimension)
	queries := rng.GenerateRandomVectors(numQueries, dimension)
	groundTruth := computeGroundTruth(vectors, queries, k, distance.SquaredL2)

	h, err := hnsw.New(func(o *hnsw.Options) {
		o.Dimension = dimension
		o.M = 16
		o.EF = 200
	})
	require.NoError(t, err)

	for _, vec := range vectors {
		_, err := h.Insert(ctx, vec)
		require.NoError(t, err)
	}

	efValues := []int{16, 32, 64, 128, 200, 400}
	expectedMinRecall := []float64{0.50, 0.70, 0.85, 0.95, 0.97, 0.99}

	for i, ef := range efValues {
		totalRecall := 0.0

		for qi, query := range queries {
			results, err := h.KNNSearch(ctx, query, k, &index.SearchOptions{
				EFSearch: ef,
				Filter:   func(id uint32) bool { return true },
			})
			require.NoError(t, err)

			predicted := make([]uint32, len(results))
			for j, r := range results {
				predicted[j] = r.ID
			}

			totalRecall += computeRecall(predicted, groundTruth[qi].TopK, k)
		}

		avgRecall := totalRecall / float64(numQueries)
		t.Logf("EF=%d: Recall@%d = %.4f", ef, k, avgRecall)

		assert.GreaterOrEqual(t, avgRecall, expectedMinRecall[i],
			"EF=%d: Recall too low: got %.4f, want >= %.4f", ef, avgRecall, expectedMinRecall[i])
	}
}

// ===========================================================================
// Flat Index Recall Tests (Should be 100% recall - it's exact search)
// ===========================================================================

func TestFlat_Recall(t *testing.T) {
	ctx := context.Background()

	for _, cfg := range recallTestConfigs {
		if cfg.FilterRatio > 0 {
			continue // Skip filtered tests for Flat - tested separately
		}

		t.Run(cfg.Name, func(t *testing.T) {
			rng := testutil.NewRNG(42)
			vectors := rng.GenerateRandomVectors(cfg.NumVectors, cfg.Dimension)
			queries := rng.GenerateRandomVectors(cfg.NumQueries, cfg.Dimension)

			distFunc := getDistanceFunc(cfg.DistanceType)
			groundTruth := computeGroundTruth(vectors, queries, cfg.K, distFunc)

			f, err := flat.New(func(o *flat.Options) {
				o.Dimension = cfg.Dimension
				o.DistanceType = cfg.DistanceType
			})
			require.NoError(t, err)

			for i, vec := range vectors {
				id, err := f.Insert(ctx, vec)
				require.NoError(t, err)
				assert.Equal(t, uint32(i), id)
			}

			totalRecall := 0.0
			for qi, query := range queries {
				results, err := f.KNNSearch(ctx, query, cfg.K, &index.SearchOptions{
					Filter: func(id uint32) bool { return true },
				})
				require.NoError(t, err)

				predicted := make([]uint32, len(results))
				for i, r := range results {
					predicted[i] = r.ID
				}

				recall := computeRecall(predicted, groundTruth[qi].TopK, cfg.K)
				totalRecall += recall
			}

			avgRecall := totalRecall / float64(cfg.NumQueries)
			t.Logf("Flat Recall@%d: %.4f", cfg.K, avgRecall)

			// Flat index should have perfect recall (it's exact search)
			assert.InDelta(t, 1.0, avgRecall, 0.001,
				"Flat index should have perfect recall, got %.4f", avgRecall)
		})
	}
}

// ===========================================================================
// DiskANN Recall Tests
// ===========================================================================

func TestDiskANN_Recall(t *testing.T) {
	ctx := context.Background()

	// Use subset of configs for DiskANN (it's slower to build)
	diskannConfigs := []TestRecallConfig{
		{
			Name:         "Small_L2",
			NumVectors:   1000,
			Dimension:    64,
			NumQueries:   50,
			K:            10,
			DistanceType: index.DistanceTypeSquaredL2,
			MinRecall:    0.85, // DiskANN may have slightly lower recall
			MinRecall1:   0.80,
		},
		{
			Name:         "Medium_L2",
			NumVectors:   2000,
			Dimension:    64,
			NumQueries:   50,
			K:            10,
			DistanceType: index.DistanceTypeSquaredL2,
			MinRecall:    0.85,
			MinRecall1:   0.80,
		},
	}

	for _, cfg := range diskannConfigs {
		t.Run(cfg.Name, func(t *testing.T) {
			rng := testutil.NewRNG(42)
			vectors := rng.GenerateRandomVectors(cfg.NumVectors, cfg.Dimension)
			queries := rng.GenerateRandomVectors(cfg.NumQueries, cfg.Dimension)

			distFunc := getDistanceFunc(cfg.DistanceType)
			groundTruth := computeGroundTruth(vectors, queries, cfg.K, distFunc)

			// Create temp directory for DiskANN files
			tmpDir, err := os.MkdirTemp("", "diskann-recall-test-*")
			require.NoError(t, err)
			defer os.RemoveAll(tmpDir)

			// Build DiskANN index
			indexPath := filepath.Join(tmpDir, "test-index")
			builder, err := diskann.NewBuilder(cfg.Dimension, cfg.DistanceType, indexPath, &diskann.Options{
				R:            32,
				L:            50,
				Alpha:        1.2,
				PQSubvectors: 8,
				PQCentroids:  256,
			})
			require.NoError(t, err)

			_, err = builder.AddBatch(vectors)
			require.NoError(t, err)

			err = builder.Build(ctx)
			require.NoError(t, err)

			// Open for search
			idx, err := diskann.Open(indexPath, nil)
			require.NoError(t, err)
			defer idx.Close()

			totalRecall := 0.0
			totalRecall1 := 0.0
			for qi, query := range queries {
				results, err := idx.KNNSearch(ctx, query, cfg.K, &index.SearchOptions{
					Filter: func(id uint32) bool { return true },
				})
				require.NoError(t, err)

				predicted := make([]uint32, len(results))
				for i, r := range results {
					predicted[i] = r.ID
				}

				recall := computeRecall(predicted, groundTruth[qi].TopK, cfg.K)
				recall1 := computeRecall(predicted, groundTruth[qi].TopK, 1)
				totalRecall += recall
				totalRecall1 += recall1
			}

			avgRecall := totalRecall / float64(cfg.NumQueries)
			avgRecall1 := totalRecall1 / float64(cfg.NumQueries)

			t.Logf("DiskANN Recall@%d: %.4f, Recall@1: %.4f", cfg.K, avgRecall, avgRecall1)

			assert.GreaterOrEqual(t, avgRecall, cfg.MinRecall,
				"Recall@%d too low: got %.4f, want >= %.4f", cfg.K, avgRecall, cfg.MinRecall)
			assert.GreaterOrEqual(t, avgRecall1, cfg.MinRecall1,
				"Recall@1 too low: got %.4f, want >= %.4f", avgRecall1, cfg.MinRecall1)
		})
	}
}

// ===========================================================================
// Incremental Index Recall Tests (HNSW + DiskANN mutable mode)
// ===========================================================================

func TestHNSW_RecallAfterUpdates(t *testing.T) {
	ctx := context.Background()
	rng := testutil.NewRNG(42)

	numVectors := 1000
	dimension := 64
	numQueries := 50
	k := 10

	vectors := rng.GenerateRandomVectors(numVectors, dimension)
	queries := rng.GenerateRandomVectors(numQueries, dimension)

	h, err := hnsw.New(func(o *hnsw.Options) {
		o.Dimension = dimension
		o.M = 16
		o.EF = 200
	})
	require.NoError(t, err)

	// Insert all vectors
	for _, vec := range vectors {
		_, err := h.Insert(ctx, vec)
		require.NoError(t, err)
	}

	// Delete 20% of vectors
	deleteIDs := make([]uint32, 0, numVectors/5)
	for i := uint32(0); i < uint32(numVectors); i += 5 {
		err := h.Delete(ctx, i)
		require.NoError(t, err)
		deleteIDs = append(deleteIDs, i)
	}

	// Compute ground truth excluding deleted vectors
	deletedSet := make(map[uint32]struct{})
	for _, id := range deleteIDs {
		deletedSet[id] = struct{}{}
	}

	filter := func(id uint32) bool {
		_, deleted := deletedSet[id]
		return !deleted
	}

	groundTruth := computeFilteredGroundTruth(vectors, queries, k, distance.SquaredL2, filter)

	// Search and measure recall
	totalRecall := 0.0
	for qi, query := range queries {
		results, err := h.KNNSearch(ctx, query, k, &index.SearchOptions{
			EFSearch: 200,
			Filter:   filter,
		})
		require.NoError(t, err)

		predicted := make([]uint32, len(results))
		for i, r := range results {
			predicted[i] = r.ID
		}

		recall := computeRecall(predicted, groundTruth[qi].TopK, k)
		totalRecall += recall
	}

	avgRecall := totalRecall / float64(numQueries)
	t.Logf("HNSW Recall@%d after deletions: %.4f", k, avgRecall)

	// Should still maintain good recall after deletions
	assert.GreaterOrEqual(t, avgRecall, 0.85,
		"Recall after deletions too low: got %.4f, want >= 0.85", avgRecall)
}

func TestHNSW_RecallAfterMixedOperations(t *testing.T) {
	ctx := context.Background()
	rng := testutil.NewRNG(42)

	dimension := 64
	k := 10

	h, err := hnsw.New(func(o *hnsw.Options) {
		o.Dimension = dimension
		o.M = 16
		o.EF = 200
	})
	require.NoError(t, err)

	// Track all vectors by their actual IDs
	allVectors := make(map[uint32][]float32)

	// Phase 1: Insert 500 vectors
	vectors1 := rng.GenerateRandomVectors(500, dimension)
	for _, vec := range vectors1 {
		id, err := h.Insert(ctx, vec)
		require.NoError(t, err)
		allVectors[id] = vec
	}

	// Phase 2: Delete 100 vectors (IDs 0-99)
	for i := uint32(0); i < 100; i++ {
		err := h.Delete(ctx, i)
		require.NoError(t, err)
		delete(allVectors, i)
	}

	// Phase 3: Insert 500 more vectors (they may reuse deleted IDs)
	vectors2 := rng.GenerateRandomVectors(500, dimension)
	for _, vec := range vectors2 {
		id, err := h.Insert(ctx, vec)
		require.NoError(t, err)
		allVectors[id] = vec
	}

	// Build ground truth from the actual current state
	currentVectors := make([][]float32, 0, len(allVectors))
	currentIDs := make([]uint32, 0, len(allVectors))
	for id, vec := range allVectors {
		currentIDs = append(currentIDs, id)
		currentVectors = append(currentVectors, vec)
	}

	// Verify the index has the expected size
	assert.Equal(t, 900, len(currentVectors), "Expected 900 vectors in index")

	// Generate queries and compute ground truth manually
	queries := rng.GenerateRandomVectors(50, dimension)
	totalRecall := 0.0

	for _, query := range queries {
		// Compute ground truth
		type idDist struct {
			id   uint32
			dist float32
		}

		dists := make([]idDist, len(currentVectors))
		for i := range currentVectors {
			dists[i] = idDist{
				id:   currentIDs[i],
				dist: distance.SquaredL2(query, currentVectors[i]),
			}
		}

		sort.Slice(dists, func(i, j int) bool {
			return dists[i].dist < dists[j].dist
		})

		topK := k
		if topK > len(dists) {
			topK = len(dists)
		}
		gtTopK := make([]uint32, topK)
		for i := 0; i < topK; i++ {
			gtTopK[i] = dists[i].id
		}

		// Search
		results, err := h.KNNSearch(ctx, query, k, &index.SearchOptions{
			EFSearch: 200,
			Filter:   func(id uint32) bool { return true },
		})
		require.NoError(t, err)

		predicted := make([]uint32, len(results))
		for i, r := range results {
			predicted[i] = r.ID
		}

		recall := computeRecall(predicted, gtTopK, k)
		totalRecall += recall
	}

	avgRecall := totalRecall / float64(50)
	t.Logf("HNSW Recall@%d after mixed ops: %.4f", k, avgRecall)

	assert.GreaterOrEqual(t, avgRecall, 0.85,
		"Recall after mixed operations too low: got %.4f, want >= 0.85", avgRecall)
}

// ===========================================================================
// Edge Case Tests
// ===========================================================================

func TestRecall_EdgeCases(t *testing.T) {
	ctx := context.Background()

	t.Run("SingleVector", func(t *testing.T) {
		h, err := hnsw.New(func(o *hnsw.Options) {
			o.Dimension = 64
			o.M = 16
			o.EF = 200
		})
		require.NoError(t, err)

		vec := make([]float32, 64)
		for i := range vec {
			vec[i] = float32(i)
		}

		_, err = h.Insert(ctx, vec)
		require.NoError(t, err)

		results, err := h.KNNSearch(ctx, vec, 10, &index.SearchOptions{
			Filter: func(id uint32) bool { return true },
		})
		require.NoError(t, err)

		assert.Len(t, results, 1)
		assert.Equal(t, uint32(0), results[0].ID)
	})

	t.Run("DuplicateVectors", func(t *testing.T) {
		h, err := hnsw.New(func(o *hnsw.Options) {
			o.Dimension = 64
			o.M = 16
			o.EF = 200
		})
		require.NoError(t, err)

		vec := make([]float32, 64)
		for i := range vec {
			vec[i] = 1.0
		}

		// Insert same vector 10 times
		for i := 0; i < 10; i++ {
			_, err = h.Insert(ctx, vec)
			require.NoError(t, err)
		}

		results, err := h.KNNSearch(ctx, vec, 5, &index.SearchOptions{
			Filter: func(id uint32) bool { return true },
		})
		require.NoError(t, err)

		assert.Len(t, results, 5)

		// All distances should be 0
		for _, r := range results {
			assert.InDelta(t, 0, r.Distance, 1e-6)
		}
	})

	t.Run("ZeroVector", func(t *testing.T) {
		h, err := hnsw.New(func(o *hnsw.Options) {
			o.Dimension = 64
			o.M = 16
			o.EF = 200
		})
		require.NoError(t, err)

		rng := testutil.NewRNG(42)
		vectors := rng.GenerateRandomVectors(100, 64)
		for _, vec := range vectors {
			_, err := h.Insert(ctx, vec)
			require.NoError(t, err)
		}

		// Search with zero vector
		zeroVec := make([]float32, 64)
		results, err := h.KNNSearch(ctx, zeroVec, 10, &index.SearchOptions{
			Filter: func(id uint32) bool { return true },
		})
		require.NoError(t, err)

		assert.Len(t, results, 10)
	})

	t.Run("KLargerThanDataset", func(t *testing.T) {
		h, err := hnsw.New(func(o *hnsw.Options) {
			o.Dimension = 64
			o.M = 16
			o.EF = 200
		})
		require.NoError(t, err)

		rng := testutil.NewRNG(42)
		vectors := rng.GenerateRandomVectors(5, 64)
		for _, vec := range vectors {
			_, err := h.Insert(ctx, vec)
			require.NoError(t, err)
		}

		query := rng.GenerateRandomVectors(1, 64)[0]
		results, err := h.KNNSearch(ctx, query, 100, &index.SearchOptions{
			Filter: func(id uint32) bool { return true },
		})
		require.NoError(t, err)

		assert.Len(t, results, 5) // Should return all 5 vectors
	})
}

// ===========================================================================
// Helper functions
// ===========================================================================

func computeFilteredGroundTruth(vectors [][]float32, queries [][]float32, k int, distFunc func(a, b []float32) float32, filter func(uint32) bool) []GroundTruth {
	groundTruth := make([]GroundTruth, len(queries))

	for qi, query := range queries {
		type idDist struct {
			id   uint32
			dist float32
		}

		dists := make([]idDist, 0, len(vectors))
		for vi, vec := range vectors {
			if filter(uint32(vi)) {
				dists = append(dists, idDist{id: uint32(vi), dist: distFunc(query, vec)})
			}
		}

		sort.Slice(dists, func(i, j int) bool {
			return dists[i].dist < dists[j].dist
		})

		topK := k
		if topK > len(dists) {
			topK = len(dists)
		}

		groundTruth[qi] = GroundTruth{
			QueryIdx: qi,
			TopK:     make([]uint32, topK),
			Dists:    make([]float32, topK),
		}
		for i := 0; i < topK; i++ {
			groundTruth[qi].TopK[i] = dists[i].id
			groundTruth[qi].Dists[i] = dists[i].dist
		}
	}

	return groundTruth
}

// ===========================================================================
// Benchmark recall computation overhead
// ===========================================================================

func BenchmarkGroundTruthComputation(b *testing.B) {
	rng := testutil.NewRNG(42)
	vectors := rng.GenerateRandomVectors(10000, 128)
	queries := rng.GenerateRandomVectors(100, 128)

	b.ResetTimer()
	for b.Loop() {
		_ = computeGroundTruth(vectors, queries, 10, distance.SquaredL2)
	}
}

// ===========================================================================
// Recall Report (prints comprehensive metrics)
// ===========================================================================

func TestRecallReport(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping recall report in short mode")
	}

	ctx := context.Background()
	rng := testutil.NewRNG(42)

	numVectors := 5000
	dimension := 128
	numQueries := 200
	k := 10

	vectors := rng.GenerateRandomVectors(numVectors, dimension)
	queries := rng.GenerateRandomVectors(numQueries, dimension)
	groundTruth := computeGroundTruth(vectors, queries, k, distance.SquaredL2)

	// Build indexes
	t.Log("Building indexes...")

	// HNSW
	h, err := hnsw.New(func(o *hnsw.Options) {
		o.Dimension = dimension
		o.M = 16
		o.EF = 200
	})
	require.NoError(t, err)

	for _, vec := range vectors {
		_, _ = h.Insert(ctx, vec)
	}

	// Flat
	f, err := flat.New(func(o *flat.Options) {
		o.Dimension = dimension
	})
	require.NoError(t, err)

	for _, vec := range vectors {
		_, _ = f.Insert(ctx, vec)
	}

	t.Log("\n=== RECALL REPORT ===")
	t.Logf("Dataset: %d vectors, %d dimensions", numVectors, dimension)
	t.Logf("Queries: %d, K: %d\n", numQueries, k)

	// HNSW metrics at different EF values
	t.Log("HNSW Recall vs EF:")
	for _, ef := range []int{32, 64, 128, 200, 400} {
		metrics := computeIndexMetrics(ctx, h, queries, groundTruth, k, ef)
		t.Logf("  EF=%3d: Recall@%d=%.4f, Recall@1=%.4f, MRR=%.4f",
			ef, k, metrics.AvgRecall, metrics.Recall1, metrics.MRR)
	}

	// Flat metrics
	flatMetrics := computeFlatMetrics(ctx, f, queries, groundTruth, k)
	t.Logf("\nFlat (exact): Recall@%d=%.4f, Recall@1=%.4f, MRR=%.4f",
		k, flatMetrics.AvgRecall, flatMetrics.Recall1, flatMetrics.MRR)
}

func computeIndexMetrics(ctx context.Context, h *hnsw.HNSW, queries [][]float32, gt []GroundTruth, k, ef int) RecallMetrics {
	var totalRecall, totalRecall1, totalMRR float64

	for qi, query := range queries {
		results, _ := h.KNNSearch(ctx, query, k, &index.SearchOptions{
			EFSearch: ef,
			Filter:   func(id uint32) bool { return true },
		})

		predicted := make([]uint32, len(results))
		for i, r := range results {
			predicted[i] = r.ID
		}

		totalRecall += computeRecall(predicted, gt[qi].TopK, k)
		totalRecall1 += computeRecall(predicted, gt[qi].TopK, 1)
		totalMRR += computeMRR(predicted, gt[qi].TopK)
	}

	n := float64(len(queries))
	return RecallMetrics{
		AvgRecall: totalRecall / n,
		Recall1:   totalRecall1 / n,
		MRR:       totalMRR / n,
	}
}

func computeFlatMetrics(ctx context.Context, f *flat.Flat, queries [][]float32, gt []GroundTruth, k int) RecallMetrics {
	var totalRecall, totalRecall1, totalMRR float64

	for qi, query := range queries {
		results, _ := f.KNNSearch(ctx, query, k, &index.SearchOptions{
			Filter: func(id uint32) bool { return true },
		})

		predicted := make([]uint32, len(results))
		for i, r := range results {
			predicted[i] = r.ID
		}

		totalRecall += computeRecall(predicted, gt[qi].TopK, k)
		totalRecall1 += computeRecall(predicted, gt[qi].TopK, 1)
		totalMRR += computeMRR(predicted, gt[qi].TopK)
	}

	n := float64(len(queries))
	return RecallMetrics{
		AvgRecall: totalRecall / n,
		Recall1:   totalRecall1 / n,
		MRR:       totalMRR / n,
	}
}

// Suppress unused import warning
var _ = math.Sqrt
