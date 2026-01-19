package manifest

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNumericFieldStats_Variance(t *testing.T) {
	tests := []struct {
		name   string
		stats  NumericFieldStats
		expect float64
	}{
		{
			name:   "empty/single value",
			stats:  NumericFieldStats{Count: 1, Sum: 10, SumSq: 100},
			expect: 0,
		},
		{
			name:   "two values - variance",
			stats:  NumericFieldStats{Count: 2, Sum: 10, SumSq: 58}, // values 3 and 7: mean=5, var=4
			expect: 4,
		},
		{
			name:   "uniform values",
			stats:  NumericFieldStats{Count: 3, Sum: 15, SumSq: 75}, // 5,5,5: var=0
			expect: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.stats.Variance()
			assert.InDelta(t, tt.expect, got, 0.001)
		})
	}
}

func TestNumericFieldStats_StdDev(t *testing.T) {
	stats := NumericFieldStats{Count: 2, Sum: 10, SumSq: 58} // var=4
	assert.InDelta(t, 2.0, stats.StdDev(), 0.001)
}

func TestSegmentStats_CanPruneNumeric(t *testing.T) {
	stats := &SegmentStats{
		Numeric: map[string]NumericFieldStats{
			"price": {Min: 10, Max: 100},
			"qty":   {Min: 5, Max: 5}, // Single value
		},
	}

	tests := []struct {
		name     string
		field    string
		op       string
		queryVal float64
		expect   bool
	}{
		// gt: prune if max <= queryVal
		{"gt_can_prune", "price", "gt", 100, true},
		{"gt_cannot_prune", "price", "gt", 50, false},

		// gte: prune if max < queryVal
		{"gte_can_prune", "price", "gte", 101, true},
		{"gte_cannot_prune", "price", "gte", 100, false},

		// lt: prune if min >= queryVal
		{"lt_can_prune", "price", "lt", 10, true},
		{"lt_cannot_prune", "price", "lt", 50, false},

		// lte: prune if min > queryVal
		{"lte_can_prune", "price", "lte", 9, true},
		{"lte_cannot_prune", "price", "lte", 10, false},

		// eq: prune if outside range
		{"eq_can_prune_low", "price", "eq", 5, true},
		{"eq_can_prune_high", "price", "eq", 150, true},
		{"eq_cannot_prune", "price", "eq", 50, false},

		// neq: only prune if single value equals query
		{"neq_can_prune", "qty", "neq", 5, true},
		{"neq_cannot_prune", "qty", "neq", 10, false},
		{"neq_multi_value", "price", "neq", 50, false},

		// between: always false (use range method)
		{"between_no_prune", "price", "between", 50, false},

		// missing field: can prune
		{"missing_field", "unknown", "eq", 50, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := stats.CanPruneNumeric(tt.field, tt.op, tt.queryVal)
			assert.Equal(t, tt.expect, got)
		})
	}

	// nil stats should not prune
	t.Run("nil_stats", func(t *testing.T) {
		var nilStats *SegmentStats
		assert.False(t, nilStats.CanPruneNumeric("price", "eq", 50))
	})
}

func TestSegmentStats_CanPruneNumericRange(t *testing.T) {
	stats := &SegmentStats{
		Numeric: map[string]NumericFieldStats{
			"price": {Min: 10, Max: 100},
		},
	}

	tests := []struct {
		name     string
		field    string
		lo, hi   float64
		incLo    bool
		incHi    bool
		canPrune bool
	}{
		// No overlap cases
		{"range_above", "price", 200, 300, true, true, true},
		{"range_below", "price", 0, 5, true, true, true},

		// Overlap cases
		{"full_overlap", "price", 50, 80, true, true, false},
		{"partial_overlap_low", "price", 5, 50, true, true, false},
		{"partial_overlap_high", "price", 50, 200, true, true, false},

		// Edge cases with include flags
		{"edge_touch_high_inclusive", "price", 100, 200, true, true, false},
		{"edge_touch_high_exclusive", "price", 100, 200, false, true, true},
		{"edge_touch_low_inclusive", "price", 0, 10, true, true, false},
		{"edge_touch_low_exclusive", "price", 0, 10, true, false, true},

		// Missing field
		{"missing_field", "unknown", 0, 100, true, true, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := stats.CanPruneNumericRange(tt.field, tt.lo, tt.hi, tt.incLo, tt.incHi)
			assert.Equal(t, tt.canPrune, got)
		})
	}

	// nil stats
	t.Run("nil_stats", func(t *testing.T) {
		var nilStats *SegmentStats
		assert.False(t, nilStats.CanPruneNumericRange("price", 0, 100, true, true))
	})
}

func TestSegmentStats_HasField(t *testing.T) {
	stats := &SegmentStats{
		HasFields: map[string]bool{
			"category": true,
			"price":    true,
		},
	}

	assert.True(t, stats.HasField("category"))
	assert.True(t, stats.HasField("price"))
	assert.False(t, stats.HasField("unknown"))

	// nil stats
	var nilStats *SegmentStats
	assert.False(t, nilStats.HasField("category"))

	// nil HasFields map
	emptyStats := &SegmentStats{}
	assert.False(t, emptyStats.HasField("category"))
}

func TestSegmentStats_EstimateSelectivity(t *testing.T) {
	stats := &SegmentStats{
		Numeric: map[string]NumericFieldStats{
			"price": {Min: 0, Max: 100},
			"qty":   {Min: 10, Max: 10}, // Single value
		},
	}

	t.Run("uniform_distribution", func(t *testing.T) {
		// Without histogram, assumes uniform
		// Range [25, 75] is 50% of [0, 100]
		sel := stats.EstimateSelectivity("price", 25, 75)
		assert.InDelta(t, 0.5, sel, 0.01)
	})

	t.Run("partial_overlap", func(t *testing.T) {
		// Range [-50, 50] clamps to [0, 50] = 50%
		sel := stats.EstimateSelectivity("price", -50, 50)
		assert.InDelta(t, 0.5, sel, 0.01)
	})

	t.Run("no_overlap", func(t *testing.T) {
		sel := stats.EstimateSelectivity("price", 200, 300)
		assert.InDelta(t, 0.0, sel, 0.01)
	})

	t.Run("single_value_match", func(t *testing.T) {
		sel := stats.EstimateSelectivity("qty", 5, 15)
		assert.InDelta(t, 1.0, sel, 0.01)
	})

	t.Run("single_value_no_match", func(t *testing.T) {
		sel := stats.EstimateSelectivity("qty", 20, 30)
		assert.InDelta(t, 0.0, sel, 0.01)
	})

	t.Run("missing_field", func(t *testing.T) {
		sel := stats.EstimateSelectivity("unknown", 0, 100)
		assert.InDelta(t, 0.0, sel, 0.01)
	})

	t.Run("nil_stats", func(t *testing.T) {
		var nilStats *SegmentStats
		sel := nilStats.EstimateSelectivity("price", 0, 100)
		assert.InDelta(t, 1.0, sel, 0.01) // Conservative
	})
}

func TestSegmentStats_EstimateCategoricalSelectivity(t *testing.T) {
	stats := &SegmentStats{
		LiveRows: 1000,
		Categorical: map[string]CategoricalStats{
			"category": {
				DistinctCount: 10,
				TopK: []ValueFreq{
					{Value: "electronics", Count: 500},
					{Value: "books", Count: 200},
					{Value: "toys", Count: 100},
				},
			},
		},
	}

	t.Run("value_in_topk", func(t *testing.T) {
		sel := stats.EstimateCategoricalSelectivity("category", "electronics")
		assert.InDelta(t, 0.5, sel, 0.01) // 500/1000
	})

	t.Run("value_not_in_topk", func(t *testing.T) {
		sel := stats.EstimateCategoricalSelectivity("category", "furniture")
		assert.InDelta(t, 0.1, sel, 0.01) // 1/10 distinct
	})

	t.Run("missing_field", func(t *testing.T) {
		sel := stats.EstimateCategoricalSelectivity("unknown", "value")
		assert.InDelta(t, 0.0, sel, 0.01)
	})

	t.Run("nil_stats", func(t *testing.T) {
		var nilStats *SegmentStats
		sel := nilStats.EstimateCategoricalSelectivity("category", "value")
		assert.InDelta(t, 1.0, sel, 0.01) // Conservative
	})
}

func TestSegmentStats_IsPure(t *testing.T) {
	stats := &SegmentStats{
		Categorical: map[string]CategoricalStats{
			"category": {
				DominantValue: "electronics",
				DominantRatio: 0.95,
			},
			"status": {
				DominantValue: "active",
				DominantRatio: 0.5,
			},
		},
	}

	t.Run("pure_segment", func(t *testing.T) {
		val, pure := stats.IsPure("category", 0.9)
		assert.True(t, pure)
		assert.Equal(t, "electronics", val)
	})

	t.Run("not_pure_below_threshold", func(t *testing.T) {
		val, pure := stats.IsPure("status", 0.9)
		assert.False(t, pure)
		assert.Empty(t, val)
	})

	t.Run("missing_field", func(t *testing.T) {
		val, pure := stats.IsPure("unknown", 0.9)
		assert.False(t, pure)
		assert.Empty(t, val)
	})

	t.Run("nil_stats", func(t *testing.T) {
		var nilStats *SegmentStats
		val, pure := nilStats.IsPure("category", 0.9)
		assert.False(t, pure)
		assert.Empty(t, val)
	})
}

func TestSegmentStats_CanPruneCategorical(t *testing.T) {
	stats := &SegmentStats{
		Categorical: map[string]CategoricalStats{
			"category": {
				DominantValue: "electronics",
				DominantRatio: 0.999,
				DistinctCount: 3,
				TopK: []ValueFreq{
					{Value: "electronics", Count: 999},
					{Value: "books", Count: 1},
					{Value: "toys", Count: 0},
				},
			},
			"status": {
				DominantValue: "active",
				DominantRatio: 0.5,
				DistinctCount: 100, // High cardinality
			},
			"type": {
				DominantValue: "typeA",
				DominantRatio: 0.6, // Not pure enough to auto-prune
				DistinctCount: 3,
				TopK: []ValueFreq{
					{Value: "typeA", Count: 60},
					{Value: "typeB", Count: 30},
					{Value: "typeC", Count: 10},
				},
			},
		},
	}

	t.Run("prune_pure_different_value", func(t *testing.T) {
		// Segment is pure "electronics", query for "furniture" can be pruned
		assert.True(t, stats.CanPruneCategorical("category", "furniture"))
	})

	t.Run("no_prune_matching_dominant", func(t *testing.T) {
		assert.False(t, stats.CanPruneCategorical("category", "electronics"))
	})

	t.Run("prune_topk_complete_value_missing", func(t *testing.T) {
		// TopK has all 3 distinct values, "furniture" not in TopK
		assert.True(t, stats.CanPruneCategorical("category", "furniture"))
	})

	t.Run("no_prune_value_in_topk", func(t *testing.T) {
		// "typeB" is in TopK and segment is not pure for different value
		assert.False(t, stats.CanPruneCategorical("type", "typeB"))
	})

	t.Run("no_prune_high_cardinality", func(t *testing.T) {
		// High cardinality, TopK incomplete, can't prune
		assert.False(t, stats.CanPruneCategorical("status", "pending"))
	})

	t.Run("prune_missing_field", func(t *testing.T) {
		assert.True(t, stats.CanPruneCategorical("unknown", "value"))
	})

	t.Run("nil_stats", func(t *testing.T) {
		var nilStats *SegmentStats
		assert.False(t, nilStats.CanPruneCategorical("category", "value"))
	})
}

func TestSegmentStats_GetHistogramSelectivity(t *testing.T) {
	// Create stats with populated histogram
	stats := &SegmentStats{
		Numeric: map[string]NumericFieldStats{
			"price": {
				Min: 0,
				Max: 100,
				// 1000 values distributed across 16 bins
				Histogram: func() [16]uint32 {
					var h [16]uint32
					// Put most values in first few bins (log-scaled)
					h[0] = 200
					h[1] = 150
					h[2] = 100
					h[3] = 100
					h[4] = 80
					h[5] = 70
					h[6] = 60
					h[7] = 50
					h[8] = 40
					h[9] = 35
					h[10] = 30
					h[11] = 25
					h[12] = 20
					h[13] = 15
					h[14] = 15
					h[15] = 10
					return h
				}(),
			},
			"empty": {Min: 0, Max: 100}, // No histogram
		},
	}

	t.Run("histogram_selectivity", func(t *testing.T) {
		// Query a range and verify selectivity is estimated
		sel := stats.GetHistogramSelectivity("price", 0, 50)
		assert.Greater(t, sel, 0.0)
		assert.LessOrEqual(t, sel, 1.0)
	})

	t.Run("no_histogram", func(t *testing.T) {
		sel := stats.GetHistogramSelectivity("empty", 0, 50)
		assert.Equal(t, -1.0, sel) // Not available
	})

	t.Run("missing_field", func(t *testing.T) {
		sel := stats.GetHistogramSelectivity("unknown", 0, 100)
		assert.Equal(t, 0.0, sel)
	})

	t.Run("nil_stats", func(t *testing.T) {
		var nilStats *SegmentStats
		sel := nilStats.GetHistogramSelectivity("price", 0, 100)
		assert.Equal(t, -1.0, sel)
	})
}

func TestSegmentStats_ShouldUseBitmap(t *testing.T) {
	stats := &SegmentStats{
		LiveRows: 10000,
		Categorical: map[string]CategoricalStats{
			"pure_field": {
				DominantValue: "value",
				DominantRatio: 0.9,
				DistinctCount: 10,
			},
			"low_cardinality": {
				DominantValue: "yes",
				DominantRatio: 0.6,
				DistinctCount: 2,
			},
			"high_cardinality": {
				DominantValue: "cat1",
				DominantRatio: 0.1,
				DistinctCount: 100,
			},
		},
	}

	t.Run("pure_segment_no_bitmap", func(t *testing.T) {
		// Pure segment with matching value: scan is faster
		assert.False(t, stats.ShouldUseBitmap("pure_field", "value", 10000))
	})

	t.Run("low_cardinality_no_bitmap", func(t *testing.T) {
		assert.False(t, stats.ShouldUseBitmap("low_cardinality", "yes", 10000))
	})

	t.Run("high_cardinality_use_bitmap", func(t *testing.T) {
		assert.True(t, stats.ShouldUseBitmap("high_cardinality", "cat99", 10000))
	})

	t.Run("missing_field_no_bitmap", func(t *testing.T) {
		assert.False(t, stats.ShouldUseBitmap("unknown", "value", 10000))
	})

	t.Run("nil_stats_default_bitmap", func(t *testing.T) {
		var nilStats *SegmentStats
		assert.True(t, nilStats.ShouldUseBitmap("field", "value", 10000))
	})
}

func TestSegmentStats_NeedsCompaction(t *testing.T) {
	t.Run("high_deleted_ratio", func(t *testing.T) {
		stats := &SegmentStats{
			TotalRows:    1000,
			LiveRows:     500,
			DeletedRatio: 0.5,
		}
		assert.True(t, stats.NeedsCompaction(0.3))
	})

	t.Run("low_deleted_ratio", func(t *testing.T) {
		stats := &SegmentStats{
			TotalRows:    1000,
			LiveRows:     900,
			DeletedRatio: 0.1,
		}
		assert.False(t, stats.NeedsCompaction(0.3))
	})

	t.Run("nil_stats", func(t *testing.T) {
		var nilStats *SegmentStats
		assert.False(t, nilStats.NeedsCompaction(0.3))
	})
}

func TestSegmentStats_IsLowEntropy(t *testing.T) {
	t.Run("low_entropy", func(t *testing.T) {
		stats := &SegmentStats{
			FilterEntropy: 0.2,
		}
		assert.True(t, stats.IsLowEntropy(0.3))
	})

	t.Run("high_entropy", func(t *testing.T) {
		stats := &SegmentStats{
			FilterEntropy: 0.8,
		}
		assert.False(t, stats.IsLowEntropy(0.3))
	})

	t.Run("nil_stats", func(t *testing.T) {
		var nilStats *SegmentStats
		assert.False(t, nilStats.IsLowEntropy(0.3))
	})
}

func TestSegmentStats_CanPruneByDistance(t *testing.T) {
	// CanPruneByDistance uses Radius95 for triangle inequality bound
	t.Run("query_outside_radius", func(t *testing.T) {
		stats := &SegmentStats{
			Vector: &VectorStats{
				Radius95: 10.0, // 95% of vectors within this radius
			},
		}
		// Distance to centroid is 15, radius95 is 10
		// Best case distance is 15-10=5, threshold is 3
		assert.True(t, stats.CanPruneByDistance(15.0, 3.0))
	})

	t.Run("query_within_radius", func(t *testing.T) {
		stats := &SegmentStats{
			Vector: &VectorStats{
				Radius95: 10.0,
			},
		}
		// Distance to centroid is 5, radius95 is 10
		// Best case distance is 0 (query inside cluster)
		assert.False(t, stats.CanPruneByDistance(5.0, 3.0))
	})

	t.Run("zero_radius95", func(t *testing.T) {
		stats := &SegmentStats{
			Vector: &VectorStats{
				Radius95: 0, // Uninitalized
			},
		}
		assert.False(t, stats.CanPruneByDistance(15.0, 3.0))
	})

	t.Run("no_vector_stats", func(t *testing.T) {
		stats := &SegmentStats{}
		assert.False(t, stats.CanPruneByDistance(15.0, 3.0))
	})

	t.Run("nil_stats", func(t *testing.T) {
		var nilStats *SegmentStats
		assert.False(t, nilStats.CanPruneByDistance(15.0, 3.0))
	})
}

func TestSegmentStats_SegmentPriority(t *testing.T) {
	// SegmentPriority: higher score = search this segment first
	// Closer query (normalizedDist < 2) gets bonus score
	stats := &SegmentStats{
		Vector: &VectorStats{
			AvgDistanceToCentroid: 5.0,
			Radius95:              8.0,
		},
	}

	// Both queries get positive priority scores
	p1 := stats.SegmentPriority(3.0)  // Close query (3/5 = 0.6 < 2)
	p2 := stats.SegmentPriority(15.0) // Far query (15/5 = 3 > 2)

	// Closer query should have HIGHER priority (search first)
	assert.Greater(t, p1, p2)
	assert.Greater(t, p1, float32(0))
	assert.Greater(t, p2, float32(0))

	// nil stats returns base priority (1.0) with no bonuses
	var nilStats *SegmentStats
	assert.Equal(t, float32(0), nilStats.SegmentPriority(5.0))
}

func TestBloomFilter(t *testing.T) {
	t.Run("basic_operations", func(t *testing.T) {
		bf := NewBloomFilterForSize(1000)
		require.NotNil(t, bf)

		bf.Add("hello")
		bf.Add("world")

		assert.True(t, bf.MayContain("hello"))
		assert.True(t, bf.MayContain("world"))
		// False positives are possible but unlikely for small sets
		// "nothere" should likely return false
		// Don't assert false - bloom filters have FP rate
	})

	t.Run("empty_bloom", func(t *testing.T) {
		bf := NewBloomFilterForSize(100)
		assert.False(t, bf.MayContain("anything"))
	})

	t.Run("count_tracking", func(t *testing.T) {
		bf := NewBloomFilterForSize(100)
		assert.Equal(t, uint32(0), bf.Count())

		bf.Add("a")
		bf.Add("b")
		assert.Equal(t, uint32(2), bf.Count())
	})

	t.Run("false_positive_rate", func(t *testing.T) {
		bf := NewBloomFilterForSize(1000)
		for i := 0; i < 500; i++ {
			bf.Add(string(rune('a' + i)))
		}
		// Should be around 1% FPR for half-full filter
		fpr := bf.EstimatedFalsePositiveRate()
		assert.Greater(t, fpr, 0.0)
		assert.Less(t, fpr, 0.1) // Should be well under 10%
	})
}

func TestNewSegmentStats(t *testing.T) {
	stats := NewSegmentStats()
	require.NotNil(t, stats)

	// Verify maps are initialized
	assert.NotNil(t, stats.Numeric)
	assert.NotNil(t, stats.Categorical)
	assert.NotNil(t, stats.HasFields)
}

// Test that stats work correctly with edge cases
func TestSegmentStats_EdgeCases(t *testing.T) {
	t.Run("zero_live_rows", func(t *testing.T) {
		stats := &SegmentStats{
			TotalRows:    100,
			LiveRows:     0,
			DeletedRatio: 1.0,
		}
		assert.True(t, stats.NeedsCompaction(0.3))
	})

	t.Run("inf_values_in_numeric", func(t *testing.T) {
		stats := &SegmentStats{
			Numeric: map[string]NumericFieldStats{
				"field": {Min: math.Inf(-1), Max: math.Inf(1)},
			},
		}
		// Should handle infinity gracefully
		assert.False(t, stats.CanPruneNumeric("field", "eq", 100))
	})
}
