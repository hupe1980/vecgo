package manifest

import (
	"math"
	"testing"

	"github.com/hupe1980/vecgo/metadata"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestStatsCollector_Empty(t *testing.T) {
	sc := NewStatsCollector(128, true)
	stats := sc.Finalize()

	// Empty collector returns nil
	assert.Nil(t, stats)
}

func TestStatsCollector_BasicCounts(t *testing.T) {
	sc := NewStatsCollector(4, false) // No vector tracking

	// Add 10 live rows
	for i := 0; i < 10; i++ {
		sc.Add([]float32{float32(i), 0, 0, 0}, metadata.Document{
			"id": metadata.Int(int64(i)),
		})
	}

	// Add 2 deleted rows
	sc.AddDeleted()
	sc.AddDeleted()

	stats := sc.Finalize()
	require.NotNil(t, stats)

	assert.Equal(t, uint32(12), stats.TotalRows)
	assert.Equal(t, uint32(10), stats.LiveRows)
	assert.InDelta(t, float32(2)/float32(12), stats.DeletedRatio, 0.001)
}

func TestStatsCollector_NumericFields(t *testing.T) {
	sc := NewStatsCollector(4, false)

	// Add rows with numeric fields
	values := []float64{10, 20, 30, 40, 50}
	for _, v := range values {
		sc.Add([]float32{1, 0, 0, 0}, metadata.Document{
			"price": metadata.Float(v),
		})
	}

	stats := sc.Finalize()
	require.NotNil(t, stats)
	require.Contains(t, stats.Numeric, "price")

	priceStats := stats.Numeric["price"]
	assert.Equal(t, 10.0, priceStats.Min)
	assert.Equal(t, 50.0, priceStats.Max)
	assert.Equal(t, uint32(5), priceStats.Count)
	assert.InDelta(t, 150.0, priceStats.Sum, 0.001) // 10+20+30+40+50=150

	// Variance = E[X^2] - E[X]^2
	// E[X] = 30, E[X^2] = (100+400+900+1600+2500)/5 = 1100
	// Var = 1100 - 900 = 200
	assert.InDelta(t, 200.0, priceStats.Variance(), 0.1)
}

func TestStatsCollector_NumericWithNaN(t *testing.T) {
	sc := NewStatsCollector(4, false)

	sc.Add([]float32{1, 0, 0, 0}, metadata.Document{
		"value": metadata.Float(10.0),
	})
	sc.Add([]float32{1, 0, 0, 0}, metadata.Document{
		"value": metadata.Float(math.NaN()),
	})
	sc.Add([]float32{1, 0, 0, 0}, metadata.Document{
		"value": metadata.Float(20.0),
	})

	stats := sc.Finalize()
	require.NotNil(t, stats)

	valueStats := stats.Numeric["value"]
	assert.True(t, valueStats.HasNaN)
	assert.Equal(t, 10.0, valueStats.Min)
	assert.Equal(t, 20.0, valueStats.Max)
}

func TestStatsCollector_IntegerFields(t *testing.T) {
	sc := NewStatsCollector(4, false)

	sc.Add([]float32{1, 0, 0, 0}, metadata.Document{
		"count": metadata.Int(5),
	})
	sc.Add([]float32{1, 0, 0, 0}, metadata.Document{
		"count": metadata.Int(15),
	})

	stats := sc.Finalize()
	require.NotNil(t, stats)
	require.Contains(t, stats.Numeric, "count")

	countStats := stats.Numeric["count"]
	assert.Equal(t, 5.0, countStats.Min)
	assert.Equal(t, 15.0, countStats.Max)
}

func TestStatsCollector_CategoricalFields(t *testing.T) {
	sc := NewStatsCollector(4, false)

	// 60% "electronics", 30% "books", 10% "toys"
	categories := []string{
		"electronics", "electronics", "electronics", "electronics", "electronics", "electronics",
		"books", "books", "books",
		"toys",
	}

	for _, cat := range categories {
		sc.Add([]float32{1, 0, 0, 0}, metadata.Document{
			"category": metadata.String(cat),
		})
	}

	stats := sc.Finalize()
	require.NotNil(t, stats)
	require.Contains(t, stats.Categorical, "category")

	catStats := stats.Categorical["category"]
	assert.Equal(t, uint32(3), catStats.DistinctCount)
	assert.Equal(t, "electronics", catStats.DominantValue)
	assert.InDelta(t, 0.6, catStats.DominantRatio, 0.01)

	// Check TopK contains all values
	assert.Len(t, catStats.TopK, 3)

	// Entropy should be non-zero (not perfectly uniform, not pure)
	assert.Greater(t, catStats.Entropy, float32(0))
	assert.Less(t, catStats.Entropy, float32(1))
}

func TestStatsCollector_BooleanFields(t *testing.T) {
	sc := NewStatsCollector(4, false)

	// 70% true, 30% false
	for i := 0; i < 7; i++ {
		sc.Add([]float32{1, 0, 0, 0}, metadata.Document{
			"active": metadata.Bool(true),
		})
	}
	for i := 0; i < 3; i++ {
		sc.Add([]float32{1, 0, 0, 0}, metadata.Document{
			"active": metadata.Bool(false),
		})
	}

	stats := sc.Finalize()
	require.NotNil(t, stats)
	require.Contains(t, stats.Categorical, "active")

	activeStats := stats.Categorical["active"]
	assert.Equal(t, uint32(2), activeStats.DistinctCount)
	assert.Equal(t, "true", activeStats.DominantValue)
	assert.InDelta(t, 0.7, activeStats.DominantRatio, 0.01)
}

func TestStatsCollector_HasFields(t *testing.T) {
	sc := NewStatsCollector(4, false)

	sc.Add([]float32{1, 0, 0, 0}, metadata.Document{
		"field1": metadata.Int(1),
		"field2": metadata.String("value"),
	})
	sc.Add([]float32{1, 0, 0, 0}, metadata.Document{
		"field3": metadata.Float(1.5),
	})

	stats := sc.Finalize()
	require.NotNil(t, stats)

	assert.True(t, stats.HasFields["field1"])
	assert.True(t, stats.HasFields["field2"])
	assert.True(t, stats.HasFields["field3"])
	assert.False(t, stats.HasFields["field4"])
}

func TestStatsCollector_VectorStats(t *testing.T) {
	// Use small dimension to enable vector tracking
	sc := NewStatsCollector(4, true)

	// Add vectors with varying norms
	vectors := [][]float32{
		{1, 0, 0, 0}, // norm = 1
		{2, 0, 0, 0}, // norm = 2
		{3, 0, 0, 0}, // norm = 3
		{0, 4, 0, 0}, // norm = 4
		{0, 0, 5, 0}, // norm = 5
	}

	for _, vec := range vectors {
		sc.Add(vec, nil)
	}

	stats := sc.Finalize()
	require.NotNil(t, stats)
	require.NotNil(t, stats.Vector)

	assert.InDelta(t, float32(1), stats.Vector.MinNorm, 0.01)
	assert.InDelta(t, float32(5), stats.Vector.MaxNorm, 0.01)
	assert.InDelta(t, float32(3), stats.Vector.MeanNorm, 0.01) // (1+2+3+4+5)/5=3

	// Centroid should be computed
	assert.NotEmpty(t, stats.Vector.Centroid)
	assert.Len(t, stats.Vector.Centroid, 4)

	// Distance stats should be computed
	assert.Greater(t, stats.Vector.AvgDistanceToCentroid, float32(0))
	assert.Greater(t, stats.Vector.RadiusMax, float32(0))
	assert.Greater(t, stats.Vector.Radius95, float32(0))
}

func TestStatsCollector_LargeDimNoVectorTracking(t *testing.T) {
	// Dimensions > 256 should not track vectors
	sc := NewStatsCollector(512, true) // Request tracking, but dim > 256

	sc.Add(make([]float32, 512), nil)

	stats := sc.Finalize()
	require.NotNil(t, stats)

	// No vector stats for large dimensions
	assert.Nil(t, stats.Vector)
}

func TestStatsCollector_TimestampSorting(t *testing.T) {
	t.Run("sorted_timestamps", func(t *testing.T) {
		sc := NewStatsCollector(4, false)

		// Add rows with sorted timestamps
		for i := 0; i < 10; i++ {
			sc.Add([]float32{1, 0, 0, 0}, metadata.Document{
				"timestamp": metadata.Float(float64(i * 1000)),
			})
		}

		stats := sc.Finalize()
		require.NotNil(t, stats)
		require.NotNil(t, stats.Shape)

		assert.True(t, stats.Shape.IsSortedByTimestamp)
		assert.Equal(t, "timestamp", stats.Shape.TimestampField)
	})

	t.Run("unsorted_timestamps", func(t *testing.T) {
		sc := NewStatsCollector(4, false)

		// Add rows with unsorted timestamps
		timestamps := []float64{1000, 500, 2000, 100, 3000}
		for _, ts := range timestamps {
			sc.Add([]float32{1, 0, 0, 0}, metadata.Document{
				"timestamp": metadata.Float(ts),
			})
		}

		stats := sc.Finalize()
		require.NotNil(t, stats)
		require.NotNil(t, stats.Shape)

		assert.False(t, stats.Shape.IsSortedByTimestamp)
	})

	t.Run("sorted_ts_field", func(t *testing.T) {
		// Alternative timestamp field names
		sc := NewStatsCollector(4, false)

		for i := 0; i < 5; i++ {
			sc.Add([]float32{1, 0, 0, 0}, metadata.Document{
				"ts": metadata.Int(int64(i)),
			})
		}

		stats := sc.Finalize()
		require.NotNil(t, stats)

		assert.True(t, stats.Shape.IsSortedByTimestamp)
		assert.Equal(t, "ts", stats.Shape.TimestampField)
	})
}

func TestStatsCollector_AppendOnly(t *testing.T) {
	t.Run("append_only_no_deletes", func(t *testing.T) {
		sc := NewStatsCollector(4, false)

		for i := 0; i < 5; i++ {
			sc.Add([]float32{float32(i), 0, 0, 0}, nil)
		}

		stats := sc.Finalize()
		require.NotNil(t, stats)
		require.NotNil(t, stats.Shape)

		assert.True(t, stats.Shape.IsAppendOnly)
	})

	t.Run("not_append_only_with_deletes", func(t *testing.T) {
		sc := NewStatsCollector(4, false)

		for i := 0; i < 5; i++ {
			sc.Add([]float32{float32(i), 0, 0, 0}, nil)
		}
		sc.AddDeleted()

		stats := sc.Finalize()
		require.NotNil(t, stats)
		require.NotNil(t, stats.Shape)

		assert.False(t, stats.Shape.IsAppendOnly)
	})
}

func TestStatsCollector_Histogram(t *testing.T) {
	sc := NewStatsCollector(4, false)

	// Add many values to trigger histogram computation
	for i := 0; i < 1000; i++ {
		val := float64(i) // 0 to 999
		sc.Add([]float32{1, 0, 0, 0}, metadata.Document{
			"value": metadata.Float(val),
		})
	}

	stats := sc.Finalize()
	require.NotNil(t, stats)

	valueStats := stats.Numeric["value"]

	// Check histogram is populated
	var totalInHistogram uint32
	for _, count := range valueStats.Histogram {
		totalInHistogram += count
	}
	assert.Greater(t, totalInHistogram, uint32(0))

	// Check per-bin min/max are populated
	nonEmptyBins := 0
	for i := 0; i < HistogramBins; i++ {
		if valueStats.Histogram[i] > 0 {
			nonEmptyBins++
			// Bin should have valid min/max
			assert.LessOrEqual(t, valueStats.HistogramMin[i], valueStats.HistogramMax[i])
		}
	}
	assert.Greater(t, nonEmptyBins, 0)
}

func TestStatsCollector_CategoricalBloom(t *testing.T) {
	sc := NewStatsCollector(4, false)

	// Add many distinct values to trigger Bloom filter creation
	for i := 0; i < 100; i++ {
		sc.Add([]float32{1, 0, 0, 0}, metadata.Document{
			"category": metadata.String("cat" + string(rune('A'+i%26)) + string(rune('0'+i/26))),
		})
	}

	stats := sc.Finalize()
	require.NotNil(t, stats)

	catStats := stats.Categorical["category"]

	// With > TopKLimit distinct values, Bloom should be created
	// Note: Bloom is not serialized in JSON, so this tests internal state
	assert.Greater(t, catStats.DistinctCount, uint32(TopKLimit))
	assert.Len(t, catStats.TopK, TopKLimit) // TopK is limited
}

func TestStatsCollector_FilterEntropy(t *testing.T) {
	t.Run("low_entropy_pure_segment", func(t *testing.T) {
		sc := NewStatsCollector(4, false)

		// All same category = very low entropy
		for i := 0; i < 100; i++ {
			sc.Add([]float32{1, 0, 0, 0}, metadata.Document{
				"category": metadata.String("only_value"),
			})
		}

		stats := sc.Finalize()
		require.NotNil(t, stats)

		// Single value = 0 entropy
		assert.InDelta(t, 0.0, stats.FilterEntropy, 0.01)
	})

	t.Run("high_entropy_uniform_segment", func(t *testing.T) {
		sc := NewStatsCollector(4, false)

		// Many different categories = high entropy
		for i := 0; i < 100; i++ {
			sc.Add([]float32{1, 0, 0, 0}, metadata.Document{
				"category": metadata.String("cat" + string(rune('A'+i))),
			})
		}

		stats := sc.Finalize()
		require.NotNil(t, stats)

		// Uniform distribution = high entropy (close to 1)
		assert.Greater(t, stats.FilterEntropy, float32(0.9))
	})
}

func TestStatsCollector_ClusterTightness(t *testing.T) {
	sc := NewStatsCollector(4, true)

	// Add tightly clustered vectors (all identical)
	// This creates zero variance, meaning perfectly tight cluster
	for i := 0; i < 100; i++ {
		sc.Add([]float32{1, 1, 1, 1}, nil)
	}

	stats := sc.Finalize()
	require.NotNil(t, stats)
	require.NotNil(t, stats.Vector)
	require.NotNil(t, stats.Shape)

	// ClusterTightness depends on Radius95 vs AvgDist ratio
	// With identical vectors, AvgDistanceToCentroid should be 0
	// which means any comparison needs careful handling
	// Just verify the field is computed without crashing
	assert.GreaterOrEqual(t, stats.Shape.ClusterTightness, float32(0))
	assert.LessOrEqual(t, stats.Shape.ClusterTightness, float32(1))
}

func TestStatsCollector_MixedFieldTypes(t *testing.T) {
	sc := NewStatsCollector(4, false)

	// Add rows with mixed field types
	for i := 0; i < 10; i++ {
		sc.Add([]float32{float32(i), 0, 0, 0}, metadata.Document{
			"int_field":    metadata.Int(int64(i)),
			"float_field":  metadata.Float(float64(i) * 1.5),
			"string_field": metadata.String("value" + string(rune('0'+i))),
			"bool_field":   metadata.Bool(i%2 == 0),
		})
	}

	stats := sc.Finalize()
	require.NotNil(t, stats)

	// All fields should be tracked
	assert.Contains(t, stats.Numeric, "int_field")
	assert.Contains(t, stats.Numeric, "float_field")
	assert.Contains(t, stats.Categorical, "string_field")
	assert.Contains(t, stats.Categorical, "bool_field")
	assert.True(t, stats.HasFields["int_field"])
	assert.True(t, stats.HasFields["float_field"])
	assert.True(t, stats.HasFields["string_field"])
	assert.True(t, stats.HasFields["bool_field"])
}

func TestStatsCollector_LargeDataset(t *testing.T) {
	sc := NewStatsCollector(4, false)

	// Add many rows to test sampling limits
	for i := 0; i < MaxHistogramSamples+1000; i++ {
		sc.Add([]float32{float32(i), 0, 0, 0}, metadata.Document{
			"value": metadata.Float(float64(i)),
		})
	}

	stats := sc.Finalize()
	require.NotNil(t, stats)

	// Should still produce valid stats despite sampling
	assert.Equal(t, uint32(MaxHistogramSamples+1000), stats.TotalRows)
	assert.Contains(t, stats.Numeric, "value")
}
