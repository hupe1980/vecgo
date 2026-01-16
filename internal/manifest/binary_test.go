package manifest

import (
	"bytes"
	"testing"
	"time"

	"github.com/hupe1980/vecgo/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestBinaryRoundTrip(t *testing.T) {
	m := &Manifest{
		Version:       1,
		ID:            1,
		CreatedAt:     time.Now(),
		Dim:           2,
		Metric:        "L2",
		NextSegmentID: 2,
		MaxLSN:        123,
		Segments: []SegmentInfo{
			{
				ID:       1,
				Level:    0,
				RowCount: 10,
				Size:     1024,
				MinID:    1,
				MaxID:    10,
				Path:     "segment_1.bin",
			},
		},
		PKIndex: PKIndexInfo{Path: "pk.idx"},
	}

	var buf bytes.Buffer
	err := m.WriteBinary(&buf)
	require.NoError(t, err)
	t.Logf("Written %d bytes", buf.Len())

	m2, err := ReadBinary(&buf)
	require.NoError(t, err)

	assert.Equal(t, m.ID, m2.ID)
	assert.Equal(t, m.Dim, m2.Dim)
	assert.Equal(t, m.Metric, m2.Metric)
	assert.Equal(t, m.NextSegmentID, m2.NextSegmentID)
	assert.Equal(t, m.MaxLSN, m2.MaxLSN)
	assert.Equal(t, len(m.Segments), len(m2.Segments))

	if len(m2.Segments) > 0 {
		s := m2.Segments[0]
		assert.Equal(t, model.SegmentID(1), s.ID)
		assert.Equal(t, 0, s.Level)
		assert.Equal(t, uint32(10), s.RowCount)
		assert.Equal(t, int64(1024), s.Size)
		assert.Equal(t, model.ID(1), s.MinID)
		assert.Equal(t, model.ID(10), s.MaxID)
		assert.Equal(t, "segment_1.bin", s.Path)
	}
	assert.Equal(t, "pk.idx", m2.PKIndex.Path)
}

func TestBinaryRoundTrip_WithStats(t *testing.T) {
	stats := &SegmentStats{
		Numeric: map[string]NumericFieldStats{
			"price": {Min: 10.5, Max: 99.9, HasNaN: false},
			"age":   {Min: 18, Max: 65, HasNaN: true},
		},
		Categorical: map[string]CategoricalStats{
			"category": {
				DistinctCount: 5,
				TopK: []ValueFreq{
					{Value: "electronics", Count: 100},
					{Value: "books", Count: 50},
				},
			},
		},
		HasFields: map[string]bool{
			"price":    true,
			"category": true,
			"age":      true,
		},
		Vector: &VectorStats{
			MinNorm:  0.5,
			MaxNorm:  2.5,
			MeanNorm: 1.2,
			Centroid: []int8{10, -20, 30, -40},
		},
	}

	m := &Manifest{
		Version:       2,
		ID:            42,
		CreatedAt:     time.Now(),
		Dim:           4,
		Metric:        "L2",
		NextSegmentID: 3,
		MaxLSN:        456,
		Segments: []SegmentInfo{
			{
				ID:       1,
				Level:    0,
				RowCount: 100,
				Size:     8192,
				MinID:    1,
				MaxID:    100,
				Path:     "segment_1.bin",
				Stats:    stats,
			},
			{
				ID:       2,
				Level:    1,
				RowCount: 500,
				Size:     40960,
				MinID:    101,
				MaxID:    600,
				Path:     "segment_2.bin",
				Stats:    nil, // Test nil stats
			},
		},
		PKIndex: PKIndexInfo{Path: "pk.idx"},
	}

	var buf bytes.Buffer
	err := m.WriteBinary(&buf)
	require.NoError(t, err)
	t.Logf("Written %d bytes with stats", buf.Len())

	m2, err := ReadBinary(&buf)
	require.NoError(t, err)

	// Basic manifest fields
	assert.Equal(t, m.ID, m2.ID)
	assert.Equal(t, m.Dim, m2.Dim)
	assert.Equal(t, 2, len(m2.Segments))

	// Segment 1 with stats
	s1 := m2.Segments[0]
	require.NotNil(t, s1.Stats, "segment 1 should have stats")

	// Numeric stats
	require.Contains(t, s1.Stats.Numeric, "price")
	assert.Equal(t, 10.5, s1.Stats.Numeric["price"].Min)
	assert.Equal(t, 99.9, s1.Stats.Numeric["price"].Max)
	assert.False(t, s1.Stats.Numeric["price"].HasNaN)

	require.Contains(t, s1.Stats.Numeric, "age")
	assert.Equal(t, float64(18), s1.Stats.Numeric["age"].Min)
	assert.Equal(t, float64(65), s1.Stats.Numeric["age"].Max)
	assert.True(t, s1.Stats.Numeric["age"].HasNaN)

	// Categorical stats
	require.Contains(t, s1.Stats.Categorical, "category")
	assert.Equal(t, uint32(5), s1.Stats.Categorical["category"].DistinctCount)
	assert.Equal(t, 2, len(s1.Stats.Categorical["category"].TopK))
	assert.Equal(t, "electronics", s1.Stats.Categorical["category"].TopK[0].Value)
	assert.Equal(t, uint32(100), s1.Stats.Categorical["category"].TopK[0].Count)

	// HasFields
	assert.True(t, s1.Stats.HasFields["price"])
	assert.True(t, s1.Stats.HasFields["category"])
	assert.True(t, s1.Stats.HasFields["age"])

	// Vector stats
	require.NotNil(t, s1.Stats.Vector)
	assert.Equal(t, float32(0.5), s1.Stats.Vector.MinNorm)
	assert.Equal(t, float32(2.5), s1.Stats.Vector.MaxNorm)
	assert.Equal(t, float32(1.2), s1.Stats.Vector.MeanNorm)
	assert.Equal(t, []int8{10, -20, 30, -40}, s1.Stats.Vector.Centroid)

	// Segment 2 without stats
	s2 := m2.Segments[1]
	assert.Nil(t, s2.Stats, "segment 2 should have nil stats")
}
