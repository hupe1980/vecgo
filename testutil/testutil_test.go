package testutil

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestUniformVectors(t *testing.T) {
	rng := NewRNG(4711)

	v := rng.UniformVectors(8, 32)

	assert.Equal(t, 8, len(v))
	assert.Equal(t, 32, len(v[0]))
	assert.LessOrEqual(t, v[0][0], float32(1.0))
	assert.GreaterOrEqual(t, v[1][0], float32(0.0))
}

func TestUniformRangeVectors(t *testing.T) {
	rng := NewRNG(4711)

	v := rng.UniformRangeVectors(8, 32)

	assert.Equal(t, 8, len(v))
	assert.Equal(t, 32, len(v[0]))
	assert.LessOrEqual(t, v[0][0], float32(1.0))
	assert.GreaterOrEqual(t, v[1][0], float32(-1.0))
}

func TestUnitVectors(t *testing.T) {
	rng := NewRNG(4711)

	v := rng.UnitVectors(8, 32)

	assert.Equal(t, 8, len(v))
	assert.Equal(t, 32, len(v[0]))

	// Check normalization
	for _, vec := range v {
		var sum float32
		for _, val := range vec {
			sum += val * val
		}
		assert.InDelta(t, float32(1.0), sum, 1e-5)
	}
}

func TestClusteredVectors(t *testing.T) {
	rng := NewRNG(4711)

	v := rng.ClusteredVectors(100, 32, 5, 0.1)

	assert.Equal(t, 100, len(v))
	assert.Equal(t, 32, len(v[0]))
}

func TestReset(t *testing.T) {
	rng := NewRNG(4711)
	v1 := rng.UniformVectors(1, 10)

	rng.Reset()
	v2 := rng.UniformVectors(1, 10)

	assert.Equal(t, v1, v2)
}

// ============================================================================
// Adversarial Distribution Tests
// ============================================================================

func TestSegmentLocalSkewBuckets(t *testing.T) {
	rng := NewRNG(42)
	numVecs := 10000
	bucketCount := 100
	numSegments := 10
	localDominance := 0.90

	buckets := rng.SegmentLocalSkewBuckets(numVecs, bucketCount, numSegments, localDominance)

	assert.Equal(t, numVecs, len(buckets))

	// Check that each segment has a dominant bucket
	segmentSize := numVecs / numSegments
	for seg := 0; seg < numSegments; seg++ {
		start := seg * segmentSize
		end := start + segmentSize
		if end > numVecs {
			end = numVecs
		}

		// Count bucket frequencies in this segment
		counts := make(map[int64]int)
		for i := start; i < end; i++ {
			counts[buckets[i]]++
		}

		// Find the most frequent bucket
		var maxCount int
		for _, c := range counts {
			if c > maxCount {
				maxCount = c
			}
		}

		// The dominant bucket should have > 50% (likely ~90%)
		segLen := end - start
		dominantRatio := float64(maxCount) / float64(segLen)
		assert.Greater(t, dominantRatio, 0.5, "segment %d should have dominant bucket", seg)
	}
}

func TestBooleanAdversarialBuckets(t *testing.T) {
	rng := NewRNG(42)
	numVecs := 10000
	bucketCount := 100

	bucketA, bucketB := rng.BooleanAdversarialBuckets(numVecs, bucketCount)

	assert.Equal(t, numVecs, len(bucketA))
	assert.Equal(t, numVecs, len(bucketB))

	// Check bimodal distribution in bucketA
	selectiveBuckets := bucketCount / 10
	selectiveCount := 0
	for _, b := range bucketA {
		if b < int64(selectiveBuckets) {
			selectiveCount++
		}
	}

	// ~10% should be in selective buckets
	selectiveRatio := float64(selectiveCount) / float64(numVecs)
	assert.InDelta(t, 0.10, selectiveRatio, 0.05, "selective buckets should be ~10%")

	// Check correlation between A and B
	sameCount := 0
	for i := range numVecs {
		if bucketA[i] == bucketB[i] {
			sameCount++
		}
	}

	// ~50% should be correlated
	correlationRatio := float64(sameCount) / float64(numVecs)
	assert.InDelta(t, 0.50, correlationRatio, 0.10, "~50% correlation expected")
}
