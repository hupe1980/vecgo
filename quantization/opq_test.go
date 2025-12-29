package quantization

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewOptimizedProductQuantizer(t *testing.T) {
	opq, err := NewOptimizedProductQuantizer(128, 8, 256, 10)
	require.NoError(t, err)
	require.NotNil(t, opq)
	assert.False(t, opq.IsTrained())

	// Test invalid dimension
	_, err = NewOptimizedProductQuantizer(127, 8, 256, 10)
	assert.Error(t, err)

	// Test too many centroids
	_, err = NewOptimizedProductQuantizer(128, 8, 300, 10)
	assert.Error(t, err)
}

func TestOPQ_TrainEncodeDecode(t *testing.T) {
	dim := 32
	numVectors := 1000

	// Generate training data
	vectors := generateRandomVectors(numVectors, dim)

	// Create and train OPQ
	opq, err := NewOptimizedProductQuantizer(dim, 8, 256, 10)
	require.NoError(t, err)

	err = opq.Train(vectors)
	require.NoError(t, err)
	assert.True(t, opq.IsTrained())

	// Test encode/decode
	testVec := vectors[0]
	codes := opq.Encode(testVec)
	assert.Equal(t, 8, len(codes)) // 8 subvectors = 8 bytes

	reconstructed := opq.Decode(codes)
	assert.Equal(t, dim, len(reconstructed))

	// Verify compression ratio
	ratio := opq.CompressionRatio()
	assert.InDelta(t, 16.0, ratio, 0.1) // 32*4 / 8 = 16x
}

func TestOPQ_RotationOrthogonality(t *testing.T) {
	dim := 32
	numVectors := 500

	vectors := generateRandomVectors(numVectors, dim)

	opq, err := NewOptimizedProductQuantizer(dim, 8, 256, 5)
	require.NoError(t, err)

	err = opq.Train(vectors)
	require.NoError(t, err)

	// Verify rotation matrix is orthogonal: R * R^T = I
	product := make([][]float32, dim)
	for i := range product {
		product[i] = make([]float32, dim)
	}

	// Compute R * R^T
	for i := 0; i < dim; i++ {
		for j := 0; j < dim; j++ {
			sum := float32(0)
			for k := 0; k < dim; k++ {
				sum += opq.rotation[i][k] * opq.rotation[j][k]
			}
			product[i][j] = sum
		}
	}

	// Check if result is approximately identity matrix
	for i := 0; i < dim; i++ {
		for j := 0; j < dim; j++ {
			expected := float32(0)
			if i == j {
				expected = 1
			}
			assert.InDelta(t, expected, product[i][j], 0.1, "R*R^T should be identity at [%d,%d]", i, j)
		}
	}
}

func TestOPQ_AsymmetricDistance(t *testing.T) {
	dim := 64
	numVectors := 500

	vectors := generateRandomVectors(numVectors, dim)

	opq, err := NewOptimizedProductQuantizer(dim, 8, 256, 10)
	require.NoError(t, err)

	err = opq.Train(vectors)
	require.NoError(t, err)

	// Test asymmetric distance computation
	query := vectors[0]
	target := vectors[1]

	codes := opq.Encode(target)
	asymDist := opq.ComputeAsymmetricDistance(query, codes)

	// Asymmetric distance should be positive
	assert.Greater(t, asymDist, float32(0))

	// Distance to itself should be small (but not zero due to quantization error)
	selfCodes := opq.Encode(query)
	selfDist := opq.ComputeAsymmetricDistance(query, selfCodes)
	assert.Less(t, selfDist, asymDist) // Distance to self should be less than to other vector
}

func TestOPQ_ReconstructionQuality(t *testing.T) {
	dim := 64
	numVectors := 1000

	vectors := generateRandomVectors(numVectors, dim)

	// Train OPQ
	opq, err := NewOptimizedProductQuantizer(dim, 8, 256, 15)
	require.NoError(t, err)
	err = opq.Train(vectors)
	require.NoError(t, err)

	// Train standard PQ for comparison
	pq, err := NewProductQuantizer(dim, 8, 256)
	require.NoError(t, err)
	err = pq.Train(vectors)
	require.NoError(t, err)

	// Measure reconstruction error for both
	opqError := float32(0)
	pqError := float32(0)

	testVectors := vectors[:100] // Test on subset
	for _, vec := range testVectors {
		// OPQ reconstruction
		opqCodes := opq.Encode(vec)
		opqRecon := opq.Decode(opqCodes)
		opqError += l2DistanceSquared(vec, opqRecon)

		// PQ reconstruction
		pqCodes := pq.Encode(vec)
		pqRecon := pq.Decode(pqCodes)
		pqError += l2DistanceSquared(vec, pqRecon)
	}

	opqError /= float32(len(testVectors))
	pqError /= float32(len(testVectors))

	// OPQ should have lower or comparable reconstruction error
	// Note: Due to randomness, this isn't always guaranteed, but should trend toward better
	t.Logf("OPQ reconstruction error: %.4f", opqError)
	t.Logf("PQ reconstruction error: %.4f", pqError)

	// We expect OPQ to be better or at least not much worse (within 20%)
	assert.LessOrEqual(t, opqError, pqError*1.2, "OPQ should have comparable or better reconstruction quality")
}

func TestOPQ_BytesPerVector(t *testing.T) {
	opq, err := NewOptimizedProductQuantizer(128, 8, 256, 10)
	require.NoError(t, err)

	assert.Equal(t, 8, opq.BytesPerVector())
}

func TestOPQ_NotTrainedPanic(t *testing.T) {
	opq, err := NewOptimizedProductQuantizer(32, 4, 256, 10)
	require.NoError(t, err)

	vec := make([]float32, 32)

	// Should panic when not trained
	assert.Panics(t, func() {
		opq.Encode(vec)
	})

	assert.Panics(t, func() {
		codes := make([]byte, 4)
		opq.Decode(codes)
	})

	assert.Panics(t, func() {
		codes := make([]byte, 4)
		opq.ComputeAsymmetricDistance(vec, codes)
	})
}

func TestOPQ_EmptyTrainingData(t *testing.T) {
	opq, err := NewOptimizedProductQuantizer(32, 4, 256, 10)
	require.NoError(t, err)

	err = opq.Train([][]float32{})
	assert.Error(t, err)
}

func TestOPQ_DimensionMismatch(t *testing.T) {
	opq, err := NewOptimizedProductQuantizer(32, 4, 256, 10)
	require.NoError(t, err)

	wrongDimVectors := generateRandomVectors(100, 64)
	err = opq.Train(wrongDimVectors)
	assert.Error(t, err)
}

func BenchmarkOPQ_Train(b *testing.B) {
	dim := 128
	numVectors := 10000

	vectors := generateRandomVectors(numVectors, dim)

	b.ResetTimer()
	for b.Loop() {
		opq, _ := NewOptimizedProductQuantizer(dim, 8, 256, 10)
		_ = opq.Train(vectors)
	}
}

func BenchmarkOPQ_Encode(b *testing.B) {
	dim := 128
	numVectors := 1000

	vectors := generateRandomVectors(numVectors, dim)

	opq, _ := NewOptimizedProductQuantizer(dim, 8, 256, 10)
	_ = opq.Train(vectors)

	testVec := vectors[0]

	b.ResetTimer()
	for b.Loop() {
		_ = opq.Encode(testVec)
	}
}

func BenchmarkOPQ_Decode(b *testing.B) {
	dim := 128
	numVectors := 1000

	vectors := generateRandomVectors(numVectors, dim)

	opq, _ := NewOptimizedProductQuantizer(dim, 8, 256, 10)
	_ = opq.Train(vectors)

	codes := opq.Encode(vectors[0])

	b.ResetTimer()
	for b.Loop() {
		_ = opq.Decode(codes)
	}
}

func BenchmarkOPQ_AsymmetricDistance(b *testing.B) {
	dim := 128
	numVectors := 1000

	vectors := generateRandomVectors(numVectors, dim)

	opq, _ := NewOptimizedProductQuantizer(dim, 8, 256, 10)
	_ = opq.Train(vectors)

	query := vectors[0]
	codes := opq.Encode(vectors[1])

	b.ResetTimer()
	for b.Loop() {
		_ = opq.ComputeAsymmetricDistance(query, codes)
	}
}

func BenchmarkOPQ_vs_PQ_ReconstructionQuality(b *testing.B) {
	dim := 128
	numVectors := 5000

	vectors := generateRandomVectors(numVectors, dim)

	b.Run("OPQ", func(b *testing.B) {
		opq, _ := NewOptimizedProductQuantizer(dim, 8, 256, 10)
		_ = opq.Train(vectors)

		testVectors := vectors[:100]
		totalError := float32(0)

		b.ResetTimer()
		for b.Loop() {
			for _, vec := range testVectors {
				codes := opq.Encode(vec)
				recon := opq.Decode(codes)
				totalError += l2DistanceSquared(vec, recon)
			}
		}
		b.ReportMetric(float64(totalError)/float64(b.N*len(testVectors)), "avg_error")
	})

	b.Run("PQ", func(b *testing.B) {
		pq, _ := NewProductQuantizer(dim, 8, 256)
		_ = pq.Train(vectors)

		testVectors := vectors[:100]
		totalError := float32(0)

		b.ResetTimer()
		for b.Loop() {
			for _, vec := range testVectors {
				codes := pq.Encode(vec)
				recon := pq.Decode(codes)
				totalError += l2DistanceSquared(vec, recon)
			}
		}
		b.ReportMetric(float64(totalError)/float64(b.N*len(testVectors)), "avg_error")
	})
}

// Helper function to compute squared L2 distance
func l2DistanceSquared(a, b []float32) float32 {
	sum := float32(0)
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return sum
}

// Helper function to generate random vectors
func generateRandomVectors(n, dim int) [][]float32 {
	vectors := make([][]float32, n)
	for i := range vectors {
		vectors[i] = make([]float32, dim)
		for j := range vectors[i] {
			vectors[i][j] = float32(math.Sin(float64(i*dim + j)))
		}
	}
	return vectors
}
