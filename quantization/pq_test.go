package quantization

import (
	"math"
	"math/rand"
	"testing"
)

func TestProductQuantizer(t *testing.T) {
	const (
		dimension     = 128
		numVectors    = 1000
		numSubvectors = 8
		numCentroids  = 256
	)

	// Create quantizer
	pq, err := NewProductQuantizer(dimension, numSubvectors, numCentroids)
	if err != nil {
		t.Fatalf("Failed to create PQ: %v", err)
	}

	// Generate random training vectors
	trainingVectors := make([][]float32, numVectors)
	for i := range trainingVectors {
		trainingVectors[i] = generateRandomVector(dimension)
	}

	// Train
	if err := pq.Train(trainingVectors); err != nil {
		t.Fatalf("Training failed: %v", err)
	}

	if !pq.IsTrained() {
		t.Error("Quantizer should be trained")
	}

	// Test encode/decode
	testVec := generateRandomVector(dimension)
	codes := pq.Encode(testVec)

	if len(codes) != numSubvectors {
		t.Errorf("Expected %d codes, got %d", numSubvectors, len(codes))
	}

	reconstructed := pq.Decode(codes)

	if len(reconstructed) != dimension {
		t.Errorf("Expected %d dimensions, got %d", dimension, len(reconstructed))
	}

	// Compute reconstruction error
	var mse float32
	for i := range testVec {
		diff := testVec[i] - reconstructed[i]
		mse += diff * diff
	}
	mse /= float32(dimension)

	t.Logf("Reconstruction MSE: %f", mse)

	// MSE should be reasonable (< 0.1 for normalized vectors)
	if mse > 0.5 {
		t.Errorf("MSE too high: %f", mse)
	}

	// Test compression ratio
	ratio := pq.CompressionRatio()
	expectedRatio := float64(dimension*4) / float64(numSubvectors)
	if math.Abs(ratio-expectedRatio) > 0.01 {
		t.Errorf("Expected compression ratio %.2f, got %.2f", expectedRatio, ratio)
	}

	t.Logf("Compression ratio: %.1fx (%.0f bytes → %d bytes)",
		ratio, float64(dimension*4), pq.BytesPerVector())
}

func TestProductQuantizerAsymmetricDistance(t *testing.T) {
	const (
		dimension     = 64
		numVectors    = 500
		numSubvectors = 8
		numCentroids  = 256
	)

	pq, _ := NewProductQuantizer(dimension, numSubvectors, numCentroids)

	trainingVectors := make([][]float32, numVectors)
	for i := range trainingVectors {
		trainingVectors[i] = generateRandomVector(dimension)
	}

	pq.Train(trainingVectors)

	// Test asymmetric distance vs decoded distance
	query := generateRandomVector(dimension)
	testVec := generateRandomVector(dimension)

	codes := pq.Encode(testVec)

	// Asymmetric distance (fast)
	adcDist := pq.ComputeAsymmetricDistance(query, codes)

	// Full distance (slow - requires decoding)
	decoded := pq.Decode(codes)
	fullDist := squaredL2(query, decoded)

	// They should be equal (same computation)
	if math.Abs(float64(adcDist-fullDist)) > 0.001 {
		t.Errorf("ADC distance mismatch: adc=%f, full=%f", adcDist, fullDist)
	}

	t.Logf("ADC distance: %f, Full distance: %f", adcDist, fullDist)
}

func TestProductQuantizerInvalidDimension(t *testing.T) {
	// Dimension not divisible by numSubvectors
	_, err := NewProductQuantizer(100, 7, 256)
	if err == nil {
		t.Error("Expected error for invalid dimension")
	}

	// Too many centroids
	_, err = NewProductQuantizer(128, 8, 300)
	if err == nil {
		t.Error("Expected error for too many centroids")
	}
}

func BenchmarkProductQuantizerEncode(b *testing.B) {
	const (
		dimension     = 128
		numSubvectors = 8
		numCentroids  = 256
	)

	pq, _ := NewProductQuantizer(dimension, numSubvectors, numCentroids)

	// Train with sample data
	trainingVectors := make([][]float32, 1000)
	for i := range trainingVectors {
		trainingVectors[i] = generateRandomVector(dimension)
	}
	pq.Train(trainingVectors)

	testVec := generateRandomVector(dimension)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = pq.Encode(testVec)
	}
}

func BenchmarkProductQuantizerAsymmetricDistance(b *testing.B) {
	const (
		dimension     = 128
		numSubvectors = 8
		numCentroids  = 256
	)

	pq, _ := NewProductQuantizer(dimension, numSubvectors, numCentroids)

	trainingVectors := make([][]float32, 1000)
	for i := range trainingVectors {
		trainingVectors[i] = generateRandomVector(dimension)
	}
	pq.Train(trainingVectors)

	query := generateRandomVector(dimension)
	codes := pq.Encode(generateRandomVector(dimension))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = pq.ComputeAsymmetricDistance(query, codes)
	}
}

// Helper functions

func generateRandomVector(dim int) []float32 {
	vec := make([]float32, dim)
	for i := range vec {
		vec[i] = rand.Float32()*2 - 1 // Range: [-1, 1]
	}
	// Normalize
	var norm float32
	for _, v := range vec {
		norm += v * v
	}
	norm = float32(math.Sqrt(float64(norm)))
	if norm > 0 {
		for i := range vec {
			vec[i] /= norm
		}
	}
	return vec
}

func squaredL2(a, b []float32) float32 {
	var sum float32
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return sum
}
