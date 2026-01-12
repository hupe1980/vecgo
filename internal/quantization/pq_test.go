package quantization

import (
	"math"
	"testing"

	"github.com/hupe1980/vecgo/testutil"
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

	rng := testutil.NewRNG(0)

	// Generate random training vectors
	trainingVectors := rng.UnitVectors(numVectors, dimension)

	// Train
	if err := pq.Train(trainingVectors); err != nil {
		t.Fatalf("Training failed: %v", err)
	}

	if !pq.IsTrained() {
		t.Error("Quantizer should be trained")
	}

	// Test encode/decode
	testVec := rng.UnitVectors(1, dimension)[0]
	codes, err := pq.Encode(testVec)
	if err != nil {
		t.Fatalf("Encode failed: %v", err)
	}

	if len(codes) != numSubvectors {
		t.Errorf("Expected %d codes, got %d", numSubvectors, len(codes))
	}

	reconstructed, err := pq.Decode(codes)
	if err != nil {
		t.Fatalf("Decode failed: %v", err)
	}

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

	rng := testutil.NewRNG(0)

	trainingVectors := rng.UnitVectors(numVectors, dimension)

	pq.Train(trainingVectors)

	// Test asymmetric distance vs decoded distance
	query := rng.UnitVectors(1, dimension)[0]
	testVec := rng.UnitVectors(1, dimension)[0]

	codes, err := pq.Encode(testVec)
	if err != nil {
		t.Fatalf("Encode failed: %v", err)
	}

	// Asymmetric distance (fast)
	adcDist, err := pq.ComputeAsymmetricDistance(query, codes)
	if err != nil {
		t.Fatalf("ComputeAsymmetricDistance failed: %v", err)
	}

	// Full distance (slow - requires decoding)
	decoded, err := pq.Decode(codes)
	if err != nil {
		t.Fatalf("Decode failed: %v", err)
	}
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

	rng := testutil.NewRNG(0)

	// Train with sample data
	trainingVectors := rng.UnitVectors(1000, dimension)
	pq.Train(trainingVectors)

	testVec := rng.UnitVectors(1, dimension)[0]

	b.ResetTimer()
	for b.Loop() {
		_, _ = pq.Encode(testVec)
	}
}

func BenchmarkProductQuantizerAsymmetricDistance(b *testing.B) {
	const (
		dimension     = 128
		numSubvectors = 8
		numCentroids  = 256
	)

	pq, _ := NewProductQuantizer(dimension, numSubvectors, numCentroids)

	rng := testutil.NewRNG(0)

	trainingVectors := rng.UnitVectors(1000, dimension)
	pq.Train(trainingVectors)

	query := rng.UnitVectors(1, dimension)[0]
	codes, _ := pq.Encode(rng.UnitVectors(1, dimension)[0])

	b.ResetTimer()
	for b.Loop() {
		_, _ = pq.ComputeAsymmetricDistance(query, codes)
	}
}

func squaredL2(a, b []float32) float32 {
	var sum float32
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return sum
}
