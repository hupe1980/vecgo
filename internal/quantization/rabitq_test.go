package quantization

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"
)

func TestRaBitQ_EncodeDecode(t *testing.T) {
	dim := 128
	rq := NewRaBitQuantizer(dim)

	// Create random vector
	vec := make([]float32, dim)
	var sumSq float32
	for i := 0; i < dim; i++ {
		vec[i] = rand.Float32()*2 - 1
		sumSq += vec[i] * vec[i]
	}
	norm := float32(math.Sqrt(float64(sumSq)))

	// Encode
	encoded, err := rq.Encode(vec)
	if err != nil {
		t.Fatalf("Encode failed: %v", err)
	}

	// Verify length
	expectedLen := ((dim + 63) / 64 * 8) + 4
	if len(encoded) != expectedLen {
		t.Errorf("Expected length %d, got %d", expectedLen, len(encoded))
	}

	// Verify norm stored correctly
	storedBits := binary.LittleEndian.Uint32(encoded[len(encoded)-4:])
	storedNorm := math.Float32frombits(storedBits)
	if math.Abs(float64(storedNorm-norm)) > 1e-5 {
		t.Errorf("Stored norm mismatch: got %f, want %f", storedNorm, norm)
	}

	// Decode
	decoded, err := rq.Decode(encoded)
	if err != nil {
		t.Fatalf("Decode failed: %v", err)
	}

	// Verify decoded vector norm
	var decSumSq float32
	for _, v := range decoded {
		decSumSq += v * v
	}
	decNorm := float32(math.Sqrt(float64(decSumSq)))

	if math.Abs(float64(decNorm-norm)) > 1e-4 {
		t.Errorf("Decoded norm mismatch: got %f, want %f", decNorm, norm)
	}
}

func TestRaBitQ_Distance(t *testing.T) {
	dim := 128
	rq := NewRaBitQuantizer(dim)

	vec1 := make([]float32, dim)
	vec2 := make([]float32, dim)

	// Seed for determinism
	rnd := rand.New(rand.NewSource(42))

	for i := 0; i < dim; i++ {
		vec1[i] = rnd.Float32()*2 - 1
		vec2[i] = rnd.Float32()*2 - 1
	}

	encoded2, _ := rq.Encode(vec2)

	// Approximate distance
	distApprox, err := rq.Distance(vec1, encoded2)
	if err != nil {
		t.Fatalf("Distance failed: %v", err)
	}

	// Real L2 distance squared
	var distReal float32
	for i := 0; i < dim; i++ {
		diff := vec1[i] - vec2[i]
		distReal += diff * diff
	}

	t.Logf("Real L2^2: %f, Approx RaBitQ: %f", distReal, distApprox)

	// RaBitQ is approximate, so we don't expect exact match.
	// But it should be somewhat correlated.
	if distApprox < 0 {
		t.Errorf("Distance is negative: %f", distApprox)
	}
}
