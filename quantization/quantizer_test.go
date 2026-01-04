package quantization

import (
	"math"
	"testing"
)

func TestScalarQuantizer_Train(t *testing.T) {
	vectors := [][]float32{
		{-1.0, 0.0, 1.0},
		{-0.5, 0.5, 2.0},
		{-2.0, 1.0, 3.0},
	}

	sq := NewScalarQuantizer(3)
	err := sq.Train(vectors)
	if err != nil {
		t.Fatalf("Train failed: %v", err)
	}

	// Check per-dimension min/max
	// Dim 0: -1.0, -0.5, -2.0 -> min=-2.0, max=-0.5
	if sq.Min(0) != -2.0 {
		t.Errorf("Dim 0: Expected min=-2.0, got %f", sq.Min(0))
	}
	if sq.Max(0) != -0.5 {
		t.Errorf("Dim 0: Expected max=-0.5, got %f", sq.Max(0))
	}

	// Dim 2: 1.0, 2.0, 3.0 -> min=1.0, max=3.0
	if sq.Min(2) != 1.0 {
		t.Errorf("Dim 2: Expected min=1.0, got %f", sq.Min(2))
	}
	if sq.Max(2) != 3.0 {
		t.Errorf("Dim 2: Expected max=3.0, got %f", sq.Max(2))
	}
}

func TestScalarQuantizer_EncodeDecode(t *testing.T) {
	// Manually setup a trained quantizer for testing
	sq := NewScalarQuantizer(5)
	sq.trained = true
	sq.mins = []float32{-1.0, -1.0, -1.0, -1.0, -1.0}
	sq.maxs = []float32{1.0, 1.0, 1.0, 1.0, 1.0}
	sq.scales = make([]float32, 5)
	sq.invScales = make([]float32, 5)
	for i := 0; i < 5; i++ {
		sq.scales[i] = 255.0 / 2.0
		sq.invScales[i] = 2.0 / 255.0
	}

	original := []float32{-1.0, -0.5, 0.0, 0.5, 1.0}

	// Encode
	quantized := sq.Encode(original)
	if len(quantized) != len(original) {
		t.Fatalf("Expected %d bytes, got %d", len(original), len(quantized))
	}

	// Decode
	decoded := sq.Decode(quantized)
	if len(decoded) != len(original) {
		t.Fatalf("Expected %d floats, got %d", len(original), len(decoded))
	}

	// Check reconstruction error
	maxError := float32(0.0)
	for i := range original {
		err := float32(math.Abs(float64(original[i] - decoded[i])))
		if err > maxError {
			maxError = err
		}
	}

	// Error should be small (within one quantization step)
	expectedMaxError := (sq.Max(0) - sq.Min(0)) / 255.0
	if maxError > expectedMaxError*1.1 { // Allow 10% tolerance
		t.Errorf("Reconstruction error too large: %f (expected <= %f)", maxError, expectedMaxError)
	}
}

func TestScalarQuantizer_CompressionRatio(t *testing.T) {
	sq := NewScalarQuantizer(128)
	ratio := sq.CompressionRatio()

	if ratio != 4.0 {
		t.Errorf("Expected compression ratio 4.0, got %f", ratio)
	}
}

func TestScalarQuantizer_BytesPerDimension(t *testing.T) {
	sq := NewScalarQuantizer(128)
	if sq.BytesPerDimension() != 1 {
		t.Errorf("Expected 1 byte per dimension, got %d", sq.BytesPerDimension())
	}
}

func TestScalarQuantizer_EmptyVectors(t *testing.T) {
	sq := NewScalarQuantizer(128)
	err := sq.Train([][]float32{})

	if err == nil {
		t.Error("Expected error for empty vectors")
	}
}

func TestScalarQuantizer_UniformValues(t *testing.T) {
	vectors := [][]float32{
		{5.0, 5.0, 5.0},
		{5.0, 5.0, 5.0},
	}

	sq := NewScalarQuantizer(3)
	err := sq.Train(vectors)
	if err != nil {
		t.Fatalf("Train failed: %v", err)
	}

	// Should handle uniform values by adding a small range
	if sq.Max(0) <= sq.Min(0) {
		t.Error("Max should be greater than min even for uniform values")
	}

	// Encode/decode should still work
	original := []float32{5.0, 5.0, 5.0}
	quantized := sq.Encode(original)
	decoded := sq.Decode(quantized)

	for i := range decoded {
		if math.Abs(float64(decoded[i]-5.0)) > 0.01 {
			t.Errorf("Expected decoded value ~5.0, got %f", decoded[i])
		}
	}
}

func TestScalarQuantizer_Clamping(t *testing.T) {
	sq := NewScalarQuantizer(3)
	sq.trained = true
	sq.mins = []float32{0.0, 0.0, 0.0}
	sq.maxs = []float32{1.0, 1.0, 1.0}
	sq.scales = []float32{255.0, 255.0, 255.0}
	sq.invScales = []float32{1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0}

	// Test values outside trained range
	original := []float32{-1.0, 0.5, 2.0}
	quantized := sq.Encode(original)
	decoded := sq.Decode(quantized)

	// Values should be clamped to [0, 1]
	if decoded[0] < -0.01 { // Small tolerance
		t.Errorf("Expected clamped value >= 0, got %f", decoded[0])
	}
	if decoded[2] > 1.01 {
		t.Errorf("Expected clamped value <= 1, got %f", decoded[2])
	}
}

func BenchmarkScalarQuantizer_Encode(b *testing.B) {
	dim := 128
	sq := NewScalarQuantizer(dim)
	sq.trained = true
	sq.mins = make([]float32, dim)
	sq.maxs = make([]float32, dim)
	sq.scales = make([]float32, dim)
	sq.invScales = make([]float32, dim)
	for i := 0; i < dim; i++ {
		sq.mins[i] = -1.0
		sq.maxs[i] = 1.0
		sq.scales[i] = 255.0 / 2.0
		sq.invScales[i] = 2.0 / 255.0
	}

	vec := make([]float32, dim)
	for i := range vec {
		vec[i] = float32(i%256)/128.0 - 1.0
	}

	b.ResetTimer()
	for b.Loop() {
		_ = sq.Encode(vec)
	}
}

func BenchmarkScalarQuantizer_Decode(b *testing.B) {
	dim := 128
	sq := NewScalarQuantizer(dim)
	sq.trained = true
	sq.mins = make([]float32, dim)
	sq.maxs = make([]float32, dim)
	sq.scales = make([]float32, dim)
	sq.invScales = make([]float32, dim)
	for i := 0; i < dim; i++ {
		sq.mins[i] = -1.0
		sq.maxs[i] = 1.0
		sq.scales[i] = 255.0 / 2.0
		sq.invScales[i] = 2.0 / 255.0
	}

	vec := make([]float32, dim)
	for i := range vec {
		vec[i] = float32(i%256)/128.0 - 1.0
	}

	quantized := sq.Encode(vec)

	b.ResetTimer()
	for b.Loop() {
		_ = sq.Decode(quantized)
	}
}

func TestScalarQuantizer_L2DistanceBatch(t *testing.T) {
	dim := 4
	sq := NewScalarQuantizer(dim)
	sq.trained = true
	sq.mins = []float32{0, 0, 0, 0}
	sq.maxs = []float32{10, 10, 10, 10}
	sq.scales = make([]float32, dim)
	sq.invScales = make([]float32, dim)
	for i := 0; i < dim; i++ {
		sq.scales[i] = 255.0 / 10.0
		sq.invScales[i] = 10.0 / 255.0
	}

	query := []float32{1, 2, 3, 4}

	// Create 2 vectors
	v1 := []float32{1, 2, 3, 4} // Distance 0
	v2 := []float32{2, 3, 4, 5} // Distance 1+1+1+1 = 4

	code1 := sq.Encode(v1)
	code2 := sq.Encode(v2)

	codes := make([]byte, 0, dim*2)
	codes = append(codes, code1...)
	codes = append(codes, code2...)

	out := make([]float32, 2)
	sq.L2DistanceBatch(query, codes, 2, out)

	// Check v1 distance (should be close to 0)
	if out[0] > 0.1 {
		t.Errorf("Expected distance ~0, got %f", out[0])
	}

	// Check v2 distance (should be close to 4)
	if math.Abs(float64(out[1]-4.0)) > 0.2 {
		t.Errorf("Expected distance ~4, got %f", out[1])
	}
}
