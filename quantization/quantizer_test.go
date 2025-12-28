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

	sq := NewScalarQuantizer()
	err := sq.Train(vectors)
	if err != nil {
		t.Fatalf("Train failed: %v", err)
	}

	if sq.min != -2.0 {
		t.Errorf("Expected min=-2.0, got %f", sq.min)
	}
	if sq.max != 3.0 {
		t.Errorf("Expected max=3.0, got %f", sq.max)
	}
}

func TestScalarQuantizer_EncodeDecode(t *testing.T) {
	sq := NewScalarQuantizer()
	sq.min = -1.0
	sq.max = 1.0

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
	expectedMaxError := (sq.max - sq.min) / 255.0
	if maxError > expectedMaxError*1.1 { // Allow 10% tolerance
		t.Errorf("Reconstruction error too large: %f (expected <= %f)", maxError, expectedMaxError)
	}
}

func TestScalarQuantizer_CompressionRatio(t *testing.T) {
	sq := NewScalarQuantizer()
	ratio := sq.CompressionRatio()

	if ratio != 4.0 {
		t.Errorf("Expected compression ratio 4.0, got %f", ratio)
	}
}

func TestScalarQuantizer_BytesPerDimension(t *testing.T) {
	sq := NewScalarQuantizer()
	if sq.BytesPerDimension() != 1 {
		t.Errorf("Expected 1 byte per dimension, got %d", sq.BytesPerDimension())
	}
}

func TestScalarQuantizer_EmptyVectors(t *testing.T) {
	sq := NewScalarQuantizer()
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

	sq := NewScalarQuantizer()
	err := sq.Train(vectors)
	if err != nil {
		t.Fatalf("Train failed: %v", err)
	}

	// Should handle uniform values by adding a small range
	if sq.max <= sq.min {
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
	sq := NewScalarQuantizer()
	sq.min = 0.0
	sq.max = 1.0

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
	sq := NewScalarQuantizer()
	sq.min = -1.0
	sq.max = 1.0

	vec := make([]float32, 128)
	for i := range vec {
		vec[i] = float32(i%256)/128.0 - 1.0
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = sq.Encode(vec)
	}
}

func BenchmarkScalarQuantizer_Decode(b *testing.B) {
	sq := NewScalarQuantizer()
	sq.min = -1.0
	sq.max = 1.0

	vec := make([]float32, 128)
	for i := range vec {
		vec[i] = float32(i%256)/128.0 - 1.0
	}

	quantized := sq.Encode(vec)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = sq.Decode(quantized)
	}
}
