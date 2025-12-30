package quantization

import (
	"testing"

	"github.com/hupe1980/vecgo/testutil"
)

func TestBinaryQuantizer_Basic(t *testing.T) {
	bq := NewBinaryQuantizer(128)

	// Test with threshold 0.0 (sign-based)
	vec := make([]float32, 128)
	for i := range vec {
		if i%2 == 0 {
			vec[i] = 1.0
		} else {
			vec[i] = -1.0
		}
	}

	encoded := bq.EncodeUint64(vec)
	if len(encoded) != 2 { // 128 bits = 2 uint64
		t.Errorf("expected 2 uint64 words, got %d", len(encoded))
	}

	// Verify encoding: even bits should be 1, odd bits should be 0
	// This creates a pattern of 0x5555... (alternating bits)
	expected := uint64(0x5555555555555555)
	if encoded[0] != expected {
		t.Errorf("expected first word %x, got %x", expected, encoded[0])
	}
	if encoded[1] != expected {
		t.Errorf("expected second word %x, got %x", expected, encoded[1])
	}
}

func TestBinaryQuantizer_Train(t *testing.T) {
	bq := NewBinaryQuantizer(4)

	vectors := [][]float32{
		{1.0, 2.0, 3.0, 4.0},
		{5.0, 6.0, 7.0, 8.0},
	}

	if err := bq.Train(vectors); err != nil {
		t.Fatalf("Train failed: %v", err)
	}

	if !bq.IsTrained() {
		t.Error("expected IsTrained to be true after training")
	}

	// Mean should be (1+2+3+4+5+6+7+8)/8 = 4.5
	expectedThreshold := float32(4.5)
	if bq.Threshold() != expectedThreshold {
		t.Errorf("expected threshold %f, got %f", expectedThreshold, bq.Threshold())
	}
}

func TestBinaryQuantizer_WithThreshold(t *testing.T) {
	bq := NewBinaryQuantizer(8).WithThreshold(0.5)

	// vec[i]: 0.0, 0.4, 0.5, 0.6, 1.0, -1.0, 0.5, 0.49
	// >= 0.5:  0,   0,   1,   1,   1,    0,   1,   0
	// Bits:    0    1    2    3    4     5    6    7
	// Binary (LSB first): bit2=1, bit3=1, bit4=1, bit6=1
	// = 0b01011100 = 0x5C
	vec := []float32{0.0, 0.4, 0.5, 0.6, 1.0, -1.0, 0.5, 0.49}
	encoded := bq.EncodeUint64(vec)

	expected := uint64(0b01011100)
	if encoded[0] != expected {
		t.Errorf("expected %08b, got %08b", expected, encoded[0])
	}
}

func TestHammingDistance(t *testing.T) {
	tests := []struct {
		a, b     []uint64
		expected int
	}{
		{[]uint64{0}, []uint64{0}, 0},
		{[]uint64{1}, []uint64{0}, 1},
		{[]uint64{0xFF}, []uint64{0}, 8},
		{[]uint64{0xFFFFFFFFFFFFFFFF}, []uint64{0}, 64},
		{[]uint64{0x5555555555555555}, []uint64{0xAAAAAAAAAAAAAAAA}, 64}, // All bits differ
		{[]uint64{0, 0}, []uint64{0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF}, 128},
	}

	for i, tt := range tests {
		dist := HammingDistance(tt.a, tt.b)
		if dist != tt.expected {
			t.Errorf("test %d: expected %d, got %d", i, tt.expected, dist)
		}
	}
}

func TestHammingDistanceBytes(t *testing.T) {
	tests := []struct {
		a, b     []byte
		expected int
	}{
		{[]byte{0}, []byte{0}, 0},
		{[]byte{1}, []byte{0}, 1},
		{[]byte{0xFF}, []byte{0}, 8},
		{[]byte{0xFF, 0xFF}, []byte{0, 0}, 16},
		{[]byte{0x55}, []byte{0xAA}, 8}, // All bits differ in byte
	}

	for i, tt := range tests {
		dist := HammingDistanceBytes(tt.a, tt.b)
		if dist != tt.expected {
			t.Errorf("test %d: expected %d, got %d", i, tt.expected, dist)
		}
	}
}

func TestBinaryQuantizer_EncodeDecode_Roundtrip(t *testing.T) {
	bq := NewBinaryQuantizer(128).WithThreshold(0.0)

	// Create random vector
	rng := testutil.NewRNG(42)
	vec := rng.UniformRangeVectors(1, 128)[0]

	encoded := bq.Encode(vec)
	decoded := bq.Decode(encoded)

	// Check that signs are preserved
	for i := range vec {
		originalSign := vec[i] >= 0
		decodedSign := decoded[i] >= 0
		if originalSign != decodedSign {
			t.Errorf("sign mismatch at index %d: original=%v (%.2f), decoded=%v (%.2f)",
				i, originalSign, vec[i], decodedSign, decoded[i])
		}
	}
}

func TestBinaryQuantizer_BytesTotal(t *testing.T) {
	tests := []struct {
		dim      int
		expected int
	}{
		{8, 1},
		{16, 2},
		{64, 8},
		{128, 16},
		{100, 13}, // ceil(100/8) = 13
		{1, 1},
	}

	for _, tt := range tests {
		bq := NewBinaryQuantizer(tt.dim)
		if bq.BytesTotal() != tt.expected {
			t.Errorf("dim=%d: expected %d bytes, got %d", tt.dim, tt.expected, bq.BytesTotal())
		}
	}
}

func TestBinaryQuantizer_CompressionRatio(t *testing.T) {
	bq := NewBinaryQuantizer(128)
	if bq.CompressionRatio() != 32.0 {
		t.Errorf("expected compression ratio 32x, got %f", bq.CompressionRatio())
	}
}

func TestNormalizedHammingDistance(t *testing.T) {
	a := []uint64{0}
	b := []uint64{0xFF} // 8 bits set

	dist := NormalizedHammingDistance(a, b, 64)
	expected := float32(8) / float32(64)
	if dist != expected {
		t.Errorf("expected %f, got %f", expected, dist)
	}
}

// BenchmarkHammingDistance benchmarks Hamming distance for 128-dim vectors
func BenchmarkHammingDistance_128dim(b *testing.B) {
	a := make([]uint64, 2) // 128 bits
	c := make([]uint64, 2)
	a[0] = 0x5555555555555555
	a[1] = 0xAAAAAAAAAAAAAAAA
	c[0] = 0xAAAAAAAAAAAAAAAA
	c[1] = 0x5555555555555555

	b.ResetTimer()
	for b.Loop() {
		_ = HammingDistance(a, c)
	}
}

// BenchmarkHammingDistance benchmarks Hamming distance for 768-dim vectors (common embedding size)
func BenchmarkHammingDistance_768dim(b *testing.B) {
	a := make([]uint64, 12) // 768 bits
	c := make([]uint64, 12)
	for i := range a {
		a[i] = 0x5555555555555555
		c[i] = 0xAAAAAAAAAAAAAAAA
	}

	b.ResetTimer()
	for b.Loop() {
		_ = HammingDistance(a, c)
	}
}

// BenchmarkBinaryEncode benchmarks encoding for 128-dim vectors
func BenchmarkBinaryEncode_128dim(b *testing.B) {
	bq := NewBinaryQuantizer(128)
	rng := testutil.NewRNG(42)
	vec := rng.UniformRangeVectors(1, 128)[0]

	b.ResetTimer()
	for b.Loop() {
		_ = bq.EncodeUint64(vec)
	}
}

// BenchmarkBinaryEncode benchmarks encoding for 768-dim vectors
func BenchmarkBinaryEncode_768dim(b *testing.B) {
	bq := NewBinaryQuantizer(768)
	rng := testutil.NewRNG(42)
	vec := rng.UniformRangeVectors(1, 768)[0]

	b.ResetTimer()
	for b.Loop() {
		_ = bq.EncodeUint64(vec)
	}
}
