// Package quantization provides vector quantization implementations for memory-efficient storage.
package quantization

import (
	"encoding/binary"
	"errors"
	"math"
)

// Quantizer defines the interface for vector quantization methods.
type Quantizer interface {
	// Encode quantizes a float32 vector to compressed representation
	Encode(v []float32) []byte

	// Decode reconstructs a float32 vector from quantized representation
	Decode(b []byte) []float32

	// Train calibrates the quantizer on a set of vectors (optional for some quantizers)
	Train(vectors [][]float32) error

	// BytesPerDimension returns the storage size per dimension
	BytesPerDimension() int
}

// ScalarQuantizer implements 8-bit scalar quantization.
// It compresses float32 vectors (4 bytes/dim) to uint8 (1 byte/dim) for 4x memory savings.
type ScalarQuantizer struct {
	min float32 // Global minimum value
	max float32 // Global maximum value
}

// NewScalarQuantizer creates a new 8-bit scalar quantizer.
func NewScalarQuantizer() *ScalarQuantizer {
	return &ScalarQuantizer{
		min: 0,
		max: 1,
	}
}

// Train calibrates the quantizer by finding min/max values across all vectors.
func (sq *ScalarQuantizer) Train(vectors [][]float32) error {
	if len(vectors) == 0 {
		return errors.New("no vectors provided for training")
	}

	sq.min = math.MaxFloat32
	sq.max = -math.MaxFloat32

	for _, vec := range vectors {
		for _, val := range vec {
			if val < sq.min {
				sq.min = val
			}
			if val > sq.max {
				sq.max = val
			}
		}
	}

	// Handle edge case where all values are the same
	if sq.min == sq.max {
		sq.max = sq.min + 1
	}

	return nil
}

// Encode quantizes a float32 vector to 8-bit representation.
// Each dimension is linearly mapped from [min, max] to [0, 255].
func (sq *ScalarQuantizer) Encode(v []float32) []byte {
	quantized := make([]byte, len(v))
	scale := 255.0 / (sq.max - sq.min)

	for i, val := range v {
		// Clamp to [min, max]
		if val < sq.min {
			val = sq.min
		} else if val > sq.max {
			val = sq.max
		}

		// Map to [0, 255]
		normalized := (val - sq.min) * scale
		quantized[i] = uint8(normalized + 0.5) // Round to nearest
	}

	return quantized
}

// Decode reconstructs a float32 vector from quantized representation.
func (sq *ScalarQuantizer) Decode(b []byte) []float32 {
	decoded := make([]float32, len(b))
	scale := (sq.max - sq.min) / 255.0

	for i, val := range b {
		decoded[i] = float32(val)*scale + sq.min
	}

	return decoded
}

// BytesPerDimension returns 1 (uint8 storage).
func (sq *ScalarQuantizer) BytesPerDimension() int {
	return 1
}

// Min returns the minimum value used for quantization
func (sq *ScalarQuantizer) Min() float32 {
	return sq.min
}

// Max returns the maximum value used for quantization
func (sq *ScalarQuantizer) Max() float32 {
	return sq.max
}

// MarshalBinary implements encoding.BinaryMarshaler.
// Format (little-endian): [min:float32][max:float32]
func (sq *ScalarQuantizer) MarshalBinary() ([]byte, error) {
	b := make([]byte, 8)
	binary.LittleEndian.PutUint32(b[0:4], math.Float32bits(sq.min))
	binary.LittleEndian.PutUint32(b[4:8], math.Float32bits(sq.max))
	return b, nil
}

// UnmarshalBinary implements encoding.BinaryUnmarshaler.
func (sq *ScalarQuantizer) UnmarshalBinary(data []byte) error {
	if len(data) != 8 {
		return errors.New("invalid scalar quantizer binary length")
	}
	sq.min = math.Float32frombits(binary.LittleEndian.Uint32(data[0:4]))
	sq.max = math.Float32frombits(binary.LittleEndian.Uint32(data[4:8]))
	return nil
}

// CompressionRatio returns the memory compression ratio (always 4.0 for 8-bit quantization).
func (sq *ScalarQuantizer) CompressionRatio() float64 {
	return 4.0 // float32 (4 bytes) -> uint8 (1 byte)
}

// QuantizationError estimates the average quantization error per dimension.
func (sq *ScalarQuantizer) QuantizationError() float32 {
	// Maximum error is 1 quantization step
	return (sq.max - sq.min) / 512.0 // 255 steps, error is ±0.5 steps
}
