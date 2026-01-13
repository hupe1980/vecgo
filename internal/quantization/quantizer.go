// Package quantization provides vector quantization implementations for memory-efficient storage.
package quantization

import (
	"encoding/binary"
	"errors"
	"math"
	"sync"

	"github.com/hupe1980/vecgo/internal/simd"
)

// Quantizer defines the interface for vector quantization methods.
type Quantizer interface {
	// Encode quantizes a float32 vector to compressed representation
	Encode(v []float32) ([]byte, error)

	// Decode reconstructs a float32 vector from quantized representation
	Decode(b []byte) ([]float32, error)

	// Train calibrates the quantizer on a set of vectors (optional for some quantizers)
	Train(vectors [][]float32) error

	// BytesPerDimension returns the storage size per dimension
	BytesPerDimension() int
}

// ScalarQuantizer implements 8-bit scalar quantization.
// It compresses float32 vectors (4 bytes/dim) to uint8 (1 byte/dim) for 4x memory savings.
//
// This implementation uses per-dimension min/max values to maximize precision,
// which significantly improves recall compared to global min/max.
type ScalarQuantizer struct {
	mins      []float32 // Per-dimension minimum values
	maxs      []float32 // Per-dimension maximum values
	scales    []float32 // Precomputed scales: 255 / (max - min)
	invScales []float32 // Precomputed inverse scales: (max - min) / 255
	dimension int       // Vector dimension
	trained   bool      // Whether the quantizer has been trained

	// Pools for temporary buffers
	bytePool  *sync.Pool
	floatPool *sync.Pool
}

// Mins returns the per-dimension minimum values.
func (sq *ScalarQuantizer) Mins() []float32 {
	return sq.mins
}

// Maxs returns the per-dimension maximum values.
func (sq *ScalarQuantizer) Maxs() []float32 {
	return sq.maxs
}

// SetBounds initializes the quantizer with pre-computed bounds.
func (sq *ScalarQuantizer) SetBounds(mins, maxs []float32) error {
	if len(mins) != sq.dimension || len(maxs) != sq.dimension {
		return errors.New("dimension mismatch")
	}
	sq.mins = make([]float32, sq.dimension)
	sq.maxs = make([]float32, sq.dimension)
	sq.scales = make([]float32, sq.dimension)
	sq.invScales = make([]float32, sq.dimension)

	copy(sq.mins, mins)
	copy(sq.maxs, maxs)

	for i := range sq.dimension {
		diff := sq.maxs[i] - sq.mins[i]
		if diff < 1e-9 {
			sq.scales[i] = 0
			sq.invScales[i] = 0
		} else {
			sq.scales[i] = 255.0 / diff
			sq.invScales[i] = diff / 255.0
		}
	}
	sq.trained = true
	return nil
}

// L2Distance computes the squared L2 distance between a float32 query and a quantized vector.
func (sq *ScalarQuantizer) L2Distance(q []float32, code []byte) (float32, error) {
	if len(q) != sq.dimension || len(code) != sq.dimension {
		return 0, errors.New("dimension mismatch")
	}
	var dist float32
	for i := 0; i < sq.dimension; i++ {
		// Reconstruct value: min + code * scale
		// Note: scale = 255 / (max - min), so we multiply by invScale = (max - min) / 255
		val := sq.mins[i] + float32(code[i])*sq.invScales[i]
		diff := q[i] - val
		dist += diff * diff
	}
	return dist, nil
}

// L2DistanceBatch computes the squared L2 distance between a float32 query and a batch of quantized vectors.
func (sq *ScalarQuantizer) L2DistanceBatch(q []float32, codes []byte, n int, out []float32) error {
	if len(q) != sq.dimension {
		return errors.New("query dimension mismatch")
	}
	if len(codes) < n*sq.dimension {
		return errors.New("codes buffer too small")
	}
	if len(out) < n {
		return errors.New("output buffer too small")
	}
	simd.Sq8uL2BatchPerDimension(q, codes, sq.mins, sq.invScales, sq.dimension, out)
	return nil
}

// DotProduct computes the dot product between a float32 query and a quantized vector.
func (sq *ScalarQuantizer) DotProduct(q []float32, code []byte) (float32, error) {
	if len(q) != sq.dimension || len(code) != sq.dimension {
		return 0, errors.New("dimension mismatch")
	}
	var dot float32
	for i := 0; i < sq.dimension; i++ {
		val := sq.mins[i] + float32(code[i])*sq.invScales[i]
		dot += q[i] * val
	}
	return dot, nil
}

// NewScalarQuantizer creates a new 8-bit scalar quantizer for the given dimension.
func NewScalarQuantizer(dimension int) *ScalarQuantizer {
	return &ScalarQuantizer{
		dimension: dimension,
		trained:   false,
		bytePool: &sync.Pool{
			New: func() any {
				return make([]byte, dimension)
			},
		},
		floatPool: &sync.Pool{
			New: func() any {
				return make([]float32, dimension)
			},
		},
	}
}

// Train calibrates the quantizer by finding min/max values per dimension across all vectors.
func (sq *ScalarQuantizer) Train(vectors [][]float32) error {
	if len(vectors) == 0 {
		return errors.New("no vectors provided for training")
	}

	dim := len(vectors[0])
	if dim != sq.dimension {
		return errors.New("vector dimension mismatch")
	}

	sq.mins = make([]float32, dim)
	sq.maxs = make([]float32, dim)
	sq.scales = make([]float32, dim)
	sq.invScales = make([]float32, dim)

	// Initialize min/max
	for i := range dim {
		sq.mins[i] = math.MaxFloat32
		sq.maxs[i] = -math.MaxFloat32
	}

	// Find min/max per dimension
	for _, vec := range vectors {
		if len(vec) != dim {
			return errors.New("inconsistent vector dimension")
		}
		for i, val := range vec {
			if val < sq.mins[i] {
				sq.mins[i] = val
			}
			if val > sq.maxs[i] {
				sq.maxs[i] = val
			}
		}
	}

	// Compute scales
	for i := range dim {
		// Handle edge case where min == max (constant dimension)
		if sq.mins[i] == sq.maxs[i] {
			sq.maxs[i] = sq.mins[i] + 1e-6 // Avoid division by zero
		}

		rangeVal := sq.maxs[i] - sq.mins[i]
		sq.scales[i] = 255.0 / rangeVal
		sq.invScales[i] = rangeVal / 255.0
	}

	sq.trained = true
	return nil
}

// Encode quantizes a float32 vector to 8-bit representation.
// Each dimension is linearly mapped from [min, max] to [0, 255].
func (sq *ScalarQuantizer) Encode(v []float32) ([]byte, error) {
	if !sq.trained {
		return nil, errors.New("ScalarQuantizer not trained")
	}
	if len(v) != sq.dimension {
		return nil, errors.New("vector dimension mismatch")
	}

	// Use pooled buffer if possible, but Encode returns a new slice usually.
	// If we want to return a new slice, we can't pool it unless we change API to EncodeInto.
	// For now, we allocate. To optimize, we'd need EncodeInto.
	// However, the user suggestion was "Pool buffers".
	// If the caller expects to own the returned slice, we must allocate.
	// Let's stick to allocation for safety unless we change the interface.
	// But wait, the user said "Pool buffers... quantized := make([]byte, len(v))".
	// If we return it, we can't pool it easily without a Release() mechanism.
	// So we will allocate for now to be safe with the interface.
	quantized := make([]byte, len(v))

	for i, val := range v {
		minVal := sq.mins[i]
		maxVal := sq.maxs[i]
		scale := sq.scales[i]

		// Clamp to [min, max]
		if val < minVal {
			val = minVal
		} else if val > maxVal {
			val = maxVal
		}

		// Map to [0, 255]
		normalized := (val - minVal) * scale
		quantized[i] = uint8(normalized + 0.5) // Round to nearest
	}

	return quantized, nil
}

// Decode reconstructs a float32 vector from quantized representation.
func (sq *ScalarQuantizer) Decode(b []byte) ([]float32, error) {
	if !sq.trained {
		return nil, errors.New("ScalarQuantizer not trained")
	}
	if len(b) != sq.dimension {
		return nil, errors.New("vector dimension mismatch")
	}

	decoded := make([]float32, len(b))
	invScales := sq.invScales
	mins := sq.mins

	// Bounds check elimination hint
	if len(invScales) < len(b) || len(mins) < len(b) {
		return nil, errors.New("invalid quantizer state")
	}

	for i, val := range b {
		decoded[i] = float32(val)*invScales[i] + mins[i]
	}

	return decoded, nil
}

// BytesPerDimension returns 1 (uint8 storage).
func (sq *ScalarQuantizer) BytesPerDimension() int {
	return 1
}

// Min returns the minimum value used for quantization for a specific dimension.
func (sq *ScalarQuantizer) Min(dim int) float32 {
	if !sq.trained || dim < 0 || dim >= len(sq.mins) {
		return 0
	}
	return sq.mins[dim]
}

// Max returns the maximum value used for quantization for a specific dimension.
func (sq *ScalarQuantizer) Max(dim int) float32 {
	if !sq.trained || dim < 0 || dim >= len(sq.maxs) {
		return 0
	}
	return sq.maxs[dim]
}

// MarshalBinary implements encoding.BinaryMarshaler.
// Format (little-endian):
// [dimension:uint32]
// [min_0:float32][max_0:float32]...[min_n:float32][max_n:float32]
func (sq *ScalarQuantizer) MarshalBinary() ([]byte, error) {
	if !sq.trained {
		return nil, errors.New("ScalarQuantizer not trained")
	}

	buf := make([]byte, 4+sq.dimension*8)
	binary.LittleEndian.PutUint32(buf[0:4], uint32(sq.dimension))

	offset := 4
	for i := 0; i < sq.dimension; i++ {
		binary.LittleEndian.PutUint32(buf[offset:offset+4], math.Float32bits(sq.mins[i]))
		binary.LittleEndian.PutUint32(buf[offset+4:offset+8], math.Float32bits(sq.maxs[i]))
		offset += 8
	}
	return buf, nil
}

// UnmarshalBinary implements encoding.BinaryUnmarshaler.
func (sq *ScalarQuantizer) UnmarshalBinary(data []byte) error {
	if len(data) < 4 {
		return errors.New("invalid scalar quantizer binary length")
	}

	sq.dimension = int(binary.LittleEndian.Uint32(data[0:4]))
	expectedLen := 4 + sq.dimension*8
	if len(data) != expectedLen {
		return errors.New("invalid scalar quantizer binary length for dimension")
	}

	sq.mins = make([]float32, sq.dimension)
	sq.maxs = make([]float32, sq.dimension)
	sq.scales = make([]float32, sq.dimension)
	sq.invScales = make([]float32, sq.dimension)

	offset := 4
	for i := 0; i < sq.dimension; i++ {
		sq.mins[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[offset : offset+4]))
		sq.maxs[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[offset+4 : offset+8]))
		offset += 8

		// Recompute scales
		if sq.mins[i] == sq.maxs[i] {
			sq.maxs[i] = sq.mins[i] + 1e-6
		}
		rangeVal := sq.maxs[i] - sq.mins[i]
		sq.scales[i] = 255.0 / rangeVal
		sq.invScales[i] = rangeVal / 255.0
	}

	sq.trained = true

	// Re-initialize pools
	sq.bytePool = &sync.Pool{
		New: func() interface{} {
			return make([]byte, sq.dimension)
		},
	}
	sq.floatPool = &sync.Pool{
		New: func() any {
			return make([]float32, sq.dimension)
		},
	}

	return nil
}

// CompressionRatio returns the memory compression ratio (always 4.0 for 8-bit quantization).
func (sq *ScalarQuantizer) CompressionRatio() float64 {
	return 4.0 // float32 (4 bytes) -> uint8 (1 byte)
}

// QuantizationError estimates the average quantization error per dimension.
// This is a theoretical lower bound assuming uniform distribution.
func (sq *ScalarQuantizer) QuantizationError() float32 {
	if !sq.trained {
		return 0
	}
	// Average error across dimensions
	var totalRange float32
	for i := 0; i < sq.dimension; i++ {
		totalRange += (sq.maxs[i] - sq.mins[i])
	}
	avgRange := totalRange / float32(sq.dimension)
	return avgRange / 512.0
}
