package quantization

import (
	"encoding/binary"
	"errors"
	"math"

	"github.com/hupe1980/vecgo/internal/simd"
)

// Int4Quantizer implements 4-bit scalar quantization.
// It compresses float32 vectors (4 bytes/dim) to 4 bits (0.5 byte/dim) for 8x memory savings.
// Two dimensions are packed into a single byte.
type Int4Quantizer struct {
	min         []float32
	diff        []float32
	dim         int
	lookupTable []float32 // Pre-computed dequantization table (16 * dim)
}

// NewInt4Quantizer creates a new Int4Quantizer.
func NewInt4Quantizer(dim int) *Int4Quantizer {
	return &Int4Quantizer{
		dim: dim,
	}
}

// Train calculates min/max ranges for quantization.
func (q *Int4Quantizer) Train(vectors [][]float32) error {
	if len(vectors) == 0 {
		return nil
	}

	q.dim = len(vectors[0])
	q.min = make([]float32, q.dim)
	max := make([]float32, q.dim)

	// Initialize with first vector
	copy(q.min, vectors[0])
	copy(max, vectors[0])

	for _, v := range vectors[1:] {
		for i, val := range v {
			if val < q.min[i] {
				q.min[i] = val
			}
			if val > max[i] {
				max[i] = val
			}
		}
	}

	q.diff = make([]float32, q.dim)
	for i := 0; i < q.dim; i++ {
		q.diff[i] = max[i] - q.min[i]
		if q.diff[i] == 0 {
			q.diff[i] = 1.0 // Avoid division by zero
		}
	}

	// Pre-compute lookup table for SIMD-optimized distance
	q.lookupTable = simd.BuildInt4LookupTable(q.min, q.diff)

	return nil
}

// Encode quantizes a vector to 4-bit packed bytes.
func (q *Int4Quantizer) Encode(v []float32) ([]byte, error) {
	if len(v) != q.dim {
		return nil, errors.New("dimension mismatch")
	}

	// Output size is ceil(dim / 2)
	outSize := (q.dim + 1) / 2
	out := make([]byte, outSize)

	for i := 0; i < q.dim; i += 2 {
		// First value (high nibble)
		val1 := v[i]
		norm1 := (val1 - q.min[i]) / q.diff[i]
		if norm1 < 0 {
			norm1 = 0
		} else if norm1 > 1 {
			norm1 = 1
		}
		quant1 := byte(math.Round(float64(norm1) * 15))

		// Second value (low nibble), if exists
		quant2 := byte(0)
		if i+1 < q.dim {
			val2 := v[i+1]
			norm2 := (val2 - q.min[i+1]) / q.diff[i+1]
			if norm2 < 0 {
				norm2 = 0
			} else if norm2 > 1 {
				norm2 = 1
			}
			quant2 = byte(math.Round(float64(norm2) * 15))
		}

		// Pack: High nibble | Low nibble
		out[i/2] = (quant1 << 4) | (quant2 & 0x0F)
	}

	return out, nil
}

// Decode reconstructs the vector.
func (q *Int4Quantizer) Decode(b []byte) ([]float32, error) {
	expectedSize := (q.dim + 1) / 2
	if len(b) != expectedSize {
		return nil, errors.New("dimension mismatch")
	}

	out := make([]float32, q.dim)

	for i := 0; i < q.dim; i += 2 {
		byteVal := b[i/2]

		// Unpack high nibble
		quant1 := (byteVal >> 4) & 0x0F
		out[i] = float32(quant1)/15.0*q.diff[i] + q.min[i]

		// Unpack low nibble
		if i+1 < q.dim {
			quant2 := byteVal & 0x0F
			out[i+1] = float32(quant2)/15.0*q.diff[i+1] + q.min[i+1]
		}
	}

	return out, nil
}

// L2Distance computes squared L2 distance between query and compressed vector.
// Uses SIMD-optimized precomputed lookup tables for fast distance calculation.
func (q *Int4Quantizer) L2Distance(query []float32, code []byte) (float32, error) {
	if len(query) != q.dim || len(code) != (q.dim+1)/2 {
		return 0, errors.New("dimension mismatch")
	}

	// Use SIMD-optimized precomputed lookup if available
	if q.lookupTable != nil {
		return simd.Int4L2DistancePrecomputed(query, code, q.lookupTable), nil
	}

	// Fallback to direct SIMD computation
	return simd.Int4L2Distance(query, code, q.min, q.diff), nil
}

// L2DistanceBatch computes squared L2 distance for a batch of codes.
// Uses SIMD-optimized batch computation for better cache locality.
func (q *Int4Quantizer) L2DistanceBatch(query []float32, codes []byte, n int, out []float32) error {
	codeSize := (q.dim + 1) / 2
	if len(codes) < n*codeSize {
		return errors.New("codes buffer too small")
	}
	if len(out) < n {
		return errors.New("output buffer too small")
	}

	// Use SIMD-optimized batch computation
	simd.Int4L2DistanceBatch(query, codes, q.dim, n, q.min, q.diff, out)
	return nil
}

func (q *Int4Quantizer) BytesPerDimension() int {
	return 0 // Sub-byte
}

// MarshalBinary serializes the quantizer state.
func (q *Int4Quantizer) MarshalBinary() ([]byte, error) {
	size := 4 + (q.dim * 4) + (q.dim * 4)
	buf := make([]byte, size)

	binary.LittleEndian.PutUint32(buf[0:4], uint32(q.dim))

	offset := 4
	for _, val := range q.min {
		binary.LittleEndian.PutUint32(buf[offset:], math.Float32bits(val))
		offset += 4
	}
	for _, val := range q.diff {
		binary.LittleEndian.PutUint32(buf[offset:], math.Float32bits(val))
		offset += 4
	}

	return buf, nil
}

// UnmarshalBinary deserializes the quantizer state.
func (q *Int4Quantizer) UnmarshalBinary(data []byte) error {
	if len(data) < 4 {
		return errors.New("data too short")
	}

	q.dim = int(binary.LittleEndian.Uint32(data[0:4]))
	expectedSize := 4 + (q.dim * 8)
	if len(data) != expectedSize {
		return errors.New("data size mismatch")
	}

	q.min = make([]float32, q.dim)
	q.diff = make([]float32, q.dim)

	offset := 4
	for i := 0; i < q.dim; i++ {
		q.min[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[offset:]))
		offset += 4
	}
	for i := 0; i < q.dim; i++ {
		q.diff[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[offset:]))
		offset += 4
	}

	// Rebuild lookup table for SIMD-optimized distance
	q.lookupTable = simd.BuildInt4LookupTable(q.min, q.diff)

	return nil
}
