package quantization

import (
	"encoding/binary"
	"errors"
	"math"
	"sync"
	"unsafe"

	"github.com/hupe1980/vecgo/internal/simd"
)

// RaBitQuantizer implements Randomized Binary Quantization (RaBitQ).
// It improves upon standard Binary Quantization by preserving vector magnitudes
// and correcting distance estimates.
//
// Reference: "RaBitQ: Quantizing High-Dimensional Vectors with a Single Bit per Dimension"
//
// The basic idea is:
// L2^2(x, y) = ||x||^2 + ||y||^2 - 2<x, y>
// We approximate <x, y> using Hamming distance of binary codes and the norms.
// <x, y> ≈ (||x|| * ||y|| / Dim) * (Dim - 2 * Hamming(Bx, By))
//
// Storage format:
// [Binary Code (Dim/8 bytes)] + [Norm (4 bytes float32)]
type RaBitQuantizer struct {
	dimension int
	threshold float32 // Usually 0.0

	// Pool for temporary uint64 buffers used in distance calculations
	uint64Pool *sync.Pool
}

// NewRaBitQuantizer creates a new RaBitQ quantizer.
func NewRaBitQuantizer(dimension int) *RaBitQuantizer {
	numWords := (dimension + 63) / 64
	return &RaBitQuantizer{
		dimension: dimension,
		threshold: 0.0,
		uint64Pool: &sync.Pool{
			New: func() any {
				s := make([]uint64, numWords)
				return &s
			},
		},
	}
}

// Encode quantizes a float32 vector to RaBitQ representation.
// Returns binary code followed by float32 norm (little endian).
func (rq *RaBitQuantizer) Encode(v []float32) ([]byte, error) {
	if len(v) != rq.dimension {
		return nil, errors.New("vector dimension mismatch")
	}

	// 1. Compute Norm
	var sumSq float32
	for _, val := range v {
		sumSq += val * val
	}
	norm := float32(math.Sqrt(float64(sumSq)))

	// 2. Binary Encode
	numWords := (len(v) + 63) / 64
	numBytes := numWords * 8
	result := make([]byte, numBytes+4) // +4 for float32 norm

	// Encode binary part
	for i, val := range v {
		if val >= rq.threshold {
			byteIdx := (i/64)*8 + (i%64)/8
			bitIdx := i % 8
			result[byteIdx] |= 1 << bitIdx
		}
	}

	// 3. Append Norm
	binary.LittleEndian.PutUint32(result[numBytes:], math.Float32bits(norm))

	return result, nil
}

// Decode reconstructs a float32 vector (approximate).
// It restores magnitude but direction is quantized to binary hypercube vertices.
func (rq *RaBitQuantizer) Decode(b []byte) ([]float32, error) {
	numWords := (rq.dimension + 63) / 64
	numBytes := numWords * 8

	if len(b) < numBytes+4 {
		return nil, errors.New("invalid encoded data length")
	}

	// Extract norm
	normBits := binary.LittleEndian.Uint32(b[numBytes:])
	norm := math.Float32frombits(normBits)

	// Reconstruct
	// Each component is +/- (norm / sqrt(Dim))
	scale := norm / float32(math.Sqrt(float64(rq.dimension)))
	decoded := make([]float32, rq.dimension)

	for i := 0; i < rq.dimension; i++ {
		byteIdx := i / 8
		bitIdx := i % 8 // simplified byte indexing compared to word-based
		// Wait, Encode used: byteIdx := (i / 64) * 8 + (i % 64) / 8
		// Let's match that.
		// i/64 is word index. (i%64)/8 is byte within word.
		// i/8 gives byte index in linear byte array. (i/64)*8 + (i%64)/8 == i/8 (integer division).
		// Yes, it matches.

		if byteIdx < numBytes && (b[byteIdx]&(1<<bitIdx)) != 0 {
			decoded[i] = scale
		} else {
			decoded[i] = -scale
		}
	}

	return decoded, nil
}

// Distance computes the approximate L2 distance squared between a query vector and a quantized vector.
func (rq *RaBitQuantizer) Distance(query []float32, code []byte) (float32, error) {
	// Unpack norm from code
	numWords := (rq.dimension + 63) / 64
	numBytes := numWords * 8
	if len(code) < numBytes+4 {
		return 0, errors.New("invalid code length")
	}

	yNormBits := binary.LittleEndian.Uint32(code[numBytes:])
	yNorm := math.Float32frombits(yNormBits)

	// Compute query norm (can be optimized if caller provides it)
	// For now, compute it.
	var qSumSq float32
	for _, v := range query {
		qSumSq += v * v
	}
	qNorm := float32(math.Sqrt(float64(qSumSq)))

	// Compute Hamming distance
	// Map query to binary code locally
	qCodesPtr := rq.uint64Pool.Get().(*[]uint64)
	qCodes := *qCodesPtr
	defer rq.uint64Pool.Put(qCodesPtr)

	// Zero out
	for i := range qCodes {
		qCodes[i] = 0
	}

	// Basic sign encoding for query
	for i, val := range query {
		if val >= rq.threshold {
			wordIdx := i / 64
			bitIdx := i % 64
			qCodes[wordIdx] |= 1 << bitIdx
		}
	}

	// Code provided as []byte, convert/cast to []uint64 for POPCNT
	// Note: code is little endian, so we can cast.
	// We need to be careful with alignment if we cast, but []byte usually isn't aligned.
	// Copying to a pooled buffer is safer and maybe faster than unaligned reads if SIMD enforces it.
	// Or use simd.Hamming which takes []byte.

	// However, we have qCodes as []uint64. We need to compare it to code[:numBytes].
	// simd.Hamming takes []byte.
	// Let's create a temporary []byte view of qCodes.
	qBytes := unsafe.Slice((*byte)(unsafe.Pointer(&qCodes[0])), len(qCodes)*8)

	hamming := float32(simd.Hamming(qBytes, code[:numBytes]))

	// RaBitQ Formula:
	// d^2(x,y) ≈ ||x||^2 + ||y||^2 - 2 * (||x||*||y||/D) * (D - 2*Hamming)
	//          = ||x||^2 + ||y||^2 - 2*||x||*||y|| + 4*||x||*||y||*Hamming/D
	//          = (||x|| - ||y||)^2 + (4*||x||*||y||/D) * Hamming

	term1 := (qNorm - yNorm)
	term1Sq := term1 * term1

	term2 := (4.0 * qNorm * yNorm / float32(rq.dimension)) * hamming

	return term1Sq + term2, nil
}

// Train is a no-op for basic RaBitQ (unless we add rotation learning).
func (rq *RaBitQuantizer) Train(vectors [][]float32) error {
	return nil
}

func (rq *RaBitQuantizer) BytesPerDimension() int {
	return 0 // Sub-byte
}

func (rq *RaBitQuantizer) BytesTotal() int {
	numWords := (rq.dimension + 63) / 64
	return (numWords * 8) + 4
}
