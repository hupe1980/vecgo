// Package quantization provides vector quantization implementations for memory-efficient storage.
package quantization

import (
	"errors"
	"math/bits"
)

// BinaryQuantizer implements binary quantization (1-bit per dimension).
// It compresses float32 vectors (4 bytes/dim) to bits (0.125 bytes/dim) for 32x memory savings.
//
// Binary quantization uses a simple threshold: values >= threshold become 1, otherwise 0.
// Distance is computed using Hamming distance (popcount of XOR), which is extremely fast
// on modern CPUs using the POPCNT instruction.
//
// Trade-offs:
//   - 32x compression ratio (vs float32)
//   - Very fast distance computation (Hamming via POPCNT)
//   - Significant accuracy loss for fine-grained similarity
//   - Best used for coarse filtering or with reranking
type BinaryQuantizer struct {
	dimension int     // Expected vector dimension
	threshold float32 // Value threshold for binary encoding
	trained   bool    // Whether threshold has been calibrated
}

// NewBinaryQuantizer creates a new binary quantizer for the given dimension.
// The default threshold is 0.0 (sign-based quantization).
func NewBinaryQuantizer(dimension int) *BinaryQuantizer {
	return &BinaryQuantizer{
		dimension: dimension,
		threshold: 0.0,
		trained:   false,
	}
}

// WithThreshold sets a custom threshold for binary encoding.
// Values >= threshold become 1, values < threshold become 0.
func (bq *BinaryQuantizer) WithThreshold(threshold float32) *BinaryQuantizer {
	bq.threshold = threshold
	bq.trained = true
	return bq
}

// Train calibrates the quantizer by computing the mean value across all vectors.
// The mean is used as the threshold for binary encoding.
func (bq *BinaryQuantizer) Train(vectors [][]float32) error {
	if len(vectors) == 0 {
		return errors.New("no vectors provided for training")
	}

	// Compute global mean
	var sum float64
	var count int
	for _, vec := range vectors {
		for _, val := range vec {
			sum += float64(val)
			count++
		}
	}

	if count > 0 {
		bq.threshold = float32(sum / float64(count))
	}
	bq.trained = true

	return nil
}

// Encode quantizes a float32 vector to binary representation.
// Each dimension is converted to a single bit based on the threshold.
// The result is packed into uint64 words for efficient storage and POPCNT operations.
//
// Storage format: ceil(dimension / 64) uint64 words, little-endian bit packing.
func (bq *BinaryQuantizer) Encode(v []float32) []byte {
	numWords := (len(v) + 63) / 64
	result := make([]byte, numWords*8)

	for i, val := range v {
		if val >= bq.threshold {
			wordIdx := i / 64
			bitIdx := i % 64
			// Set bit in the appropriate byte within the word
			byteIdx := wordIdx*8 + bitIdx/8
			result[byteIdx] |= 1 << (bitIdx % 8)
		}
	}

	return result
}

// EncodeUint64 quantizes a float32 vector to packed uint64 words.
// This is more efficient for distance computation as it avoids byte-to-uint64 conversion.
func (bq *BinaryQuantizer) EncodeUint64(v []float32) []uint64 {
	numWords := (len(v) + 63) / 64
	result := make([]uint64, numWords)

	for i, val := range v {
		if val >= bq.threshold {
			wordIdx := i / 64
			bitIdx := i % 64
			result[wordIdx] |= 1 << bitIdx
		}
	}

	return result
}

// Decode reconstructs a float32 vector from binary representation.
// Note: This is a lossy reconstruction - values are either threshold-0.5 or threshold+0.5.
func (bq *BinaryQuantizer) Decode(b []byte) []float32 {
	decoded := make([]float32, bq.dimension)

	for i := 0; i < bq.dimension; i++ {
		byteIdx := i / 8
		bitIdx := i % 8
		if byteIdx < len(b) && (b[byteIdx]&(1<<bitIdx)) != 0 {
			decoded[i] = bq.threshold + 0.5
		} else {
			decoded[i] = bq.threshold - 0.5
		}
	}

	return decoded
}

// BytesPerDimension returns the storage size per dimension (0.125 = 1 bit).
func (bq *BinaryQuantizer) BytesPerDimension() int {
	return 0 // Actually 1/8, but interface uses int. Use BytesTotal() instead.
}

// BytesTotal returns the total storage size for a vector.
func (bq *BinaryQuantizer) BytesTotal() int {
	return (bq.dimension + 7) / 8
}

// Dimension returns the expected vector dimension.
func (bq *BinaryQuantizer) Dimension() int {
	return bq.dimension
}

// Threshold returns the current threshold value.
func (bq *BinaryQuantizer) Threshold() float32 {
	return bq.threshold
}

// IsTrained returns whether the quantizer has been trained.
func (bq *BinaryQuantizer) IsTrained() bool {
	return bq.trained
}

// HammingDistance computes the Hamming distance between two binary-encoded vectors.
// This counts the number of bit positions where the vectors differ.
// Uses POPCNT (population count) for maximum performance.
func HammingDistance(a, b []uint64) int {
	if len(a) != len(b) {
		// Handle mismatched lengths by using the shorter one
		if len(a) > len(b) {
			a = a[:len(b)]
		} else {
			b = b[:len(a)]
		}
	}

	var dist int
	for i := range a {
		dist += bits.OnesCount64(a[i] ^ b[i])
	}
	return dist
}

// HammingDistanceBytes computes the Hamming distance between two byte slices.
// This is a convenience wrapper that converts bytes to uint64 for POPCNT.
func HammingDistanceBytes(a, b []byte) int {
	// Use byte-level POPCNT for smaller vectors or misaligned data
	minLen := len(a)
	if len(b) < minLen {
		minLen = len(b)
	}

	var dist int
	// Process 8 bytes at a time using uint64 POPCNT
	i := 0
	for ; i+8 <= minLen; i += 8 {
		aWord := uint64(a[i]) | uint64(a[i+1])<<8 | uint64(a[i+2])<<16 | uint64(a[i+3])<<24 |
			uint64(a[i+4])<<32 | uint64(a[i+5])<<40 | uint64(a[i+6])<<48 | uint64(a[i+7])<<56
		bWord := uint64(b[i]) | uint64(b[i+1])<<8 | uint64(b[i+2])<<16 | uint64(b[i+3])<<24 |
			uint64(b[i+4])<<32 | uint64(b[i+5])<<40 | uint64(b[i+6])<<48 | uint64(b[i+7])<<56
		dist += bits.OnesCount64(aWord ^ bWord)
	}

	// Process remaining bytes
	for ; i < minLen; i++ {
		dist += bits.OnesCount8(a[i] ^ b[i])
	}

	return dist
}

// NormalizedHammingDistance returns Hamming distance normalized to [0, 1].
// This is useful for comparing with other distance metrics.
func NormalizedHammingDistance(a, b []uint64, dimension int) float32 {
	dist := HammingDistance(a, b)
	return float32(dist) / float32(dimension)
}

// CompressionRatio returns the compression ratio vs float32 storage.
// For binary quantization, this is always 32x.
func (bq *BinaryQuantizer) CompressionRatio() float32 {
	return 32.0
}
