package simd

import (
	"encoding/binary"
	"math/bits"
)

var (
	dotImpl       = dotGeneric
	squaredL2Impl = squaredL2Generic
	scaleImpl     = scaleGeneric
	pqAdcImpl     = pqAdcLookupGeneric

	// Batch kernels
	squaredL2BatchImpl = squaredL2BatchGeneric
	dotBatchImpl       = dotBatchGeneric

	// SQ8 and Hamming kernels
	sq8uL2BatchPerDimensionImpl = sq8uL2BatchPerDimensionGeneric
	hammingImpl                 = hammingGeneric
)

// Dot calculates the dot product of two vectors.
// Public for use by the distance package.
//
// SAFETY: This function assumes len(a) == len(b).
// It does NOT perform bounds checks for performance reasons.
// Callers MUST ensure lengths match to avoid buffer over-reads (especially with SIMD).
func Dot(a, b []float32) float32 {
	return dotImpl(a, b)
}

// DotBatch calculates dot products for a batch of vectors.
// targets is a flattened array of N vectors, each of dimension dim.
// out must have length N (len(targets) / dim).
func DotBatch(query []float32, targets []float32, dim int, out []float32) {
	dotBatchImpl(query, targets, dim, out)
}

func dotGeneric(a, b []float32) float32 {
	var ret float32
	for i := range a {
		ret += a[i] * b[i]
	}

	return ret
}

func dotBatchGeneric(query []float32, targets []float32, dim int, out []float32) {
	if dim <= 0 || len(out) == 0 {
		return
	}
	if len(query) < dim {
		return
	}

	q := query[:dim]
	maxVal := len(targets) / dim
	n := len(out)
	if maxVal < n {
		n = maxVal
	}

	for i := 0; i < n; i++ {
		offset := i * dim
		vec := targets[offset : offset+dim]
		out[i] = dotImpl(q, vec)
	}
}

// SquaredL2 calculates the squared L2 distance.
// Public for use by the distance package.
//
// SAFETY: This function assumes len(a) == len(b).
// It does NOT perform bounds checks for performance reasons.
// Callers MUST ensure lengths match to avoid buffer over-reads (especially with SIMD).
func SquaredL2(a, b []float32) float32 {
	return squaredL2Impl(a, b)
}

// SquaredL2Batch calculates squared L2 distance for a batch of vectors.
// targets is a flattened array of N vectors, each of dimension dim.
// out must have length N (len(targets) / dim).
func SquaredL2Batch(query []float32, targets []float32, dim int, out []float32) {
	squaredL2BatchImpl(query, targets, dim, out)
}

func squaredL2BatchGeneric(query []float32, targets []float32, dim int, out []float32) {
	if dim <= 0 || len(out) == 0 {
		return
	}
	if len(query) < dim {
		return
	}

	q := query[:dim]
	maxVal := len(targets) / dim
	n := len(out)
	if maxVal < n {
		n = maxVal
	}

	for i := 0; i < n; i++ {
		offset := i * dim
		vec := targets[offset : offset+dim]
		out[i] = squaredL2Impl(q, vec)
	}
}

// ScaleInPlace multiplies all elements of a by scalar.
//
// This is primarily used by distance normalization.
func ScaleInPlace(a []float32, scalar float32) {
	scaleImpl(a, scalar)
}

// PqAdcLookup computes the sum of distances from a precomputed table.
// table: M x 256 floats (flattened)
// codes: M bytes
// m: number of subvectors
func PqAdcLookup(table []float32, codes []byte, m int) float32 {
	return pqAdcImpl(table, codes, m)
}

func squaredL2Generic(a, b []float32) float32 {
	var distance float32
	for i := range a {
		distance += (a[i] - b[i]) * (a[i] - b[i])
	}

	return distance
}

func scaleGeneric(a []float32, scalar float32) {
	for i := range a {
		a[i] *= scalar
	}
}

func pqAdcLookupGeneric(table []float32, codes []byte, m int) float32 {
	var sum float32
	for i := range m {
		sum += table[i*256+int(codes[i])]
	}
	return sum
}

// Sq8uL2BatchPerDimension computes squared L2 distance between a float32 query and SQ8 (uint8) encoded vectors with per-dimension scaling.
func Sq8uL2BatchPerDimension(query []float32, codes []byte, mins []float32, invScales []float32, dim int, out []float32) {
	sq8uL2BatchPerDimensionImpl(query, codes, mins, invScales, dim, out)
}

// Hamming computes the Hamming distance between a and b.
func Hamming(a, b []byte) int64 {
	return hammingImpl(a, b)
}

func sq8uL2BatchPerDimensionGeneric(query []float32, codes []byte, mins []float32, invScales []float32, dim int, out []float32) {
	n := len(out)
	for i := range n {
		code := codes[i*dim : (i+1)*dim]
		var sum float32
		for j := range dim {
			val := mins[j] + float32(code[j])*invScales[j]
			diff := query[j] - val
			sum += diff * diff
		}
		out[i] = sum
	}
}

func hammingGeneric(a, b []byte) int64 {
	var sum int64
	n := len(a)
	for n >= 8 {
		v1 := binary.LittleEndian.Uint64(a)
		v2 := binary.LittleEndian.Uint64(b)
		sum += int64(bits.OnesCount64(v1 ^ v2))
		a = a[8:]
		b = b[8:]
		n -= 8
	}
	for i := range a {
		sum += int64(bits.OnesCount8(a[i] ^ b[i]))
	}
	return sum
}
