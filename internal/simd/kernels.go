package simd

import (
	"encoding/binary"
	"math/bits"
)

// Kernel function pointers - set once at init, zero runtime overhead.
// Generic implementations are the default; platform-specific init()
// functions override with SIMD versions when available.
var (
	kernelDot                       = dotGeneric
	kernelSquaredL2                 = squaredL2Generic
	kernelScale                     = scaleGeneric
	kernelPqAdc                     = pqAdcLookupGeneric
	kernelDotBatch                  = dotBatchGeneric
	kernelSquaredL2Batch            = squaredL2BatchGeneric
	kernelHamming                   = hammingGeneric
	kernelSQ8uL2BatchPerDim         = sq8uL2BatchPerDimensionGeneric
	kernelInt4L2Distance            = int4L2DistanceGeneric
	kernelInt4L2DistancePrecomputed = int4L2DistancePrecomputedGeneric
	kernelInt4L2DistanceBatch       = int4L2DistanceBatchGeneric
	kernelSquaredL2Int8Dequantized  = squaredL2Int8DequantizedGeneric
	kernelBuildDistanceTableInt8    = buildDistanceTableInt8Generic
	kernelFindNearestCentroidInt8   = findNearestCentroidInt8Generic
	kernelFilterRangeF64            = filterRangeF64Unrolled
	kernelFilterRangeF64Indices     = filterRangeF64IndicesGeneric
	kernelCountRangeF64             = countRangeF64Generic
	kernelGatherU32                 = gatherU32Generic
)

// ============================================================================
// Public API - Zero-overhead dispatch through function pointers
// ============================================================================

// Dot calculates the dot product of two vectors.
//
// SAFETY: Assumes len(a) == len(b). Caller MUST ensure lengths match.
func Dot(a, b []float32) float32 {
	return kernelDot(a, b)
}

// SquaredL2 calculates the squared L2 distance.
//
// SAFETY: Assumes len(a) == len(b). Caller MUST ensure lengths match.
func SquaredL2(a, b []float32) float32 {
	return kernelSquaredL2(a, b)
}

// ScaleInPlace multiplies all elements of a by scalar.
func ScaleInPlace(a []float32, scalar float32) {
	kernelScale(a, scalar)
}

// PqAdcLookup computes the sum of distances from a precomputed PQ table.
func PqAdcLookup(table []float32, codes []byte, m int) float32 {
	return kernelPqAdc(table, codes, m)
}

// DotBatch calculates dot products for a batch of vectors.
func DotBatch(query []float32, targets []float32, dim int, out []float32) {
	kernelDotBatch(query, targets, dim, out)
}

// SquaredL2Batch calculates squared L2 distance for a batch.
func SquaredL2Batch(query []float32, targets []float32, dim int, out []float32) {
	kernelSquaredL2Batch(query, targets, dim, out)
}

// Hamming computes the Hamming distance between two byte slices.
func Hamming(a, b []byte) int {
	return kernelHamming(a, b)
}

// Sq8uL2BatchPerDimension computes SQ8 L2 with per-dimension scaling.
func Sq8uL2BatchPerDimension(query []float32, codes []byte, mins, invScales []float32, dim int, out []float32) {
	kernelSQ8uL2BatchPerDim(query, codes, mins, invScales, dim, out)
}

// Int4L2Distance computes squared L2 between query and INT4 code.
func Int4L2Distance(query []float32, code []byte, minVal, diff []float32) float32 {
	return kernelInt4L2Distance(query, code, minVal, diff)
}

// Int4L2DistancePrecomputed uses precomputed lookup table.
func Int4L2DistancePrecomputed(query []float32, code []byte, lookupTable []float32) float32 {
	return kernelInt4L2DistancePrecomputed(query, code, lookupTable)
}

// BuildInt4LookupTable builds a lookup table for INT4 quantized distance calculations.
// For each dimension d and quantization level q (0..15), computes the dequantized value:
//
//	table[d*16 + q] = (float32(q) / 15.0) * diff[d] + minVal[d]
func BuildInt4LookupTable(minVal, diff []float32) []float32 {
	dim := len(minVal)
	table := make([]float32, dim*16)
	for d := 0; d < dim; d++ {
		for q := 0; q < 16; q++ {
			table[d*16+q] = (float32(q)/15.0)*diff[d] + minVal[d]
		}
	}
	return table
}

// Int4L2DistanceBatch computes INT4 L2 for multiple codes.
func Int4L2DistanceBatch(query []float32, codes []byte, dim, n int, minVal, diff []float32, out []float32) {
	kernelInt4L2DistanceBatch(query, codes, dim, n, minVal, diff, out)
}

// SquaredL2Int8Dequantized computes L2 for int8 quantized vectors.
func SquaredL2Int8Dequantized(query []float32, code []int8, scale, offset float32) float32 {
	return kernelSquaredL2Int8Dequantized(query, code, scale, offset)
}

// BuildDistanceTableInt8 builds a distance table for int8 codebook.
func BuildDistanceTableInt8(querySubvec []float32, codebook []int8, subdim int, scale, offset float32, out []float32) {
	kernelBuildDistanceTableInt8(querySubvec, codebook, subdim, scale, offset, out)
}

// FindNearestCentroidInt8 finds the nearest centroid in int8 codebook.
func FindNearestCentroidInt8(querySubvec []float32, codebook []int8, subdim int, scale, offset float32) int {
	return kernelFindNearestCentroidInt8(querySubvec, codebook, subdim, scale, offset)
}

// FilterRangeF64 returns a bitmask of indices where minVal <= values[i] <= maxVal.
func FilterRangeF64(values []float64, minVal, maxVal float64, dst []byte) []byte {
	n := len(values)
	if n == 0 {
		return dst[:0]
	}
	if cap(dst) < n {
		dst = make([]byte, n)
	} else {
		dst = dst[:n]
	}
	kernelFilterRangeF64(values, minVal, maxVal, dst)
	return dst
}

// FilterRangeF64Indices extracts indices of values in range.
// Returns the slice of indices that are in range (a subslice of the input indices buffer).
func FilterRangeF64Indices(values []float64, minVal, maxVal float64, indices []int32) []int32 {
	count := kernelFilterRangeF64Indices(values, minVal, maxVal, indices)
	return indices[:count]
}

// CountRangeF64 counts values in the given range [minVal, maxVal].
func CountRangeF64(values []float64, minVal, maxVal float64) int {
	return kernelCountRangeF64(values, minVal, maxVal)
}

// GatherU32 gathers uint32 values by indices.
// Returns dst for convenience.
func GatherU32(src []uint32, indices []int32, dst []uint32) []uint32 {
	kernelGatherU32(src, indices, dst)
	return dst[:len(indices)]
}

// ============================================================================
// Bounded Distance Functions (Early Exit Optimization)
// ============================================================================

// SquaredL2Bounded computes squared L2 distance with early exit when exceeding bound.
// Returns (distance, exceeded) where exceeded=true means distance > bound.
// This provides 10-20% speedup in HNSW traversal by avoiding full computation
// when a candidate is clearly worse than the current worst result.
//
// The early exit check is performed every 64 dimensions to balance:
// - Frequent checks: more early exits but branch overhead
// - Infrequent checks: less overhead but miss early exit opportunities
//
// For d=768, this gives ~12 check points, each after 64 dims (~8% of total work).
func SquaredL2Bounded(a, b []float32, bound float32) (float32, bool) {
	return kernelSquaredL2Bounded(a, b, bound)
}

// kernelSquaredL2Bounded is the function pointer for bounded L2.
// Platform-specific init() can override this with SIMD version.
var kernelSquaredL2Bounded = squaredL2BoundedGeneric

// squaredL2BoundedGeneric is the pure Go implementation with early exit.
func squaredL2BoundedGeneric(a, b []float32, bound float32) (float32, bool) {
	var distance float32
	n := len(a)

	// Process in blocks of 64 with early exit checks
	// 64 chosen to match cache line multiples and provide ~12 checks for d=768
	blockSize := 64
	i := 0

	for ; i+blockSize <= n; i += blockSize {
		// Unroll 8x for better ILP within block
		for j := i; j < i+blockSize; j += 8 {
			d0 := a[j] - b[j]
			d1 := a[j+1] - b[j+1]
			d2 := a[j+2] - b[j+2]
			d3 := a[j+3] - b[j+3]
			d4 := a[j+4] - b[j+4]
			d5 := a[j+5] - b[j+5]
			d6 := a[j+6] - b[j+6]
			d7 := a[j+7] - b[j+7]
			distance += d0*d0 + d1*d1 + d2*d2 + d3*d3 + d4*d4 + d5*d5 + d6*d6 + d7*d7
		}
		// Early exit check after each block
		if distance > bound {
			return distance, true
		}
	}

	// Scalar cleanup for remainder
	for ; i < n; i++ {
		d := a[i] - b[i]
		distance += d * d
	}

	return distance, distance > bound
}

// ============================================================================
// Generic implementations (pure Go fallbacks)
// ============================================================================

func dotGeneric(a, b []float32) float32 {
	var ret float32
	for i := range a {
		ret += a[i] * b[i]
	}
	return ret
}

func squaredL2Generic(a, b []float32) float32 {
	var distance float32
	for i := range a {
		d := a[i] - b[i]
		distance += d * d
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

func dotBatchGeneric(query []float32, targets []float32, dim int, out []float32) {
	if dim <= 0 || len(out) == 0 || len(query) < dim {
		return
	}
	q := query[:dim]
	n := min(len(out), len(targets)/dim)
	for i := 0; i < n; i++ {
		offset := i * dim
		out[i] = dotGeneric(q, targets[offset:offset+dim])
	}
}

func squaredL2BatchGeneric(query []float32, targets []float32, dim int, out []float32) {
	if dim <= 0 || len(out) == 0 || len(query) < dim {
		return
	}
	q := query[:dim]
	n := min(len(out), len(targets)/dim)
	for i := 0; i < n; i++ {
		offset := i * dim
		out[i] = squaredL2Generic(q, targets[offset:offset+dim])
	}
}

func hammingGeneric(a, b []byte) int {
	total := 0
	i := 0
	for ; i+8 <= len(a); i += 8 {
		v1 := binary.LittleEndian.Uint64(a[i:])
		v2 := binary.LittleEndian.Uint64(b[i:])
		total += bits.OnesCount64(v1 ^ v2)
	}
	for ; i < len(a); i++ {
		total += bits.OnesCount8(a[i] ^ b[i])
	}
	return total
}

func sq8uL2BatchPerDimensionGeneric(query []float32, codes []byte, mins, invScales []float32, dim int, out []float32) {
	n := len(out)
	for i := 0; i < n; i++ {
		var sum float32
		for d := 0; d < dim; d++ {
			c := float32(codes[i*dim+d])
			dequant := mins[d] + c*invScales[d]
			diff := query[d] - dequant
			sum += diff * diff
		}
		out[i] = sum
	}
}

func int4L2DistanceGeneric(query []float32, code []byte, minVal, diff []float32) float32 {
	var sum float32
	dim := len(query)
	for i := 0; i < dim; i += 2 {
		byteVal := code[i/2]
		quant1 := (byteVal >> 4) & 0x0F
		val1 := float32(quant1)/15.0*diff[i] + minVal[i]
		d1 := val1 - query[i]
		sum += d1 * d1

		if i+1 < dim {
			quant2 := byteVal & 0x0F
			val2 := float32(quant2)/15.0*diff[i+1] + minVal[i+1]
			d2 := val2 - query[i+1]
			sum += d2 * d2
		}
	}
	return sum
}

func int4L2DistancePrecomputedGeneric(query []float32, code []byte, lookupTable []float32) float32 {
	var sum float32
	dim := len(query)
	for i := 0; i < dim; i += 2 {
		byteVal := code[i/2]
		quant1 := int((byteVal >> 4) & 0x0F)
		val1 := lookupTable[i*16+quant1]
		d1 := val1 - query[i]
		sum += d1 * d1

		if i+1 < dim {
			quant2 := int(byteVal & 0x0F)
			val2 := lookupTable[(i+1)*16+quant2]
			d2 := val2 - query[i+1]
			sum += d2 * d2
		}
	}
	return sum
}

func int4L2DistanceBatchGeneric(query []float32, codes []byte, dim, n int, minVal, diff []float32, out []float32) {
	codeSize := (dim + 1) / 2
	for j := 0; j < n; j++ {
		code := codes[j*codeSize : (j+1)*codeSize]
		out[j] = int4L2DistanceGeneric(query, code, minVal, diff)
	}
}

func squaredL2Int8DequantizedGeneric(query []float32, code []int8, scale, offset float32) float32 {
	var sum float32
	for i := 0; i < len(query); i++ {
		v := float32(code[i])*scale + offset
		d := query[i] - v
		sum += d * d
	}
	return sum
}

func buildDistanceTableInt8Generic(querySubvec []float32, codebook []int8, subdim int, scale, offset float32, out []float32) {
	if subdim <= 0 {
		return
	}
	k := len(out)
	for ci := 0; ci < k; ci++ {
		start := ci * subdim
		end := start + subdim
		out[ci] = squaredL2Int8DequantizedGeneric(querySubvec, codebook[start:end], scale, offset)
	}
}

func findNearestCentroidInt8Generic(querySubvec []float32, codebook []int8, subdim int, scale, offset float32) int {
	if subdim <= 0 {
		return 0
	}
	k := len(codebook) / subdim
	if k <= 0 {
		return 0
	}

	bestIdx := 0
	bestDist := squaredL2Int8DequantizedGeneric(querySubvec, codebook[:subdim], scale, offset)
	for ci := 1; ci < k; ci++ {
		start := ci * subdim
		d := squaredL2Int8DequantizedGeneric(querySubvec, codebook[start:start+subdim], scale, offset)
		if d < bestDist {
			bestDist = d
			bestIdx = ci
		}
	}
	return bestIdx
}

func filterRangeF64Unrolled(values []float64, minVal, maxVal float64, dst []byte) {
	n := len(values)
	i := 0
	for ; i+8 <= n; i += 8 {
		v0, v1, v2, v3 := values[i], values[i+1], values[i+2], values[i+3]
		v4, v5, v6, v7 := values[i+4], values[i+5], values[i+6], values[i+7]
		dst[i] = boolToByte(v0 >= minVal && v0 <= maxVal)
		dst[i+1] = boolToByte(v1 >= minVal && v1 <= maxVal)
		dst[i+2] = boolToByte(v2 >= minVal && v2 <= maxVal)
		dst[i+3] = boolToByte(v3 >= minVal && v3 <= maxVal)
		dst[i+4] = boolToByte(v4 >= minVal && v4 <= maxVal)
		dst[i+5] = boolToByte(v5 >= minVal && v5 <= maxVal)
		dst[i+6] = boolToByte(v6 >= minVal && v6 <= maxVal)
		dst[i+7] = boolToByte(v7 >= minVal && v7 <= maxVal)
	}
	for ; i < n; i++ {
		dst[i] = boolToByte(values[i] >= minVal && values[i] <= maxVal)
	}
}

func boolToByte(b bool) byte {
	if b {
		return 1
	}
	return 0
}

func filterRangeF64IndicesGeneric(values []float64, minVal, maxVal float64, indices []int32) int {
	count := 0
	for i, v := range values {
		if v >= minVal && v <= maxVal {
			if count < len(indices) {
				indices[count] = int32(i)
			}
			count++
		}
	}
	return count
}

func countRangeF64Generic(values []float64, minVal, maxVal float64) int {
	count := 0
	for _, v := range values {
		if v >= minVal && v <= maxVal {
			count++
		}
	}
	return count
}

func gatherU32Generic(src []uint32, indices []int32, dst []uint32) {
	for i, idx := range indices {
		if int(idx) < len(src) {
			dst[i] = src[idx]
		}
	}
}
