package simd

// This file provides fused helpers around int8-quantized codebooks.
//
// These are intentionally written to avoid scratch allocations in hot loops.
// Architecture-specific SIMD implementations can override the impl variables
// in arch-tagged files later.

// squaredL2Int8DequantizedGeneric computes:
//
//	sum_i (query[i] - (float32(code[i])*scale + offset))^2
//
// Assumes len(query) == len(code). Caller's responsibility.
func squaredL2Int8DequantizedGeneric(query []float32, code []int8, scale, offset float32) float32 {
	var sum float32
	for i := 0; i < len(query); i++ {
		v := float32(code[i])*scale + offset
		d := query[i] - v
		sum += d * d
	}
	return sum
}

var squaredL2Int8DequantizedImpl = squaredL2Int8DequantizedGeneric

// SquaredL2Int8Dequantized computes the squared L2 distance between query and a
// dequantized int8 vector (code*scale + offset).
//
// Assumes len(query) == len(code). Caller's responsibility.
func SquaredL2Int8Dequantized(query []float32, code []int8, scale, offset float32) float32 {
	return squaredL2Int8DequantizedImpl(query, code, scale, offset)
}

// buildDistanceTableInt8Generic fills out[k] with the squared distance between
// querySubvec and centroid k from the given flat int8 codebook.
//
// codebook is expected to be laid out as K contiguous subvectors:
//
//	codebook[k*subdim : (k+1)*subdim]
//
// out must have length K.
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

var buildDistanceTableInt8Impl = buildDistanceTableInt8Generic

// BuildDistanceTableInt8 fills out with distances from querySubvec to each
// centroid in the int8 codebook.
func BuildDistanceTableInt8(querySubvec []float32, codebook []int8, subdim int, scale, offset float32, out []float32) {
	buildDistanceTableInt8Impl(querySubvec, codebook, subdim, scale, offset, out)
}

// findNearestCentroidInt8Generic returns argmin_k distance(querySubvec, centroid_k).
//
// codebook is laid out as in BuildDistanceTableInt8.
func findNearestCentroidInt8Generic(querySubvec []float32, codebook []int8, subdim int, scale, offset float32) int {
	if subdim <= 0 {
		return 0
	}
	k := len(codebook) / subdim
	if k <= 0 {
		return 0
	}

	bestIdx := 0
	var bestDist float32
	{
		c0 := codebook[:subdim]
		bestDist = squaredL2Int8DequantizedGeneric(querySubvec, c0, scale, offset)
	}

	for ci := 1; ci < k; ci++ {
		start := ci * subdim
		end := start + subdim
		d := squaredL2Int8DequantizedGeneric(querySubvec, codebook[start:end], scale, offset)
		if d < bestDist {
			bestDist = d
			bestIdx = ci
		}
	}
	return bestIdx
}

var findNearestCentroidInt8Impl = findNearestCentroidInt8Generic

// FindNearestCentroidInt8 returns the index of the nearest centroid in an int8
// codebook to the given query subvector.
func FindNearestCentroidInt8(querySubvec []float32, codebook []int8, subdim int, scale, offset float32) int {
	return findNearestCentroidInt8Impl(querySubvec, codebook, subdim, scale, offset)
}
