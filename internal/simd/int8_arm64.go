//go:build arm64 && !noasm

package simd

import "unsafe"

func init() {
	// NEON is mandatory on arm64.
	squaredL2Int8DequantizedImpl = squaredL2Int8DequantizedNEON
	buildDistanceTableInt8Impl = buildDistanceTableInt8NEON
	findNearestCentroidInt8Impl = findNearestCentroidInt8NEON
}

func squaredL2Int8DequantizedNEON(query []float32, code []int8, scale, offset float32) float32 {
	var ret float32
	if len(query) > 0 {
		squaredL2Int8DequantizedNeon(
			unsafe.Pointer(&query[0]),
			unsafe.Pointer(&code[0]),
			int64(len(query)),
			unsafe.Pointer(&scale),
			unsafe.Pointer(&offset),
			unsafe.Pointer(&ret),
		)
	}
	return ret
}

func buildDistanceTableInt8NEON(querySubvec []float32, codebook []int8, subdim int, scale, offset float32, out []float32) {
	if subdim > 0 {
		buildDistanceTableInt8Neon(
			unsafe.Pointer(&querySubvec[0]),
			unsafe.Pointer(&codebook[0]),
			int64(subdim),
			unsafe.Pointer(&scale),
			unsafe.Pointer(&offset),
			unsafe.Pointer(&out[0]),
		)
	}
}

func findNearestCentroidInt8NEON(querySubvec []float32, codebook []int8, subdim int, scale, offset float32) int {
	var idx int64
	if subdim > 0 {
		findNearestCentroidInt8Neon(
			unsafe.Pointer(&querySubvec[0]),
			unsafe.Pointer(&codebook[0]),
			int64(subdim),
			unsafe.Pointer(&scale),
			unsafe.Pointer(&offset),
			unsafe.Pointer(&idx),
		)
	}
	return int(idx)
}
